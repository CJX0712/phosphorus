"""分页 KV 物理池（T4-B）。

布局（与 ARCHITECTURE.md §2.1 / TASKS.md C4 一致）::

    kv[n_blocks, n_layer, 2, block_size, n_kv_head, d_head]
        第 3 维 2 = {K, V}

一个物理块装 `block_size` 个 token 在**全部层**上的 K/V；逻辑位置 → 物理块
的映射在 `phosphorus/block_manager.py` 的 `block_tables` 里（含 `head_offset`
滑动窗口偏移，见该模块文档）。

只存 `n_kv_head` 份（GQA 的 KV 头），查询时再 `np.repeat` 到 `n_head` ——
与 dense 参考实现逐元素等价（repeat 是精确拷贝），但内存省 `n_rep` 倍。
int8 KV 量化（dtype=uint8 + 每 token scale）为后续可选特性，当前默认 fp32。

写入协议（三段式，避免「预留」与「写入」的位置歧义）::

    cache.begin_append(seq_id, T)      # 预留块（manager.append_tokens）
    for layer in range(n_layer):
        cache.write(seq_id, layer, k, v)
    cache.end_append(seq_id)

`append()` 是单层便利封装（begin + write + end），供测试与一次性写入使用。
"""

from __future__ import annotations

import numpy as np

from .block_manager import BlockManager


class PagedKVCache:
    """分页 KV 池：begin/append 写新 token，gather 读某序列全部有效 KV。"""

    def __init__(
        self,
        manager: BlockManager,
        n_layer: int,
        n_kv_head: int,
        d_head: int,
        n_seqs: int = 1,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.mgr = manager
        self.n_layer = int(n_layer)
        self.n_kv_head = int(n_kv_head)
        self.d_head = int(d_head)
        self.n_seqs = int(n_seqs)
        self.dtype = np.dtype(dtype)
        self.kv = np.zeros(
            (manager.n_blocks, self.n_layer, 2, manager.block_size, self.n_kv_head, self.d_head),
            dtype=self.dtype,
        )
        # 逐序列 × 逐层的「有效 token 数」（逐层预算可不同，见 T4-C 驱逐）。
        # 这是连续批处理（T4-D）的关键：不同序列可有不同长度，KV 写入起点
        # 必须按「该序列自己的长度」计算，而非全局共享。
        # `mgr.seq_len[seq_id]` 取各层最大值用于块表分配，读取以 layer_len 为准。
        self.layer_len = np.zeros((self.n_seqs, self.n_layer), dtype=np.int32)
        # 每条序列「已预留、尚未 end」的批次大小
        self._pending: dict[int, int] = {}

    # ----------------------------- 属性 ----------------------------- #
    @property
    def block_size(self) -> int:
        return self.mgr.block_size

    def nbytes(self) -> int:
        return int(self.kv.nbytes)

    def reset(self) -> None:
        self.mgr.reset()
        self.kv[:] = 0
        self.layer_len[:] = 0
        self._pending.clear()

    def length(self, seq_id: int, layer: int) -> int:
        """某序列某层的有效 KV token 数。"""
        return int(self.layer_len[seq_id, layer])

    # ----------------------------- 写 ----------------------------- #
    def can_append(self, seq_id: int, num_new_tokens: int) -> bool:
        return self.mgr.can_append(seq_id, num_new_tokens)

    def begin_append(self, seq_id: int, num_new_tokens: int) -> bool:
        """为一批 T 个新 token 预留物理块；空间不足返回 False（不改状态）。"""
        if num_new_tokens <= 0:
            return True
        if seq_id in self._pending:
            raise RuntimeError(f"seq {seq_id} 上一个批次尚未 end_append")
        if not self.mgr.append_tokens(seq_id, num_new_tokens):
            return False
        self._pending[seq_id] = int(num_new_tokens)
        return True

    def end_append(self, seq_id: int) -> None:
        self._pending.pop(seq_id, None)

    def write(self, seq_id: int, layer: int, k_new: np.ndarray, v_new: np.ndarray) -> None:
        """把 layer 层新算出的 K/V 写入当前批次（自动跨块，全向量化）。

        k_new / v_new: [T, n_kv_head, d_head]，T 必须与 begin_append 一致。
        起点 = head_offset + 该层当前有效长度（驱逐后各层长度可能不同，但都正确）。
        """
        T = int(k_new.shape[0])
        if T == 0:
            return
        pending = self._pending.get(seq_id)
        if pending is None:
            raise RuntimeError(f"seq {seq_id} 没有活动批次——请先 begin_append")
        if pending != T:
            raise RuntimeError(f"seq {seq_id} 批次大小不一致：begin={pending}, write={T}")
        # 本批次起点 = 首块偏移 + 该序列该层已存长度（begin_append 已把
        # mgr.seq_len 增加 T，但 layer_len[seq_id, layer] 仍是「写入前」的值，
        # 普通追加时二者相等；多序列时各自独立，绝不共享全局长度）
        start = int(self.mgr.head_offset[seq_id]) + int(self.layer_len[seq_id, layer])
        self._write_at(seq_id, layer, k_new, v_new, start)
        self.layer_len[seq_id, layer] += T

    def _write_at(
        self,
        seq_id: int,
        layer: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
        start: int = 0,
    ) -> None:
        """把 [T, n_kv_head, d_head] 写到逻辑位置 start..start+T-1（供 compact 复用）。"""
        T = int(k_new.shape[0])
        if T == 0:
            return
        if k_new.shape[0] != v_new.shape[0]:
            raise ValueError("k / v 首维长度不一致")
        tbl = self.mgr.block_table(seq_id)
        pos = np.arange(start, start + T)
        blk = pos // self.block_size
        off = pos % self.block_size
        if int(blk[-1]) >= tbl.shape[0]:
            raise RuntimeError(f"seq {seq_id} 写入越界（簿记与写入不一致）")
        phys = tbl[blk]
        self.kv[phys, layer, 0, off] = k_new
        self.kv[phys, layer, 1, off] = v_new

    def append(self, seq_id: int, layer: int, k_new: np.ndarray, v_new: np.ndarray) -> bool:
        """单层便利封装：begin + write + end。空间不足返回 False。"""
        T = int(k_new.shape[0])
        if T == 0:
            return True
        if not self.begin_append(seq_id, T):
            return False
        self.write(seq_id, layer, k_new, v_new)
        self.end_append(seq_id)
        return True

    # ----------------------------- 读 ----------------------------- #
    def gather(self, seq_id: int, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """读出 seq_id 在 layer 层的全部有效 K/V（各 [T, n_kv_head, d_head]）。

        `T` 取该序列该层的 `layer_len[seq_id, layer]`（驱逐后各层长度可不同；
        块表按最大长度分配）。
        """
        n = int(self.layer_len[seq_id, layer])
        if n == 0:
            z = np.zeros((0, self.n_kv_head, self.d_head), dtype=self.dtype)
            return z, z.copy()
        start = int(self.mgr.head_offset[seq_id])
        tbl = self.mgr.block_table(seq_id)
        pos = np.arange(start, start + n)
        blk = pos // self.block_size
        off = pos % self.block_size
        phys = tbl[blk]
        k = self.kv[phys, layer, 0, off]
        v = self.kv[phys, layer, 1, off]
        return k, v

    def gather_heads(self, seq_id: int, layer: int, n_rep: int) -> tuple[np.ndarray, np.ndarray]:
        """gather + GQA 展开，各返回 [1, n_head, T, d_head]（dense 参考同构）。

        展开顺序与 engine 的 dense 路径一致：`np.repeat(axis=1)`，
        即 query 头 h 对应 KV 头 `h // n_rep`。
        """
        k, v = self.gather(seq_id, layer)  # 各 [T, n_kv, Dh]
        k = np.repeat(np.transpose(k, (1, 0, 2))[None], n_rep, axis=1)  # [1, H, T, Dh]
        v = np.repeat(np.transpose(v, (1, 0, 2))[None], n_rep, axis=1)
        return k, v

    # ----------------------------- 驱逐 ----------------------------- #
    def drop_front(self, seq_id: int, n: int) -> int:
        """滑动窗口式丢弃最旧 n 个 token（head_offset 机制），同步 layer_len。

        `mgr.drop_tokens` 只维护物理簿记（seq_len / head_offset / 引用计数），
        逐层有效长度 `layer_len` 需同步递减，否则二者错位导致读取越界。
        """
        n = self.mgr.drop_tokens(seq_id, n)
        for li in range(self.n_layer):
            self.layer_len[seq_id, li] = max(0, int(self.layer_len[seq_id, li]) - n)
        return n

    def compact(self, seq_id: int, keep_masks: np.ndarray) -> int:
        """按保留掩码压缩序列（T4-C 驱逐入口），逐层预算可不同。

        keep_masks: [n_layer, n] 或 [n] bool（后者对所有层相同）。
        压缩后：`layer_len[l]` = 该层保留数；`mgr.seq_len` = max(layer_len)
        （块表按最大值分配，各层只用自己前缀）；`head_offset` 归零、逻辑坐标
        重排到 [0, layer_len)。返回序列级丢弃数（n - max(layer_len)）。

        重要：必须先整体拷出保留 K/V 再 free——释放后 LIFO 会拿回同一批物理块，
        若原地搬运不同层会互相覆盖（跨块 `np.repeat` 同样受益）。
        """
        m = np.asarray(keep_masks, dtype=bool)
        n = int(self.mgr.seq_len(seq_id))
        if m.ndim == 1:
            if m.shape[0] != n:
                raise ValueError(f"keep_mask 形状应为 ({n},)，收到 {m.shape}")
            m = np.repeat(m[None], self.n_layer, axis=0)
        if m.ndim != 2 or m.shape != (self.n_layer, n):
            raise ValueError(f"keep_masks 形状应为 ({self.n_layer}, {n})，收到 {m.shape}")
        kept = m.sum(axis=1)
        n_new = int(kept.max())
        if n_new == n and bool((kept == n).all()):
            return 0
        if n_new <= 0:
            raise ValueError("保留掩码全 False：序列至少保留 1 个 token")
        # 1) 逐层拷出保留部分
        kb: list[np.ndarray | None] = [None] * self.n_layer
        vb: list[np.ndarray | None] = [None] * self.n_layer
        for li in range(self.n_layer):
            kl = int(self.layer_len[seq_id, li])
            k, v = self.gather(seq_id, li)  # [kl, n_kv, Dh]
            mk = m[li, :kl]
            kb[li] = np.ascontiguousarray(k[mk])
            vb[li] = np.ascontiguousarray(v[mk])
        # 2) 归还全部物理块（LIFO 回收，重置 head_offset / seq_len）
        self.mgr.free_seq(seq_id)
        # 3) 按最大保留数重建块表，各层只写自己的前缀
        if not self.mgr.append_tokens(seq_id, n_new):
            raise RuntimeError(f"压缩后重建失败：需要 {n_new} token 的物理块空间")
        for li in range(self.n_layer):
            self._write_at(seq_id, li, kb[li], vb[li], 0)  # 写到 [0, kept_l)
        self.layer_len[seq_id, :] = kept
        return n - n_new
