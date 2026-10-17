"""分页 KV 块管理器（T4-B）。

语义移植自 vLLM v1 的 `BlockPool` / `KVCacheManager`（Apache-2.0，
<https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py>），
按「零重写」原则只移植**块层语义**，不移植 GPU 张量与前缀缓存哈希：

* 物理块池 + `block_tables` 间接层（逻辑位置 → 物理块 id）；
* **引用计数**：`free_seq` 时 `ref_cnt -= 1`，仅当降到 0 才回到空闲队列
  （同一块被多条序列共享时不会提前回收）；
* 空闲队列对**非缓存块 LIFO**（freed 块 prepend，分配 popleft，提升局部性），
  缓存块 FIFO（LRU 淘汰）——本项目暂未启用前缀缓存，故当前等价纯 LIFO；
* `null_block`：占位块（block_id=0），构造时即从空闲队列摘出，不参与分配、
  不参与引用计数、不计入使用率；`block_tables` 中未分配槽位一律填它。

滑动窗口（T4-C 之前的简易驱逐）：每条序列记 `head_offset` = 首块内已废弃的
slot 数。逻辑位置 p 的存储位置是 `head_offset + p`，因此丢弃任意数量 token
都只需在 `head_offset` 累加满一个块时回收整块——**逻辑→物理映射永不错位**。

本模块是**纯簿记**（不碰张量数据），可无 numpy 下单测；
物理存储在 `phosphorus/kv_cache.py`。
"""

from __future__ import annotations

from collections import deque

import numpy as np

NULL_BLOCK = 0


class BlockManager:
    """固定容量的分页块管理器：`n_seqs` 条序列 × `max_blocks_per_seq` 块。"""

    def __init__(
        self, n_blocks: int, block_size: int, n_seqs: int, max_blocks_per_seq: int
    ) -> None:
        if n_blocks < 2:
            raise ValueError(f"n_blocks 必须 >= 2（含 null_block），收到 {n_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size 必须 >= 1，收到 {block_size}")
        if max_blocks_per_seq < 1:
            raise ValueError(f"max_blocks_per_seq 必须 >= 1，收到 {max_blocks_per_seq}")
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.n_seqs = int(n_seqs)
        self.max_blocks_per_seq = int(max_blocks_per_seq)

        # 物理块引用计数（null_block 永不维护）
        self.ref_cnt = np.zeros(self.n_blocks, dtype=np.int32)
        # 逻辑→物理：block_tables[s, j] = 第 j 个块的物理 id；未分配槽位 = NULL_BLOCK
        self.block_tables = np.full(
            (self.n_seqs, self.max_blocks_per_seq), NULL_BLOCK, dtype=np.int32
        )
        # 每条序列已占用的有效 token 数
        self.seq_lens = np.zeros(self.n_seqs, dtype=np.int32)
        # 每条序列首块内已废弃的 slot 数（0 <= head_offset < block_size）
        self.head_offset = np.zeros(self.n_seqs, dtype=np.int32)
        # 每条序列累计丢弃的 token 数（RoPE 位置重基准用）
        self.dropped = np.zeros(self.n_seqs, dtype=np.int32)

        # 空闲队列：除 null_block 外的全部块（vLLM: free_block_queue 初始含全部，再 popleft 出 null）
        self._free: deque[int] = deque(range(1, self.n_blocks))
        self.null_block = NULL_BLOCK

    # ----------------------------- 查询 ----------------------------- #
    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_used_blocks(self) -> int:
        """已分配块数（不含 null_block，与 vLLM get_usage 的 "subtract 1" 一致）。"""
        return (self.n_blocks - 1) - len(self._free)

    def utilization(self) -> float:
        """物理池使用率 ∈ [0, 1]（分母排除 null_block）。"""
        return self.num_used_blocks() / (self.n_blocks - 1)

    def seq_len(self, seq_id: int) -> int:
        return int(self.seq_lens[seq_id])

    def n_blocks_of(self, seq_id: int) -> int:
        """覆盖存储区间 [head_offset, head_offset + seq_len) 所需的块数。"""
        end = int(self.head_offset[seq_id]) + int(self.seq_lens[seq_id])
        return (end + self.block_size - 1) // self.block_size

    def block_table(self, seq_id: int) -> np.ndarray:
        """该序列的有效块表（int32 视图，长度 = n_blocks_of）。"""
        return self.block_tables[seq_id, : self.n_blocks_of(seq_id)]

    # ----------------------------- 分配 ----------------------------- #
    def _blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        end = int(self.head_offset[seq_id]) + int(self.seq_lens[seq_id])
        used = (end + self.block_size - 1) // self.block_size
        need_total = (end + int(num_new_tokens) + self.block_size - 1) // self.block_size
        return need_total - used

    def can_append(self, seq_id: int, num_new_tokens: int) -> bool:
        """判断是否还能追加 num_new_tokens 个 token（块数 + 表容量双约束）。"""
        need = self._blocks_needed(seq_id, num_new_tokens)
        return (
            need <= len(self._free) and self.n_blocks_of(seq_id) + need <= self.max_blocks_per_seq
        )

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> bool:
        """为 seq_id 追加 num_new_tokens 个 token 所需的物理块。

        成功返回 True；空间不足返回 False（不修改任何状态）。
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens 必须 >= 0，收到 {num_new_tokens}")
        need = self._blocks_needed(seq_id, num_new_tokens)
        used = self.n_blocks_of(seq_id)
        if need > len(self._free) or used + need > self.max_blocks_per_seq:
            return False
        for j in range(used, used + need):
            bid = self._free.popleft()
            assert self.ref_cnt[bid] == 0, "空闲块引用计数必须为 0（vLLM 不变式）"
            self.ref_cnt[bid] = 1
            self.block_tables[seq_id, j] = bid
        self.seq_lens[seq_id] += num_new_tokens
        return True

    # ----------------------------- 释放 ----------------------------- #
    def release_block(self, block_id: int) -> None:
        """单块引用计数 -1；降到 0 时 prepend 回空闲队列（LIFO 复用）。"""
        if block_id == self.null_block:
            return
        self.ref_cnt[block_id] -= 1
        if self.ref_cnt[block_id] == 0:
            self._free.appendleft(block_id)
        elif self.ref_cnt[block_id] < 0:
            raise RuntimeError(f"块 {block_id} 引用计数负数：重复释放")

    def free_seq(self, seq_id: int) -> None:
        """释放整条序列的全部块（引用计数语义与 vLLM free_blocks 一致）。"""
        for j in range(self.n_blocks_of(seq_id)):
            self.release_block(int(self.block_tables[seq_id, j]))
            self.block_tables[seq_id, j] = NULL_BLOCK
        self.seq_lens[seq_id] = 0
        self.head_offset[seq_id] = 0
        self.dropped[seq_id] = 0

    def drop_tokens(self, seq_id: int, num_tokens: int) -> int:
        """滑动窗口：丢弃序列最旧的 num_tokens 个 token，返回实际丢弃数。

        只做整块回收 + `head_offset` 累加，逻辑→物理映射永不错位；
        被丢块的 KV 数据无需搬运（vLLM 滑动窗口后端同款做法）。
        """
        n = int(self.seq_lens[seq_id])
        num_tokens = min(int(num_tokens), n)
        if num_tokens <= 0:
            return 0
        off = int(self.head_offset[seq_id]) + num_tokens
        drop_blocks, new_off = divmod(off, self.block_size)
        nblk = self.n_blocks_of(seq_id)
        if drop_blocks:
            for j in range(drop_blocks):
                self.release_block(int(self.block_tables[seq_id, j]))
                self.block_tables[seq_id, j] = NULL_BLOCK
            self.block_tables[seq_id, : nblk - drop_blocks] = self.block_tables[
                seq_id, drop_blocks:nblk
            ]
            self.block_tables[seq_id, nblk - drop_blocks :] = NULL_BLOCK
        self.head_offset[seq_id] = new_off
        self.seq_lens[seq_id] = n - num_tokens
        self.dropped[seq_id] += num_tokens
        return num_tokens

    def truncate_tail(self, seq_id: int, num_tokens: int) -> int:
        """丢弃序列**末尾** num_tokens 个 token（投机解码 reject 回滚用），返回实际数。

        与 drop_tokens 互为对称：只释放尾部整块、收缩 `seq_len`，逻辑→物理映射
        永不错位；被释放块的 KV 数据无需搬运。head_offset 不变（首块槽位保留）。
        """
        n = int(self.seq_lens[seq_id])
        num_tokens = int(min(int(num_tokens), n))
        if num_tokens <= 0:
            return 0
        end = int(self.head_offset[seq_id]) + int(self.seq_lens[seq_id])
        new_len = int(self.seq_lens[seq_id]) - num_tokens
        new_end = int(self.head_offset[seq_id]) + new_len
        old_blocks = (end + self.block_size - 1) // self.block_size
        new_blocks = (new_end + self.block_size - 1) // self.block_size
        for j in range(new_blocks, old_blocks):
            self.release_block(int(self.block_tables[seq_id, j]))
            self.block_tables[seq_id, j] = NULL_BLOCK
        self.seq_lens[seq_id] = new_len
        return num_tokens

    # ----------------------------- 共享（前缀复用预留） ----------------------------- #
    def touch(self, seq_id: int, n_blocks: int) -> None:
        """把前 n_blocks 个块的引用计数 +1（vLLM 前缀缓存命中路径预留）。

        ref_cnt == 0 的块意味着它还在空闲队列里，先摘出再加一。
        """
        for j in range(n_blocks):
            bid = int(self.block_tables[seq_id, j])
            if bid == self.null_block:
                break
            if self.ref_cnt[bid] == 0:
                self._free.remove(bid)
            self.ref_cnt[bid] += 1

    def reset(self) -> None:
        """全部归还（等价于逐条 free_seq，但 O(n_blocks)）。"""
        self._free.clear()
        self._free.extend(range(1, self.n_blocks))
        self.ref_cnt[:] = 0
        self.block_tables[:] = NULL_BLOCK
        self.seq_lens[:] = 0
        self.head_offset[:] = 0
        self.dropped[:] = 0
