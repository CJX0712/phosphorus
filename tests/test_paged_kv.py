"""T4-B：分页 KV + 块管理器 + GQA 注意力（C4 验收）。

不变量：
* ⑨ 分页注意力与 dense 参考最大绝对差 < 1e-5（且与逐元素朴素实现一致）
* ⑩ BlockManager 分配/回收 10,000 次后 utilization() 精确回到 0（无泄漏）
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phosphorus.attention import gqa_attention, paged_gqa_attention
from phosphorus.block_manager import NULL_BLOCK, BlockManager
from phosphorus.kv_cache import PagedKVCache


# --------------------------------------------------------------------------- #
# 独立朴素参考（逐元素循环，刻意不复用被测代码路径）                              #
# --------------------------------------------------------------------------- #
def naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, q_start: int) -> np.ndarray:
    """q/k/v: [1, H, T/S, Dh] -> [1, H, T, Dh]，纯 Python 逐元素实现。"""
    H, T, Dh = q.shape[1], q.shape[2], q.shape[3]
    S = k.shape[2]
    out = np.zeros_like(q)
    for h in range(H):
        for i in range(T):
            s = np.empty(S, dtype=np.float64)
            for j in range(S):
                s[j] = float(
                    np.dot(q[0, h, i].astype(np.float64), k[0, h, j].astype(np.float64))
                ) / math.sqrt(Dh)
                if j > q_start + i:
                    s[j] = -np.inf
            s = s - np.max(s)
            e = np.exp(s)
            e = e / np.sum(e)
            out[0, h, i] = sum(e[j] * v[0, h, j] for j in range(S)).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# BlockManager                                                                  #
# --------------------------------------------------------------------------- #
class TestBlockManager:
    def test_alloc_free_roundtrip(self):
        mgr = BlockManager(n_blocks=10, block_size=4, n_seqs=2, max_blocks_per_seq=4)
        assert mgr.utilization() == 0.0
        assert mgr.num_free_blocks() == 9  # 排除 null_block
        assert mgr.append_tokens(0, 10) is True  # ceil(10/4) = 3 块
        assert mgr.seq_len(0) == 10
        assert mgr.utilization() == pytest.approx(3 / 9)
        mgr.free_seq(0)
        assert mgr.seq_len(0) == 0
        assert mgr.utilization() == 0.0
        assert np.all(mgr.block_tables[0] == NULL_BLOCK)

    def test_capacity_limit(self):
        mgr = BlockManager(n_blocks=4, block_size=4, n_seqs=1, max_blocks_per_seq=2)
        assert mgr.append_tokens(0, 8) is True  # 恰好 2 块（含 null 共 3 块，够）
        assert mgr.append_tokens(0, 1) is False  # 超 max_blocks_per_seq
        assert mgr.seq_len(0) == 8  # 失败不修改状态

    def test_pool_exhaustion(self):
        mgr = BlockManager(n_blocks=3, block_size=4, n_seqs=2, max_blocks_per_seq=4)
        assert mgr.append_tokens(0, 8) is True  # 用掉 2 块（池共 2 可用）
        assert mgr.append_tokens(1, 4) is False  # 池空
        assert mgr.seq_len(1) == 0

    def test_alloc_10k_no_leak(self):
        """C4：分配/回收 10,000 次无泄漏，utilization 精确回 0。"""
        mgr = BlockManager(n_blocks=64, block_size=16, n_seqs=8, max_blocks_per_seq=4)
        rng = np.random.default_rng(0)
        for it in range(10_000):
            s = int(rng.integers(0, 8))
            mgr.append_tokens(s, int(rng.integers(1, 65)))
            if it % 3 == 0:
                mgr.free_seq(s)
        for s in range(8):
            mgr.free_seq(s)
        assert mgr.num_free_blocks() == 63
        assert mgr.utilization() == 0.0
        assert int(np.sum(mgr.ref_cnt)) == 0

    def test_sliding_window_head_offset(self):
        """任意数量丢弃后：n_blocks_of 一致、无悬挂块、seq_len 正确。"""
        mgr = BlockManager(n_blocks=16, block_size=8, n_seqs=1, max_blocks_per_seq=8)
        mgr.append_tokens(0, 40)  # ceil((0+40)/8) = 5 块
        assert mgr.n_blocks_of(0) == 5
        assert mgr.drop_tokens(0, 5) == 5  # 非整块：head_offset=5，仍 5 块
        assert mgr.head_offset[0] == 5 and mgr.seq_len(0) == 35
        assert mgr.n_blocks_of(0) == 5
        assert mgr.drop_tokens(0, 3) == 3  # head_offset=8 -> 恰好回收 1 块
        assert mgr.head_offset[0] == 0 and mgr.seq_len(0) == 32
        assert mgr.n_blocks_of(0) == 4
        assert mgr.num_free_blocks() == 16 - 1 - 4
        assert mgr.drop_tokens(0, 100) == 32  # 超过则全丢
        assert mgr.seq_len(0) == 0 and mgr.n_blocks_of(0) == 0

    def test_refcount_shared_blocks(self):
        """共享块：一条序列释放不会误回收另一条仍在用的块。"""
        mgr = BlockManager(n_blocks=8, block_size=4, n_seqs=2, max_blocks_per_seq=4)
        mgr.append_tokens(0, 8)
        bid = int(mgr.block_tables[0, 0])
        mgr.ref_cnt[bid] = 2  # 模拟第二条序列 touch 了同一块
        mgr.free_seq(0)
        assert mgr.ref_cnt[bid] == 1
        assert bid not in list(mgr._free)
        mgr.release_block(bid)
        assert mgr.ref_cnt[bid] == 0 and bid in list(mgr._free)

    def test_double_free_detected(self):
        mgr = BlockManager(n_blocks=8, block_size=4, n_seqs=1, max_blocks_per_seq=4)
        mgr.append_tokens(0, 4)
        bid = int(mgr.block_tables[0, 0])
        mgr.free_seq(0)
        with pytest.raises(RuntimeError):
            mgr.release_block(bid)


# --------------------------------------------------------------------------- #
# PagedKVCache                                                                  #
# --------------------------------------------------------------------------- #
class TestPagedKVCache:
    def _mk(self, n_blocks=8, block_size=4, n_layer=2, n_kv=2, dh=4, max_blk=8):
        mgr = BlockManager(
            n_blocks=n_blocks, block_size=block_size, n_seqs=1, max_blocks_per_seq=max_blk
        )
        return PagedKVCache(mgr, n_layer, n_kv, dh), mgr

    def _write_all_layers(self, cache, k, v):
        """按三段式把同一批 K/V 写入全部层（返回 gather 用的层数）。"""
        T = k.shape[0]
        assert cache.begin_append(0, T)
        for layer in range(cache.n_layer):
            cache.write(0, layer, k, v)
        cache.end_append(0)

    def test_append_gather_roundtrip(self):
        """跨块写入后 gather 必须逐元素还原（含滑动窗口偏移）。"""
        cache, mgr = self._mk()
        rng = np.random.default_rng(1)
        k = rng.standard_normal((10, 2, 4)).astype(np.float32)
        v = rng.standard_normal((10, 2, 4)).astype(np.float32)
        self._write_all_layers(cache, k, v)
        gk, gv = cache.gather(0, 0)
        assert np.array_equal(gk, k) and np.array_equal(gv, v)

    def test_append_across_many_blocks(self):
        """T 跨 8 个块 + 每层独立写入。"""
        cache, mgr = self._mk(n_blocks=16, block_size=4)
        rng = np.random.default_rng(2)
        T = 30
        k = rng.standard_normal((T, 2, 4)).astype(np.float32)
        v = rng.standard_normal((T, 2, 4)).astype(np.float32)
        self._write_all_layers(cache, k, v)
        for layer in range(2):
            gk, gv = cache.gather(0, layer)
            assert np.array_equal(gk, k)
        assert mgr.n_blocks_of(0) == 8  # ceil(30/4)

    def test_three_phase_protocol(self):
        """begin/write/end 三段式：位置与便利封装一致，且校验批次大小。"""
        cache, mgr = self._mk(block_size=4)
        rng = np.random.default_rng(5)
        T = 10
        k = rng.standard_normal((T, 2, 4)).astype(np.float32)
        v = rng.standard_normal((T, 2, 4)).astype(np.float32)
        assert cache.begin_append(0, T) is True
        for layer in range(2):
            cache.write(0, layer, k, v)
        with pytest.raises(RuntimeError):
            cache.write(0, 0, k[:5], v[:5])  # 批次大小不一致
        cache.end_append(0)
        gk, _ = cache.gather(0, 0)
        assert np.array_equal(gk, k)
        with pytest.raises(RuntimeError):
            cache.write(0, 0, k, v)  # 无活动批次

    def test_append_insufficient_space(self):
        cache, mgr = self._mk(n_blocks=4, block_size=4)
        k = np.zeros((4, 2, 4), dtype=np.float32)
        assert cache.append(0, 0, k, k) is True  # 1 块（池有 3 块可用）
        assert cache.append(0, 0, k, k) is True
        assert cache.append(0, 0, k, k) is True  # 3 块用尽
        assert cache.append(0, 0, k, k) is False  # 池空
        assert mgr.seq_len(0) == 12

    def test_gather_after_drop(self):
        """滑动窗口丢弃后 gather 只返回保留的 token。"""
        cache, _ = self._mk(block_size=4)
        rng = np.random.default_rng(3)
        T = 20
        k = rng.standard_normal((T, 2, 4)).astype(np.float32)
        v = rng.standard_normal((T, 2, 4)).astype(np.float32)
        assert cache.append(0, 0, k, v) is True
        cache.drop_front(0, 6)  # 经缓存层：同步 mgr.seq_len 与 layer_len，head_offset=2，seq_len=14
        gk, _ = cache.gather(0, 0)
        assert gk.shape[0] == 14
        assert np.array_equal(gk, k[6:])

    def test_layout_shape(self):
        cache, _ = self._mk(n_blocks=5, n_layer=3, n_kv=2, dh=6)
        assert cache.kv.shape == (5, 3, 2, 4, 2, 6)
        assert cache.nbytes() == 5 * 3 * 2 * 4 * 2 * 6 * 4


# --------------------------------------------------------------------------- #
# 注意力对齐                                                                    #
# --------------------------------------------------------------------------- #
class TestAttentionAlignment:
    def _mk(self, n_blocks=32, block_size=8, n_layer=2, n_kv=2, dh=8):
        mgr = BlockManager(n_blocks=n_blocks, block_size=block_size, n_seqs=1, max_blocks_per_seq=8)
        return PagedKVCache(mgr, n_layer, n_kv, dh), mgr

    @pytest.mark.parametrize(
        "T,S,q_start", [(1, 1, 0), (1, 8, 7), (4, 4, 0), (8, 20, 12), (3, 17, 5)]
    )
    def test_paged_matches_dense(self, T, S, q_start):
        """不变量 ⑨：分页 vs dense < 1e-5（且与朴素逐元素实现一致）。"""
        rng = np.random.default_rng(100 + T * 7 + S)
        H, Dh, n_kv = 4, 8, 2
        cache, mgr = self._mk(n_kv=n_kv)
        n_rep = H // n_kv
        q = rng.standard_normal((1, H, T, Dh)).astype(np.float32)
        # GQA 构造：dense 的 H 个头由 n_kv 个 KV 头 repeat 而来（与引擎一致）
        k_base = rng.standard_normal((1, n_kv, S, Dh)).astype(np.float32)
        v_base = rng.standard_normal((1, n_kv, S, Dh)).astype(np.float32)
        k_all = np.repeat(k_base, n_rep, axis=1)  # [1, H, S, Dh]
        v_all = np.repeat(v_base, n_rep, axis=1)
        # cache 只存 n_kv 份
        k_kv = k_base[0].transpose(1, 0, 2)  # [S, n_kv, Dh]
        v_kv = v_base[0].transpose(1, 0, 2)
        self._write_kv(cache, k_kv, v_kv)
        got = paged_gqa_attention(q, cache, 0, 0, n_rep, q_start)
        ref = gqa_attention(q, k_all, v_all, q_start)
        naive = naive_attention(q, k_all, v_all, q_start)
        assert np.max(np.abs(got - ref)) < 1e-5
        assert np.max(np.abs(ref - naive)) < 1e-5

    @staticmethod
    def _write_kv(cache, k_kv, v_kv):
        assert cache.begin_append(0, k_kv.shape[0])
        for layer in range(cache.n_layer):
            cache.write(0, layer, k_kv, v_kv)
        cache.end_append(0)

    def test_paged_after_drop_matches_dense(self):
        """滑动窗口后分页结果 == 对保留子集的 dense 注意力。"""
        rng = np.random.default_rng(7)
        H, Dh, n_kv = 4, 8, 2
        cache, _ = self._mk(n_kv=n_kv)
        n_rep = H // n_kv
        S = 24
        k_base = rng.standard_normal((1, n_kv, S, Dh)).astype(np.float32)
        v_base = rng.standard_normal((1, n_kv, S, Dh)).astype(np.float32)
        k_all = np.repeat(k_base, n_rep, axis=1)
        v_all = np.repeat(v_base, n_rep, axis=1)
        self._write_kv(cache, k_base[0].transpose(1, 0, 2), v_base[0].transpose(1, 0, 2))
        cache.drop_front(0, 10)  # 丢最旧 10 个，保留 [10, 24)；同步 layer_len 与 mgr.seq_len
        q = rng.standard_normal((1, H, 2, Dh)).astype(np.float32)
        # drop 之后 query 的位置要换成**保留坐标系**（首个保留 token 下标为 0）
        got = paged_gqa_attention(q, cache, 0, 0, n_rep, q_start=0)
        ref = gqa_attention(q, k_all[:, :, 10:], v_all[:, :, 10:], q_start=0)
        assert np.max(np.abs(got - ref)) < 1e-5

    def test_decode_single_token(self):
        """T=1 decode：q_start=0 时只能看到第 0 个 key（因果掩码）。"""
        rng = np.random.default_rng(8)
        H, Dh = 2, 8
        cache, mgr = self._mk(n_kv=H)
        n_rep = 1
        S = 6
        k_all = rng.standard_normal((1, H, S, Dh)).astype(np.float32)
        v_all = rng.standard_normal((1, H, S, Dh)).astype(np.float32)
        cache.append(0, 0, k_all[0].transpose(1, 0, 2), v_all[0].transpose(1, 0, 2))
        q = rng.standard_normal((1, H, 1, Dh)).astype(np.float32)
        # 正常 decode：全部可见
        got = paged_gqa_attention(q, cache, 0, 0, n_rep, q_start=S - 1)
        ref = gqa_attention(q, k_all, v_all, q_start=S - 1)
        assert np.max(np.abs(got - ref)) < 1e-5
        # 因果：q_start=0 时只看到 key 0 → softmax 单一可见键权重恒为 1
        got0 = paged_gqa_attention(q, cache, 0, 0, n_rep, q_start=0)
        assert np.max(np.abs(got0[0, :, 0] - v_all[0, :, 0])) < 1e-5
        # 半可见：q_start=2 时看到 key 0..2，可手工复算
        got2 = paged_gqa_attention(q, cache, 0, 0, n_rep, q_start=2)
        s = np.einsum("hd,hsd->hs", q[0, :, 0], k_all[0, :, :3]) / math.sqrt(Dh)
        s = s - s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w = w / w.sum(axis=1, keepdims=True)
        expect2 = np.einsum("hs,hsd->hd", w, v_all[0, :, :3])
        assert np.max(np.abs(got2[0, :, 0] - expect2)) < 1e-5
