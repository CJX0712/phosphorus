"""T4-C KV 驱逐策略测试（不变量 ⑦）。

覆盖：
* 预算守恒：逐层预算之和 == 总预算 ± 1（C5 验收）。
* 强制保留：sink + 最近窗口永不被驱逐；预算被精确尊重。
* compact 物理压缩：保留 K/V 逐位等价、layer_len 正确、物理块归还。
* 下游保真度：50% 预算下 final-logits 余弦相似度 ≥ 0.99（SnapKV/H2O/PyramidKV
  论文均以此类下游指标衡量压缩质量；原始注意力输出 L2 在 50% 下即便 oracle 也
  ~0.5，属指标固有性质，故不作为验收阈值）。
* `none` 对照组：完全不驱逐时缓存保持全量。
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphorus.block_manager import BlockManager
from phosphorus.engine import PhosModel
from phosphorus.eviction import (
    BudgetAllocator,
    EvictionConfig,
    PyramidKVPolicy,
    make_policy,
)
from phosphorus.kv_cache import PagedKVCache

CKPT = "ckpt/tiny.phos"


@pytest.fixture(scope="module")
def ckpt_path() -> str:
    import os

    if not os.path.exists(CKPT):
        pytest.skip(f"{CKPT} 不存在（请先跑 make train-fast 或 scripts/setup）")
    return CKPT


# --------------------------------------------------------------------------- #
# 1. 预算守恒（Σ == L·C ± 1）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_layer", [1, 3, 8, 32])
@pytest.mark.parametrize("seq_len", [100, 256, 1000])
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_budget_conservation_pyramid(n_layer, seq_len, ratio):
    cfg = EvictionConfig(budget_ratio=ratio)
    pol = PyramidKVPolicy(cfg)
    budgets = pol.layer_budgets(n_layer, seq_len)
    assert len(budgets) == n_layer
    C = int(round(seq_len * ratio))
    # 总和与 L·C 差 ≤ 1（最大余数法修正的验收边界）
    assert abs(sum(budgets) - n_layer * C) <= 1
    # 每层保留 ≥ 1
    assert all(b >= 1 for b in budgets)
    # 没有超出序列长度
    assert all(b <= seq_len for b in budgets)


@pytest.mark.parametrize("n_layer", [3, 8, 32])
@pytest.mark.parametrize("seq_len", [256, 1000])
def test_budget_conservation_uniform(n_layer, seq_len):
    cfg = EvictionConfig(budget_ratio=0.5)
    alloc = BudgetAllocator(cfg)
    budgets = alloc.layer_budgets(n_layer, seq_len, mode="uniform")
    C = int(round(seq_len * 0.5))
    assert sum(budgets) == n_layer * C
    assert budgets == [C] * n_layer


# --------------------------------------------------------------------------- #
# 2. 强制保留 + 预算精确
# --------------------------------------------------------------------------- #
def _make_scores(H, n_past, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((H, n_past))


@pytest.mark.parametrize("name", ["snapkv", "pyramidkv", "h2o", "streamingllm"])
def test_forced_keep_and_budget(name):
    cfg = EvictionConfig(budget_ratio=0.5, window_size=32, sink_size=4)
    pol = make_policy(name, cfg)
    H = 3
    n_total = 200
    sink = 4
    window = 32
    n_past = n_total - sink - window
    scores = _make_scores(H, n_past)
    # 预算低于 n_total：应精确保留 budget 个
    budget = 100
    mask = pol.select(scores, budget, n_total)
    assert mask.sum() == budget
    # 强制保留：前 sink 必定为 True
    assert mask[:sink].all()
    # 最近窗口（长度 = min(window, budget - sink)）必定为 True
    eff_window = min(window, budget - sink)
    assert mask[n_total - eff_window :].all()


@pytest.mark.parametrize("name", ["snapkv", "pyramidkv", "h2o", "streamingllm"])
def test_forced_keep_below_floor(name):
    cfg = EvictionConfig(budget_ratio=0.01, window_size=32, sink_size=4)
    pol = make_policy(name, cfg)
    H = 3
    n_total = 200
    sink = 4
    window = 32
    n_past = n_total - sink - window
    scores = _make_scores(H, n_past)
    # 预算极小：内容策略受 sink+window 下限约束，StreamingLLM 窗口随预算收窄
    budget = 10  # < sink(4)+window(32) = 36
    mask = pol.select(scores, budget, n_total)
    assert mask[:sink].all()  # sink 永留
    if name == "streamingllm":
        # 窗口随预算收窄：总保留 = budget（含 sink）
        assert mask.sum() == budget
        eff_window = budget - sink
    else:
        # 内容策略有 sink+window 下限：总保留 = sink + window
        assert mask.sum() == sink + window
        eff_window = window
    assert mask[n_total - eff_window :].all()


# --------------------------------------------------------------------------- #
# 3. compact 物理压缩：逐位等价 + layer_len + 空间归还
# --------------------------------------------------------------------------- #
def _make_cache(n_blocks=32, block_size=8, n_layer=3, n_kv=2, d_head=16, n_seqs=1):
    mgr = BlockManager(
        n_blocks=n_blocks, block_size=block_size, n_seqs=n_seqs, max_blocks_per_seq=n_blocks - 1
    )
    return PagedKVCache(mgr, n_layer, n_kv, d_head)


def _fill(cache, seq_id, src):
    """用三段式协议把 T 个 token 写到全部层（seq_len 只记一次，避免逐层 append 重复计数）。"""
    T = int(src[0].shape[0])
    cache.begin_append(seq_id, T)
    for li in range(len(src)):
        cache.write(seq_id, li, src[li], src[li])
    cache.end_append(seq_id)


def test_compact_roundtrip_uniform():
    L, n_kv, d = 3, 2, 16
    cache = _make_cache(n_layer=L, n_kv=n_kv, d_head=d)
    T = 40
    rng = np.random.default_rng(0)
    src = [rng.standard_normal((T, n_kv, d)).astype(np.float32) for _ in range(L)]
    _fill(cache, 0, src)
    # 保留前 24 个（全部层相同）
    keep = np.zeros(T, dtype=bool)
    keep[:24] = True
    dropped = cache.compact(0, keep)
    assert dropped == 16
    assert int(cache.layer_len[0, 0]) == 24
    assert cache.mgr.seq_len(0) == 24
    # 逐层 K/V 逐位等价
    for li in range(L):
        k, v = cache.gather(0, li)
        assert k.shape[0] == 24
        np.testing.assert_array_equal(k, src[li][:24])
        np.testing.assert_array_equal(v, src[li][:24])
    # 释放后空间应回到满（utilization 归零，layer_len 由 reset/compact 负责清零）
    cache.mgr.free_seq(0)
    assert cache.mgr.utilization() == 0.0


def test_compact_per_layer_different():
    L, n_kv, d = 3, 2, 16
    cache = _make_cache(n_layer=L, n_kv=n_kv, d_head=d)
    T = 40
    rng = np.random.default_rng(1)
    src = [rng.standard_normal((T, n_kv, d)).astype(np.float32) for _ in range(L)]
    _fill(cache, 0, src)
    # 每层保留不同数量：层 0 保留 30，层 1 保留 20，层 2 保留 10
    keep_counts = [30, 20, 10]
    masks = np.zeros((L, T), dtype=bool)
    for li in range(L):
        masks[li, : keep_counts[li]] = True
    cache.compact(0, masks)
    # 块表按 max(keep) 分配 -> 共享 seq_len = 30
    assert cache.mgr.seq_len(0) == 30
    for li in range(L):
        assert int(cache.layer_len[0, li]) == keep_counts[li]
        k, _ = cache.gather(0, li)
        assert k.shape[0] == keep_counts[li]
        np.testing.assert_array_equal(k, src[li][: keep_counts[li]])


def test_compact_noop():
    L, n_kv, d = 2, 1, 8
    cache = _make_cache(n_layer=L, n_kv=n_kv, d_head=d)
    T = 16
    rng = np.random.default_rng(2)
    src = [rng.standard_normal((T, n_kv, d)).astype(np.float32) for _ in range(L)]
    _fill(cache, 0, src)
    keep = np.ones(T, dtype=bool)
    dropped = cache.compact(0, keep)
    assert dropped == 0
    assert int(cache.layer_len[0, 0]) == T


# --------------------------------------------------------------------------- #
# 4. 下游保真度（C5 验收）：50% 预算下 final-logits 余弦 ≥ 0.99
# --------------------------------------------------------------------------- #
def test_downstream_fidelity_at_50pct(ckpt_path):
    pref = list(range(10, 210))  # 200 token 预填充（触发压缩）
    dec = list(range(210, 230))  # 20 个 decode
    # 无驱逐参考（dense 全量，作为 ground truth）
    mf = PhosModel.from_file(ckpt_path, eviction="none")
    mf.forward(np.array(pref, dtype=np.int64))
    ref_logits = []
    for t in dec:
        ref_logits.append(mf.forward(np.array([t], dtype=np.int64))[-1])

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for pol in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
        m = PhosModel.from_file(ckpt_path, eviction=pol, kv_budget=0.5)
        m.forward(np.array(pref, dtype=np.int64))
        coss = []
        for j, t in enumerate(dec):
            lg = m.forward(np.array([t], dtype=np.int64))[-1]
            coss.append(cos(lg, ref_logits[j]))
        mean_cos = float(np.mean(coss))
        # 50% 预算下各策略都应保持高保真（真实文献以 PPL/logits 衡量，非原始注意力 L2）
        assert mean_cos >= 0.99, f"{pol} 50% 预算下游余弦={mean_cos:.4f} < 0.99"


def test_none_control_no_eviction(ckpt_path):
    pref = list(range(10, 210))
    m = PhosModel.from_file(ckpt_path, eviction="none")
    m.forward(np.array(pref, dtype=np.int64))
    # none 不驱逐：缓存保持全量（无任何压缩）
    assert m.cache_len() == len(pref)


def test_eviction_reduces_cache(ckpt_path):
    pref = list(range(10, 210))
    for pol in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
        m = PhosModel.from_file(ckpt_path, eviction=pol, kv_budget=0.5)
        m.forward(np.array(pref, dtype=np.int64))
        # 预填充 200 触发压缩：缓存应明显小于全量 200
        assert m.cache_len() < 200
