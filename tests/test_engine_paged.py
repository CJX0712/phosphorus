"""T4-B 引擎级：分页 KV（paged）与 dense 参考的数值等价（不变量 ⑨）。

这些测试**显式加载真实 `.phos` 检查点**（`ckpt/tiny.phos`，由 Makefile 生成），
因此同时验证：
* 两种 kv_mode 在 fp32 / tl / i2s 三种数学后端下 logits 一致；
* 分块 prefill（chunked）与整段 prefill 一致（KV cache 增量语义）；
* 超过 max_seq_len 的滑动窗口不崩溃且与 dense 一致；
* reset() 后块管理器利用率归零。
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphorus.engine import PhosModel  # noqa: E402

CKPT = "ckpt/tiny.phos"


@pytest.fixture(scope="module")
def ckpt_path() -> str:
    import os

    if not os.path.exists(CKPT):
        pytest.skip(f"{CKPT} 不存在（请先跑 make train-fast 或 scripts/setup）")
    return CKPT


@pytest.mark.parametrize("backend", ["fp32", "tl", "i2s"])
def test_paged_equals_dense_logits(ckpt_path, backend):
    """不变量 ⑨（引擎级）：同一 checkpoint，两种 kv_mode logits 一致。"""
    dense = PhosModel.from_file(ckpt_path, backend=backend, kv_mode="dense")
    paged = PhosModel.from_file(ckpt_path, backend=backend, kv_mode="paged")
    rng = np.random.default_rng(11)
    ids = rng.integers(0, 100, size=24).astype(np.int64)
    ld = dense.forward(ids)
    lp = paged.forward(ids)
    assert np.max(np.abs(ld - lp)) < 1e-4
    # argmax 完全一致（生成行为不变）
    assert np.array_equal(np.argmax(ld, -1), np.argmax(lp, -1))


def test_chunked_prefill_equals_full(ckpt_path):
    """分块 prefill：4+4+16 与一次 24 token 的 logits 一致。"""
    paged = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged")
    rng = np.random.default_rng(12)
    ids = rng.integers(0, 100, size=24).astype(np.int64)
    full = paged.forward(ids)
    paged.reset()
    outs = [paged.forward(ids[:4]), paged.forward(ids[4:8]), paged.forward(ids[8:])]
    chunked = np.concatenate(outs, axis=0)
    assert np.max(np.abs(full - chunked)) < 1e-4


def test_decode_loop_matches_dense(ckpt_path):
    """逐 token decode 12 步：paged 与 dense 的 logits 一致。"""
    dense = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="dense")
    paged = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged")
    ids = np.arange(8, dtype=np.int64)
    ld = dense.forward(ids)[-1]
    lp = paged.forward(ids)[-1]
    assert np.max(np.abs(ld - lp)) < 1e-4
    for step in range(12):
        tok = np.array([int(np.argmax(lp))], dtype=np.int64)
        ld = dense.forward(tok)[-1]
        lp = paged.forward(tok)[-1]
        assert np.max(np.abs(ld - lp)) < 1e-4, f"decode step {step} 不一致"


def test_sliding_window_matches_dense(ckpt_path):
    """超过 max_seq_len：滑动窗口路径 paged 与 dense 行为一致。"""
    dense = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="dense")
    paged = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged")
    over = paged.max_seq_len + 20
    rng = np.random.default_rng(13)
    ids = rng.integers(0, 100, size=over).astype(np.int64)
    ld = dense.forward(ids)
    lp = paged.forward(ids)
    assert np.max(np.abs(ld - lp)) < 1e-4
    assert paged.cache_len() <= paged.max_seq_len
    # 丢弃后池利用率不超限
    assert paged.cache.mgr.utilization() <= 1.0


def test_reset_returns_pool(ckpt_path):
    """reset() 后块管理器利用率精确归零（无泄漏）。"""
    paged = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged")
    paged.forward(np.arange(200, dtype=np.int64) % 100)
    assert paged.cache.mgr.utilization() > 0.5
    paged.reset()
    assert paged.cache.mgr.utilization() == 0.0
    assert paged.cache_len() == 0


def test_generate_identical_text(ckpt_path):
    """端到端 greedy 生成：两种 kv_mode 产出同一段文本。"""
    prompt = "To be, or not to be"
    a = PhosModel.from_file(ckpt_path, backend="tl", kv_mode="dense")
    b = PhosModel.from_file(ckpt_path, backend="tl", kv_mode="paged")
    ta = a.generate(prompt, max_new_tokens=40, temperature=0.0, top_k=1, seed=0)
    tb = b.generate(prompt, max_new_tokens=40, temperature=0.0, top_k=1, seed=0)
    assert ta == tb
    assert len(ta) > 0


def test_block_size_variants(ckpt_path):
    """不同块大小（4 / 16 / 64）paged 结果互相一致。"""
    ref = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged", block_size=16)
    rng = np.random.default_rng(14)
    ids = rng.integers(0, 100, size=37).astype(np.int64)
    expect = ref.forward(ids)
    for bs in (4, 32, 64):
        m = PhosModel.from_file(ckpt_path, backend="fp32", kv_mode="paged", block_size=bs)
        got = m.forward(ids)
        assert np.max(np.abs(got - expect)) < 1e-4, f"block_size={bs} 不一致"
