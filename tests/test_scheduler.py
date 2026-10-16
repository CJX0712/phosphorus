"""T4-D 连续批处理调度器测试（C6 等价 + C7 采样）。

覆盖：
* 融合前向数值正确性：forward_step 单序列 == forward；多序列各序列 == 各自单序列。
* C6（等价）：Scheduler 跑 8 条序列，与逐条串行（KV 独立）的 token id 完全一致
  （greedy, temperature=0）。
* C6（吞吐）：调度器 8 条融合前向的 tokens/s ≥ 2× 单条串行（同 token 总量）。
* C7（采样）：temperature=0 → greedy；同 seed 可复现；随机采样与贪心结果不同；
  top_k / top_p 行为正确（由 sampling.py 自身单测兜底，这里做端到端一致性）。
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from phosphorus.engine import PhosModel
from phosphorus.sampling import sample as _sample
from phosphorus.scheduler import Scheduler

CKPT = "ckpt/tiny.phos"

PROMPTS = [
    "To be, or not to be",
    "Once upon a time",
    "The king said",
    "In the beginning",
    "It was the best",
    "She looked at",
    "All the world",
    "When the sun",
]


@pytest.fixture(scope="module")
def ckpt_path() -> str:
    if not os.path.exists(CKPT):
        pytest.skip(f"{CKPT} 不存在（请先跑 make train-fast 或 scripts/setup）")
    return CKPT


def serial_generate_ids(model, prompt_ids, max_new_tokens, temperature, top_k, top_p, seed):
    """逐条串行参考实现（每条 prompt KV 独立：reset 后单独 forward）。

    与 Scheduler 的语义对齐：每条 prompt 用独立 rng（seed + 全局下标），
    保证「融合批处理」与「逐条串行」在随机采样下也逐位一致。
    """
    model.reset()
    rng = np.random.default_rng(seed)
    logits = model.forward(np.array(prompt_ids, dtype=np.int64))
    nid = _sample(logits[-1], temperature, top_k, top_p, rng)
    gen = list(prompt_ids) + [nid]
    for _ in range(max_new_tokens - 1):
        lg = model.forward(np.array([nid], dtype=np.int64))
        nid = _sample(lg[-1], temperature, top_k, top_p, rng)
        gen.append(nid)
        if model.eos_id is not None and nid == model.eos_id:
            break
    return gen[len(prompt_ids) :]


# --------------------------------------------------------------------------- #
# 1. 融合前向数值正确性
# --------------------------------------------------------------------------- #
def test_forward_step_single_matches_forward(ckpt_path):
    """单序列 forward_step({0: ids}) 的末位 logits == forward(ids) 末位。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=1)
    rng = np.random.default_rng(7)
    ids = rng.integers(0, 100, size=9).astype(np.int64)
    model.reset()
    ref = model.forward(ids)[-1]
    model.reset()
    got = model.forward_step({0: ids})[0]
    assert np.max(np.abs(ref - got)) < 1e-5


def test_forward_step_multiseq_matches_individual(ckpt_path):
    """多序列融合：每条序列的末位 logits == 它单独 forward 的结果（KV 隔离）。"""
    seqs = {
        0: np.array([1, 2, 3], dtype=np.int64),
        1: np.array([4, 5, 6, 7], dtype=np.int64),
        2: np.array([8, 9], dtype=np.int64),
    }
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=3)
    model.reset()
    out = model.forward_step(seqs)
    assert set(out.keys()) == set(seqs.keys())
    for s, ids in seqs.items():
        m = PhosModel.from_file(ckpt_path, kv_mode="paged")
        m.reset()
        ref = m.forward(ids)[-1]
        assert np.max(np.abs(out[s] - ref)) < 1e-5


# --------------------------------------------------------------------------- #
# 2. C6 等价：8 条序列 token id 与串行完全一致
# --------------------------------------------------------------------------- #
def test_scheduler_greedy_matches_serial(ckpt_path):
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(model, max_seqs=8)
    pids = [model.tok.encode(p) for p in PROMPTS]
    got = sched.generate_ids(pids, max_new_tokens=24, temperature=0.0, top_k=1, seed=0)

    ref = []
    for j, p in enumerate(PROMPTS):
        m = PhosModel.from_file(ckpt_path, kv_mode="paged")
        ref.append(serial_generate_ids(m, m.tok.encode(p), 24, 0.0, 1, 0.95, seed=j))
    assert got == ref
    # 每条都确实生成了 token
    assert all(len(g) > 0 for g in got)


def test_scheduler_greedy_text_equivalence(ckpt_path):
    """端到端文本层面：Scheduler 生成的续写文本 == 逐条串行生成的续写文本。

    注意 Scheduler.generate 返回的是「续写部分」（不含 prompt），与 model.generate
    的「prompt + 续写」不同；故参考侧用 serial_generate_ids 取续写 id 再 decode。
    """
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(model, max_seqs=8)
    texts = sched.generate(PROMPTS, max_new_tokens=24, temperature=0.0, top_k=1, seed=0)

    ref = []
    for j, p in enumerate(PROMPTS):
        m = PhosModel.from_file(ckpt_path, kv_mode="paged")
        ids = serial_generate_ids(m, m.tok.encode(p), 24, 0.0, 1, 0.95, seed=j)
        ref.append(model.tok.decode(ids))
    assert texts == ref


# --------------------------------------------------------------------------- #
# 3. C6 吞吐：融合前向 ≥ 2× 单条串行
# --------------------------------------------------------------------------- #
def test_scheduler_throughput_speedup(ckpt_path):
    prompts = PROMPTS
    max_new = 40

    # 调度器（8 条融合）
    m_sched = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(m_sched, max_seqs=8)
    pids = [m_sched.tok.encode(p) for p in prompts]
    t0 = time.time()
    sched.generate_ids(pids, max_new_tokens=max_new, temperature=0.0, top_k=1, seed=0)
    dt_sched = time.time() - t0

    # 串行（单模型，逐条 reset，公平对比——不含模型加载开销）
    m_serial = PhosModel.from_file(ckpt_path, kv_mode="paged")
    t1 = time.time()
    for j, p in enumerate(prompts):
        serial_generate_ids(m_serial, m_serial.tok.encode(p), max_new, 0.0, 1, 0.95, seed=j)
    dt_serial = time.time() - t1

    total_tok = len(prompts) * max_new
    tput_sched = total_tok / max(dt_sched, 1e-9)
    tput_serial = total_tok / max(dt_serial, 1e-9)
    # 防御：两条都别是 0
    assert dt_sched > 0 and dt_serial > 0
    assert tput_sched >= 2.0 * tput_serial, (
        f"连续批处理吞吐未达 2×：sched={tput_sched:.1f} tok/s, "
        f"serial={tput_serial:.1f} tok/s (dt_sched={dt_sched:.3f}, dt_serial={dt_serial:.3f})"
    )


# --------------------------------------------------------------------------- #
# 4. C7 采样：greedy / 可复现 / 随机性
# --------------------------------------------------------------------------- #
def test_scheduler_reproducible_same_seed(ckpt_path):
    """同 seed 两次运行结果完全一致（C7 可复现）。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(model, max_seqs=8)
    pids = [model.tok.encode(p) for p in PROMPTS[:2]]
    a = sched.generate_ids(pids, max_new_tokens=20, temperature=0.9, top_k=10, top_p=0.95, seed=42)
    b = sched.generate_ids(pids, max_new_tokens=20, temperature=0.9, top_k=10, top_p=0.95, seed=42)
    assert a == b


def test_scheduler_stochastic_differs_from_greedy(ckpt_path):
    """随机采样（temperature>0）与贪心（temperature=0）结果不同（C7 确有随机性）。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(model, max_seqs=8)
    pids = [model.tok.encode(p) for p in PROMPTS[:2]]
    greedy = sched.generate_ids(pids, max_new_tokens=20, temperature=0.0, top_k=1, seed=0)
    stoch = sched.generate_ids(
        pids, max_new_tokens=20, temperature=0.9, top_k=10, top_p=0.95, seed=42
    )
    assert stoch != greedy


def test_scheduler_wave_splits_large_batch(ckpt_path):
    """prompt 数 > max_seqs 时按波次切分，结果与串行仍一致。"""
    many = PROMPTS * 2  # 16 条，超过 max_seqs=8
    model = PhosModel.from_file(ckpt_path, kv_mode="paged", max_seqs=8)
    sched = Scheduler(model, max_seqs=8)
    got = sched.generate_ids(
        [model.tok.encode(p) for p in many], max_new_tokens=16, temperature=0.0, top_k=1, seed=0
    )
    ref = []
    for j, p in enumerate(many):
        m = PhosModel.from_file(ckpt_path, kv_mode="paged")
        ref.append(serial_generate_ids(m, m.tok.encode(p), 16, 0.0, 1, 0.95, seed=j))
    assert got == ref
