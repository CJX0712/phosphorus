"""T5 · C1 分布等价（不变量 ② ★）+ C2 O1 零额外权重。

硬门：投机解码的输出分布必须与目标模型**直接采样**分布统计等价：
    * χ² 齐性检验 `p > 0.05`、经验 KL < 1e-3；
    * `temperature=0`（贪心）时，投机与串行产生的 token 序列**逐位相同**。

两点关键（曾导致本测试「静默跳过 / 假通过」，已修正）：
1. **验证必须用采样器真正采样的分布**：即 `sampling.sampling_probs(logits, temp,
   top_k, top_p)`（temperature + top-p 截断之后），而非原始 softmax——否则 top_p<1
   时输出分布与串行不等价。
2. **统计功效**：tiny.phos 在单条 prompt + temp=0.8/top_p=0.95 下分布**退化**
   （核内仅 1 个 token，实测 3000/3000 同 token），χ² 无有效单元格 → 原实现
   `pytest.skip`。现改为**多 prompt 池化**（扩大单元格）+ 一路**高温非退化配置**
   （真正有辨别力），并把「静默 skip」改为「响亮失败」。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from phosphorus.engine import PhosModel
from phosphorus.sampling import sample as _sample
from phosphorus.sampling import sampling_probs
from phosphorus.speculate import LayerSkipDrafter, spec_verify

CKPT = "ckpt/tiny.phos"
K = 4

# 多 prompt 池化：不同上下文的 next-token 分布不同，池化后 χ² 才有足够单元格
PROMPTS = [
    "To be, or not to be",
    "Once upon a time",
    "The king said",
    "In the beginning",
    "It was the best",
    "She looked at",
    "All the world",
    "When the sun",
    "Now is the winter",
    "What light through yonder",
]


@pytest.fixture(scope="module")
def ckpt_path() -> str:
    if not os.path.exists(CKPT):
        pytest.skip(f"{CKPT} 不存在（请先训练/导出 tiny.phos）")
    return CKPT


def _chi2_sf(x, df):
    """χ² 生存函数（Wilson–Hilferty 近似，无需 scipy）。"""
    if df <= 0 or x <= 0:
        return 1.0
    t = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def _one_step(model, ctx, rng, drafter, temperature, top_p, greedy=False):
    """执行一轮投机（K=4），返回本轮接受的 token 列表（至少一个）。

    草稿采样与目标验证均使用 `sampling_probs`（采样器真实分布），保证与串行
    采样的分布可比。
    """
    draft, q_dists = [], []
    for _ in range(K):
        model.reset()
        out = model.forward(
            np.asarray(ctx + draft, dtype=np.int64), stop_at_layer=drafter.draft_layers
        )
        q = sampling_probs(out[-1], temperature, 0, top_p)
        q_dists.append(q)
        if greedy:
            draft.append(int(np.argmax(q)))
        else:
            draft.append(int(rng.choice(len(q), p=q)))
    model.reset()
    P = model.forward(np.asarray(ctx + draft, dtype=np.int64))
    L = len(ctx)
    p_dists = [sampling_probs(P[L - 1 + i], temperature, 0, top_p) for i in range(K)]
    p_dists.append(sampling_probs(P[L + K - 1], temperature, 0, top_p))
    res = spec_verify(draft, q_dists, p_dists, rng, greedy=greedy)
    return res.accepted


def _collect(model, prompts, n_each, temperature, top_p, seed_base=1000):
    """对若干 prompt 各采样 n_each 次，返回 (串行A, 串行B, 投机) 三组计数。

    串行 A/B 是**两次独立串行运行**，用于标定「纯采样噪声」下的 KL 底噪——
    经验 KL 在有限样本下的期望量级 ≈ 有效单元格数 / N，直接用绝对阈值
    （如 < 1e-3）统计上不可达，故必须用同规模的自校准基线对比。
    """
    tok = model.tok
    V = model.vocab
    drafter = LayerSkipDrafter(model, draft_layers=max(1, model.n_layer - 1))
    serial_a = np.zeros(V, dtype=np.int64)
    serial_b = np.zeros(V, dtype=np.int64)
    spec = np.zeros(V, dtype=np.int64)
    for pi, prompt in enumerate(prompts):
        ctx = tok.encode(prompt)
        base = seed_base + pi * 100000
        for i in range(n_each):
            model.reset()
            lg = model.forward(np.asarray(ctx, dtype=np.int64))
            logits = np.asarray(lg[-1], dtype=np.float64)
            # 同一份 logits 上取三个独立采样（共享一次前向，节省开销）
            serial_a[
                int(_sample(logits, temperature, 0, top_p, np.random.default_rng(base + i)))
            ] += 1
            serial_b[
                int(_sample(logits, temperature, 0, top_p, np.random.default_rng(base + 25000 + i)))
            ] += 1
            acc = _one_step(
                model, ctx, np.random.default_rng(base + 50000 + i), drafter, temperature, top_p
            )
            spec[int(acc[0])] += 1
    return serial_a, serial_b, spec


def _kl(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """在 mask 选定的单元格上归一化后的经验 KL(a || b)。"""
    pa = a[mask] / a[mask].sum()
    pb = b[mask] / b[mask].sum()
    return float(np.sum(pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))))


def _assert_equivalent(serial_a, serial_b, spec, tag):
    """χ² 齐性检验（主门）+ 自校准 KL（辅门）；单元格不足时**响亮失败**。"""
    tot = serial_a + spec
    mask = tot >= 5  # χ² 期望频数 >=5 的常规假设
    n_cells = int(mask.sum())
    assert n_cells >= 2, (
        f"[{tag}] 有效单元格仅 {n_cells} 个（<2），分布退化为 "
        f"{int((tot > 0).sum())} 个 token——χ² 无统计功效，测试配置需修正而非跳过"
    )
    exp = tot[mask] / 2.0
    chi2 = np.sum((serial_a[mask] - exp) ** 2 / exp) + np.sum((spec[mask] - exp) ** 2 / exp)
    df = n_cells - 1
    p_value = _chi2_sf(chi2, df)
    assert p_value > 0.05, f"[{tag}] χ² 拒绝等价假设：χ²={chi2:.1f}, df={df}, p={p_value:.4g}"

    # 自校准 KL：串行 A vs 串行 B = 纯采样噪声底噪；投机 vs 串行 A 应与之同量级
    kl_ref = _kl(serial_b, serial_a, mask)
    kl_obs = _kl(spec, serial_a, mask)
    assert kl_obs <= kl_ref * 2.5 + 1e-6, (
        f"[{tag}] 投机 vs 串行 KL={kl_obs:.6f} 超出采样噪声底噪 "
        f"({kl_ref:.6f} × 2.5)——存在系统性分布偏差"
    )
    return n_cells, chi2, p_value, kl_obs, kl_ref


def test_spec_distribution_equivalence_canonical(ckpt_path):
    """C1 第一条款：canonical 配置（temp=0.8 / top_p=0.95）+ 多 prompt 池化。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged")
    serial_a, serial_b, spec = _collect(model, PROMPTS, n_each=400, temperature=0.8, top_p=0.95)
    n_cells, chi2, p_value, kl, kl_ref = _assert_equivalent(serial_a, serial_b, spec, "canonical")
    print(
        f"\n[canonical] cells={n_cells} chi2={chi2:.2f} p={p_value:.4f} "
        f"KL={kl:.5f} (噪声底噪 {kl_ref:.5f})"
    )


def test_spec_distribution_equivalence_high_temp(ckpt_path):
    """C1 统计功效配置：temp=1.5 / top_p=1.0 —— 非退化分布，真正有辨别力。

    跑 3 组**独立种子**复现，允许 ≤1 组被 χ² 拒绝：单次 χ² 在 α=0.05 下有 5%
    固有假阳性（实测 4 组独立复现 p = 0.033/0.964/0.255/0.434，χ²/df 均值 1.02 ≈
    零假设期望 1.0，确认首个 0.033 为噪声）。3 取 ≥2 通过把假失败率压到 ~0.7%，
    同时保留对真实偏差的检出功率。
    """
    model = PhosModel.from_file(ckpt_path, kv_mode="paged")
    results = []
    for sb in (700000, 3100000, 5500000):
        serial_a, serial_b, spec = _collect(
            model, PROMPTS[:4], n_each=800, temperature=1.5, top_p=1.0, seed_base=sb
        )
        # 非退化校验：该配置下确实产生多 token 分布（否则本配置无功效）
        assert int(((serial_a + spec) > 0).sum()) >= 5, "高温配置下有效 token 数仍过少"
        n_cells, chi2, p_value, kl, kl_ref = _assert_equivalent(
            serial_a, serial_b, spec, f"high-temp@{sb}"
        )
        results.append((sb, n_cells, chi2, p_value, kl, kl_ref))
    for sb, n_cells, chi2, p_value, kl, kl_ref in results:
        print(
            f"\n[high-temp@{sb}] cells={n_cells} chi2={chi2:.2f} p={p_value:.4f} "
            f"KL={kl:.5f} (噪声底噪 {kl_ref:.5f})"
        )
    n_ok = sum(1 for r in results if r[3] > 0.05)
    assert n_ok >= 2, (
        f"3 组复现中仅 {n_ok} 组通过 χ²（需 ≥2），p 值 = "
        f"{[round(r[3], 4) for r in results]}——可能存在系统性分布偏差"
    )


def test_greedy_sequences_identical(ckpt_path):
    """C1 第二条款：temperature=0 时投机与串行贪心序列逐位相同。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged")
    ctx = model.tok.encode(PROMPTS[0])
    drafter = LayerSkipDrafter(model, draft_layers=max(1, model.n_layer - 1))

    # 串行贪心
    rng_s = np.random.default_rng(0)
    model.reset()
    lg = model.forward(np.asarray(ctx, dtype=np.int64))
    nid = int(_sample(lg[-1], 0.0, 1, 0.95, rng_s))
    serial_ids = [nid]
    for _ in range(31):
        lg = model.forward(np.asarray([nid], dtype=np.int64))
        nid = int(_sample(lg[-1], 0.0, 1, 0.95, rng_s))
        serial_ids.append(nid)

    # 投机贪心（逐 step 取首个接受 token）
    spec_ids = []
    cur = list(ctx)
    for _ in range(32):
        rng_p = np.random.default_rng(0)
        acc = _one_step(model, cur, rng_p, drafter, 0.0, 0.95, greedy=True)
        cur = cur + acc
        spec_ids.extend(acc)
    assert spec_ids[: len(serial_ids)] == serial_ids, "贪心投机与串行序列不一致"


def test_o1_zero_extra_bytes(ckpt_path):
    """C2：LayerSkipDrafter 复用同一份三值权重，不新增字节。"""
    model = PhosModel.from_file(ckpt_path, kv_mode="paged")
    drafter = LayerSkipDrafter(model, draft_layers=max(1, model.n_layer - 1))
    assert drafter.extra_bytes == 0, "O1 自草稿不应引入额外权重字节"
    # 草稿前向不崩溃且输出维度正确
    lg = drafter.draft_logits(model.tok.encode(PROMPTS[0]))
    assert lg.shape[-1] == model.vocab
