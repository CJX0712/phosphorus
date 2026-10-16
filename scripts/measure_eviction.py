"""T4-C 驱逐保真度实测（修正 q_start 后复测，用于文档 BENCHMARKS）。"""
import numpy as np

from phosphorus.engine import PhosModel

CKPT = "ckpt/tiny.phos"
PREF = list(range(10, 210))      # 200 token 预填充（触发压缩）
DEC = list(range(210, 230))      # 20 个 decode
POLS = ["snapkv", "pyramidkv", "h2o", "streamingllm"]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ref_logits():
    m = PhosModel.from_file(CKPT, eviction="none")
    m.forward(np.array(PREF, dtype=np.int64))
    return [m.forward(np.array([t], dtype=np.int64))[-1] for t in DEC]


def fidelity_at(budget):
    ref = ref_logits()
    rows = []
    for pol in POLS:
        m = PhosModel.from_file(CKPT, eviction=pol, kv_budget=budget)
        m.forward(np.array(PREF, dtype=np.int64))
        coss = [cos(m.forward(np.array([t], dtype=np.int64))[-1], ref[j]) for j, t in enumerate(DEC)]
        rows.append((pol, float(np.mean(coss))))
    return rows


def budget_conservation():
    from phosphorus.eviction import EvictionConfig, PyramidKVPolicy
    out = []
    for n_layer, seq_len, ratio in [(3, 256, 0.5), (8, 256, 0.5), (32, 1000, 0.25), (8, 100, 0.75)]:
        pol = PyramidKVPolicy(EvictionConfig(budget_ratio=ratio))
        b = pol.layer_budgets(n_layer, seq_len)
        C = int(round(seq_len * ratio))
        out.append((n_layer, seq_len, ratio, sum(b), n_layer * C, abs(sum(b) - n_layer * C)))
    return out


print("=== 下游 logits 余弦（vs 无驱逐参考），越高越保真 ===")
for bud in (0.5, 0.75):
    print(f"\n预算 = {int(bud*100)}% 序列长:")
    for pol, c in fidelity_at(bud):
        print(f"  {pol:12s} mean_cos = {c:.4f}")

print("\n=== 金字塔预算守恒 |Σ − L·C| ≤ 1 ===")
for n_layer, seq_len, ratio, s, target, diff in budget_conservation():
    print(f"  L={n_layer:2d} len={seq_len:4d} r={ratio}  Σ={s}  L·C={target}  |Δ|={diff}")
