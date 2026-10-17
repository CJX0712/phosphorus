"""构造一个更深的**合成**三值模型，用于演示投机解码的加速比（T5 · C3）。

为什么需要它：投机解码的 wall-clock 收益只在「草稿远浅于全模型 (draft_layers ≪ n_layer)
且接受率 α 较高」时显现（Leviathan 2023 的经典结论）。`ckpt/tiny.phos` 仅 3 层，每步前向
仅微秒级，额外草稿调用反而占优 → 净中性（已在 bench_speculate 实测 ~0.9×）。本脚本构造一个
8 层模型，并把**深层权重整体缩小**（模拟「深层只做小修正」的良态训练模型），使 early-exit
(2~3 层) 与全模型 (8 层) 高度一致 → 高 α，从而让投机解码的加速比真实落地。

注意：权重为随机三值，不产出可懂文本；本模型仅用于**性能/正确性机制**演示，不用于质量评测。
分布等价性（不变量 ②）已由 tiny.phos 上的 χ²/KL 硬门证明。
"""

from __future__ import annotations

import argparse

import numpy as np

from phosphorus.config import ModelConfig, QuantConfig
from phosphorus.export import export_model


def build(
    cfg: ModelConfig,
    deep_scale: float,
    out_path: str,
    logit_scale: float = 1e-4,
    base_scale: float = 1.0,
) -> str:
    d = cfg.d_model
    kv = cfg.n_kv_head * (d // cfg.n_head)
    d_ff = cfg.d_ff
    V = cfg.vocab_size
    ternary = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def shapes(proj: str) -> tuple[int, int]:
        if proj in ("q_proj", "o_proj"):
            return d, d
        if proj in ("k_proj", "v_proj"):
            return kv, d
        if proj in ("gate_proj", "up_proj"):
            return d_ff, d
        return d, d_ff  # down_proj

    w: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(7)
    # 嵌入 / 终归一化。logit_scale 压低 norm_final：随机三值权重（未经训练）的
    # 残差流幅值会随层累积到 ~1e4，若不压低，最终 softmax 数值下溢成 one-hot，
    # 使 p/q 比值大量为 0、接受率崩塌（数值病理，非算法问题）。
    w["tok_emb"] = (rng.standard_normal((V, d)) * 0.05).astype(np.float32)
    w["norm_final"] = ((rng.standard_normal(d) * 0.1 + 0.5) * logit_scale).astype(np.float32)

    for i in range(cfg.n_layer):
        # 深层整体缩小：前 3 层按 base_scale（良态化残差流，防注意力分数爆炸），
        # 其余按 deep_scale（0 = 精确恒等层，使 early-exit ≡ full）。
        scale_i = base_scale if i < 3 else deep_scale
        for p in ("attn_norm", "ffn_norm", "subln_attn", "subln_ffn"):
            w[f"layers.{i}.{p}"] = (rng.standard_normal(d) * 0.1 + 0.5).astype(np.float32)
        for proj in ternary:
            out, in_ = shapes(proj)
            t = rng.integers(-1, 2, size=(out, in_)).astype(np.int8)
            w[f"layers.{i}.{proj}.weight"] = t
            # per-channel scale 承载 depth_scale：直接缩放整层贡献。
            # 注意键名：export 按去掉 .weight 的 base_key + ".scale" 查找。
            w[f"layers.{i}.{proj}.scale"] = np.full(out, scale_i, dtype=np.float32)

    qcfg = QuantConfig(scheme="ternary", block_size=128, hgf_rank=0, hgf_alpha=0.0)
    blob = export_model(w, cfg, qcfg, tokenizer=None, path=out_path)
    if out_path is None:
        with open(out_path := "__tmp_synth.phos", "wb") as f:
            f.write(blob)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpt/synthetic_8l.phos")
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--deep-scale", type=float, default=0.12)
    ap.add_argument("--logit-scale", type=float, default=1e-4)
    ap.add_argument("--base-scale", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=256)
    args = ap.parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        d_ff=args.d_ff,
        vocab_size=args.vocab,
        max_seq_len=args.max_seq_len,
        tied_embed=True,
        early_exit_layer=-1,
    )
    path = build(cfg, args.deep_scale, args.out, args.logit_scale, args.base_scale)
    print(f"wrote synthetic model -> {path} (n_layer={args.n_layer}, deep_scale={args.deep_scale})")


if __name__ == "__main__":
    main()
