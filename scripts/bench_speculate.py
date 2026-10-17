"""T5 · C3/C5/C6 投机解码基准：热 KV 增量解码吞吐 vs 串行，含接受率与加速比矩阵。

用法：
    python -m benchmarks.bench_speculate --model ckpt/tiny.phos \
        --prompt "To be, or not to be" --tokens 256 --seeds 3 \
        --K 2,4,8 --draft-layers 1,2

输出一张矩阵（K × draft_layers × 模式），给出 tok/s、加速比、平均接受率 α，
供 BENCHMARKS.md §T5 直接引用。零 torch 依赖；纯 numpy 推理。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from phosphorus import speculate as _spec
from phosphorus.engine import PhosModel
from phosphorus.speculate import LayerSkipDrafter, generate_tokens


def _run(model, ctx, spec_mode, K, draft_layers, temperature, top_p, seed, n_tokens):
    """跑一次生成，返回 (生成 token 数, 用时秒, 平均接受率 α, 文本)。

    串行与投机两条路径都走 `generate_tokens`（支持无分词器的合成模型，直接用 token id）。
    """
    drafter = None
    draft_model = None
    controller = None
    if spec_mode != "off":
        drafter = LayerSkipDrafter(model, draft_layers=draft_layers)
        draft_model = model.shadow()
        if spec_mode == "auto":
            controller = _spec.SpecController()
    t0 = time.time()
    gen, alpha, _info = generate_tokens(
        model,
        ctx,
        max_new_tokens=n_tokens,
        temperature=temperature,
        top_k=0,
        top_p=top_p,
        seed=seed,
        drafter=drafter,
        K=K,
        controller=controller,
        draft_model=draft_model,
    )
    dt = time.time() - t0
    return (
        len(gen),
        dt,
        (alpha if spec_mode != "off" else 1.0),
        model.tok.decode(ctx + gen) if model.tok else "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phosphorus 投机解码基准")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="To be, or not to be")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", default="2,4,8")
    ap.add_argument("--draft-layers", default="1,2", dest="draft_layers")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--backend", default="fp32")
    ap.add_argument("--kv", default="paged")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    model = PhosModel.from_file(args.model, backend=args.backend, kv_mode=args.kv, eviction="none")
    ctx = model.tok.encode(args.prompt) if model.tok else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Ks = [int(x) for x in args.K.split(",")]
    Dls = [int(x) for x in args.draft_layers.split(",")]

    # 预热（避免首次 numpy/BLAS 启动抖动）
    for _ in range(args.warmup):
        generate_tokens(model, ctx, max_new_tokens=8, drafter=None, seed=0)

    # 串行基线
    s_tok = 0
    s_dt = 0.0
    for s in range(args.seeds):
        nt, dt, _a, _o = _run(model, ctx, "off", 0, 0, args.temperature, args.top_p, s, args.tokens)
        s_tok += nt
        s_dt += dt
    s_rate = s_tok / max(s_dt, 1e-9)

    print(
        f"\n模型: {args.model}  prompt='{args.prompt}'  tokens/seed={args.tokens}  seeds={args.seeds}"
    )
    print(f"串行基线: {s_rate:.1f} tok/s  (生成 {s_tok} token / {s_dt:.2f}s)")
    print(f"\n{'mode':>5} {'K':>3} {'dl':>3} {'tok/s':>9} {'speedup':>9} {'α':>6}")
    print("-" * 44)

    best = (0.0, None)
    for dl in Dls:
        if not (1 <= dl < model.n_layer):
            continue
        for K in Ks:
            if K < 1:
                continue
            t_tok = 0
            t_dt = 0.0
            alpha_sum = 0.0
            for s in range(args.seeds):
                nt, dt, alpha, _o = _run(
                    model, ctx, "on", K, dl, args.temperature, args.top_p, s, args.tokens
                )
                t_tok += nt
                t_dt += dt
                alpha_sum += alpha
            rate = t_tok / max(t_dt, 1e-9)
            speedup = rate / max(s_rate, 1e-9)
            alpha = alpha_sum / args.seeds
            print(f"{'spec':>5} {K:>3} {dl:>3} {rate:>9.1f} {speedup:>9.2f} {alpha:>6.3f}")
            if speedup > best[0]:
                best = (speedup, (K, dl, rate, alpha))

    print("-" * 44)
    if best[1] is not None:
        K, dl, rate, alpha = best[1]
        print(
            f"最佳配置: K={K} draft_layers={dl} → {rate:.1f} tok/s (speedup={best[0]:.2f}×, α={alpha:.3f})"
        )
    else:
        print("无有效投机配置（draft_layers 越界或 K 非法）")

    # ---- auto（O2 控制器自适应）对照：控制器应在不划算时自动关闭投机 ---- #
    dl0 = Dls[0] if Dls else max(1, model.n_layer - 1)
    t_tok, t_dt, a_sum = 0, 0.0, 0.0
    for s in range(args.seeds):
        drafter = LayerSkipDrafter(model, draft_layers=dl0)
        ctrl = _spec.SpecController()
        t0 = time.time()
        gen, alpha, info = generate_tokens(
            model,
            ctx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=0,
            top_p=args.top_p,
            seed=s,
            drafter=drafter,
            K=max(Ks),
            controller=ctrl,
        )
        t_dt += time.time() - t0
        t_tok += len(gen)
        a_sum += alpha
    rate_a = t_tok / max(t_dt, 1e-9)
    print(
        f"{'auto':>5} {'—':>3} {dl0:>3} {rate_a:>9.1f} {rate_a / max(s_rate, 1e-9):>9.2f} {a_sum / args.seeds:>6.3f}"
        f"   终态={ctrl.state} α_hat={ctrl.alpha_hat:.3f} c_hat={ctrl.c:.3f}"
    )


if __name__ == "__main__":
    main()
