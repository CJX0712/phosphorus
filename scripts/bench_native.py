"""T4-A 基准：native C99 内核 vs numpy 实现（端到端，可复现）。

与 docs/BENCHMARKS.md §T4-A 对应。两种「数学后端」（i2s / tl）分别对比两种
「实现后端」（numpy / native）；两条路径在 int32 域逐位相等（不变量 ①），
因此本脚本同时打印一次位一致性校验，失败即非零退出。

用法::

    python -m scripts.bench_native --model ckpt/tiny.phos --reps 5

环境变量:
    PHOS_BACKEND   覆盖数学后端（默认 fp32）
    PHOS_NATIVE_DIR 原生库缓存目录
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phosphorus.engine import PhosModel  # noqa: E402

PROMPT_LEN = 32
DECODE_TOKENS = 64


def _forward_once(model: PhosModel, prompt_len: int, decode_tokens: int) -> tuple[float, float]:
    """单次 prefill + decode，返回两者耗时（秒）。"""
    start = model.bos_id if (model.bos_id is not None and 0 <= model.bos_id < model.vocab) else 0
    prompt = np.array([start] * prompt_len, dtype=np.int64)
    model.reset()
    t0 = time.perf_counter()
    logits = model.forward(prompt)
    prefill = time.perf_counter() - t0
    next_id = int(np.argmax(logits[-1]))
    t1 = time.perf_counter()
    for _ in range(decode_tokens):
        lg = model.forward(np.array([next_id], dtype=np.int64))
        next_id = int(np.argmax(lg[-1]))
    decode = time.perf_counter() - t1
    return prefill, decode


def _bench(path: str, math: str, impl: str, reps: int) -> dict[str, float]:
    prefills: list[float] = []
    decodes: list[float] = []
    for _ in range(reps):
        model = PhosModel.from_file(path, backend=math, native=impl)
        p, d = _forward_once(model, PROMPT_LEN, DECODE_TOKENS)
        prefills.append(p)
        decodes.append(d)
        del model
    return {
        "prefill_best": min(prefills),
        "prefill_med": statistics.median(prefills),
        "decode_best": min(decodes),
        "decode_med": statistics.median(decodes),
    }


def _bitwise_check(path: str) -> bool:
    """不变量 ①：i2s/tl 的 native 与 numpy 实现在 int32/fp32 域逐位相等。"""
    from phosphorus.gemm import gemm_ternary_i2s, gemm_ternary_tl, quantize_activation
    from phosphorus.native import kernels
    from phosphorus.packing import k_pad, pack_ternary_v1

    rng = np.random.default_rng(7)
    k = kernels()
    ok = True
    for N, K in ((64, 96), (192, 192)):
        W = rng.integers(-1, 2, size=(N, K)).astype(np.int32)
        x = rng.standard_normal((3, K)).astype(np.float32)
        ws = np.ones(N, dtype=np.float32)
        xq, xs = quantize_activation(x)
        rw = W.astype(np.int32).sum(axis=1)
        # --- i2s：numpy 参考 vs native f32 ---
        for t in range(3):
            ref = gemm_ternary_i2s(W, xq[t], ws, float(xs[t]), rw)
            got = k.i2s_gemm_f32(
                W.astype(np.int8),
                (xq[t].astype(np.int32) + 128).astype(np.uint8),
                rw,
                ws,
                float(xs[t]),
            )
            same = np.array_equal(ref.astype(np.float64), got.astype(np.float64))
            ok &= bool(same)
            if t == 0 or not same:
                print(f"  [check] i2s N={N} K={K} t={t} native==numpy bitwise: {same}")
        # --- tl：numpy 参考 vs native f32 ---
        Kp = k_pad(K)
        Wpad = np.zeros((N, Kp), dtype=np.int8)
        Wpad[:, :K] = W.astype(np.int8)
        packed, _ = pack_ternary_v1(Wpad)
        xpad = np.zeros((3, Kp), dtype=np.int8)
        xpad[:, :K] = xq
        for t in range(3):
            ref_t = gemm_ternary_tl(packed, xq[t], ws, float(xs[t]), N, Kp)
            got_t = k.tl_gemm_f32(packed, xpad[t], ws, float(xs[t]), N, Kp)
            same_t = np.array_equal(ref_t.astype(np.float64), got_t.astype(np.float64))
            ok &= bool(same_t)
            if t == 0 or not same_t:
                print(f"  [check] tl  N={N} K={K} t={t} native==numpy bitwise: {same_t}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="phosphorus T4-A native vs numpy 基准")
    ap.add_argument("--model", default="ckpt/tiny.phos")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    print(f"[info] model={args.model} prompt={PROMPT_LEN} decode={DECODE_TOKENS} reps={args.reps}")
    if not args.skip_check:
        print("[check] 位一致性（不变量 ①）:")
        if not _bitwise_check(args.model):
            print("[FAIL] native 与 numpy 结果不一致，基准无意义")
            return 1

    rows: list[tuple[str, str, dict[str, float]]] = []
    for math in ("i2s", "tl"):
        for impl in ("numpy", "native"):
            try:
                r = _bench(args.model, math, impl, args.reps)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {math}/{impl} 失败: {exc}")
                continue
            rows.append((math, impl, r))
            print(
                f"[bench] {math:>4} {impl:>6}: prefill_best={PROMPT_LEN / r['prefill_best']:8.1f} tok/s"
                f"  decode_best={DECODE_TOKENS / r['decode_best']:7.1f} tok/s"
                f"  (prefill_med={PROMPT_LEN / r['prefill_med']:7.1f}, decode_med={DECODE_TOKENS / r['decode_med']:7.1f})"
            )

    if not rows:
        return 1
    print()
    print(
        "| 数学后端 | 实现后端 | prefill tok/s (best) | decode tok/s (best) | prefill 加速比 | decode 加速比 |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for math in ("i2s", "tl"):
        base = next((r for m, i, r in rows if m == math and i == "numpy"), None)
        for m, impl, r in rows:
            if m != math:
                continue
            if base is None:
                continue
            print(
                f"| {math} | {impl} | {PROMPT_LEN / r['prefill_best']:.1f} | {DECODE_TOKENS / r['decode_best']:.1f}"
                f" | {base['prefill_best'] / r['prefill_best']:.2f}x | {base['decode_best'] / r['decode_best']:.2f}x |"
            )
    print()
    print("注：i2s 的 prefill 在 native 模式下仍走 numpy 批量矩阵乘——实测 crossover 显示")
    print("    T>1 时 numpy 更优（见 phosphorus/engine.py `_use_native`），故该行仅作噪声对照；")
    print("    tl 的 prefill/decode 均走原生内核。加速比 = numpy 耗时 / native 耗时。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
