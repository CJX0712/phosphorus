"""T4-A 内核级基准：native C99 内核（scalar / AVX2）vs numpy 参考实现。

与 docs/BENCHMARKS.md §T4-A「内核级」表对应。两种内核都是**逐 token**
GEMV 形态（与引擎 decode 一致）；numpy 参考分别是：
* i2s —— int32 批量矩阵乘 `W_i32 @ xu8_i32.T`（引擎 prefill 实际路径）
* tl  —— 从 5-bit 打包流解 code5 后的 LUT 累加，逐 token Python 循环
  （引擎 numpy TL 路径的实际形态）

用法::

    python -m scripts.bench_kernels --reps 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phosphorus.gemm import tl_acc_i32  # noqa: E402
from phosphorus.native import PHOS_FLAG_AVX2, PHOS_FLAG_SCALAR, kernels  # noqa: E402
from phosphorus.packing import k_pad, pack_ternary_v1, row_wsum, unpack_code5  # noqa: E402

SHAPES = [(192, 192), (512, 512), (1024, 256), (4096, 192), (4096, 384)]
TOKENS = [1, 8]


def _best(fn, reps: int) -> float:
    """返回 fn() 的 best-of-reps 耗时（秒）。"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _i2s_native_loop(k, W, xu8, rw, T: int, flags: int) -> None:
    for t in range(T):
        k.i2s_gemm_i32(W, xu8[t], rw, flags=flags)


def _tl_native_loop(k, packed, xpad, N: int, Kp: int, T: int, flags: int) -> None:
    for t in range(T):
        k.tl_gemm_i32(packed, xpad[t], N, Kp, flags=flags)


def _tl_numpy_loop(codes, xpad, Kp: int, T: int) -> None:
    for t in range(T):
        tl_acc_i32(codes, xpad[t], Kp)


def _i2s_numpy_matmul(W, xq, rw):
    """numpy 参考：int32 批量矩阵乘 + I2_S 零点校正（引擎 prefill 路径）。"""
    return (W @ (xq.astype(np.int32) + 128).T) - rw.astype(np.int32).reshape(-1, 1) * 128


def main() -> int:
    ap = argparse.ArgumentParser(description="phosphorus T4-A 内核级基准")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    reps = args.reps
    k = kernels()
    avx2 = bool(k.cpu_has_avx2())
    print(f"[info] AVX2 available: {avx2}  reps={reps}")

    rows = []
    for N, K in SHAPES:
        rng = np.random.default_rng(N * 1000 + K)
        W = rng.integers(-1, 2, size=(N, K)).astype(np.int32)
        Wi8 = W.astype(np.int8)
        rw = row_wsum(Wi8)
        Kp = k_pad(K)
        Wpad = np.zeros((N, Kp), dtype=np.int8)
        Wpad[:, :K] = Wi8
        packed, _ = pack_ternary_v1(Wpad)
        codes = unpack_code5(packed, N, Kp).astype(np.int64)
        for T in TOKENS:
            x = rng.standard_normal((T, K)).astype(np.float32)
            xq = np.clip(np.round(x * 100.0), -127, 127).astype(np.int8)
            xu8 = (xq.astype(np.int32) + 128).astype(np.uint8)
            xpad = np.zeros((T, Kp), dtype=np.int8)
            xpad[:, :K] = xq

            # ---------- 正确性（逐 token 累加后与 numpy 参考比对） ----------
            ref = np.stack(
                [(W @ xq[t].astype(np.int32)) for t in range(T)], axis=1
            )  # [N, T] int32 权威参考
            ref_lut = np.stack([tl_acc_i32(codes, xpad[t], Kp) for t in range(T)], axis=1)

            # ---------- i2s 原生（逐 token，T 次调用总耗时） ----------
            got_i2s = np.stack(
                [
                    k.i2s_gemm_i32(
                        Wi8, xu8[t], rw, flags=PHOS_FLAG_AVX2 if avx2 else PHOS_FLAG_SCALAR
                    )
                    for t in range(T)
                ],
                axis=1,
            )
            ok_i2s = np.array_equal(got_i2s, ref)
            t_i2s_sc = _best(partial(_i2s_native_loop, k, Wi8, xu8, rw, T, PHOS_FLAG_SCALAR), reps)
            t_i2s_av = (
                _best(partial(_i2s_native_loop, k, Wi8, xu8, rw, T, PHOS_FLAG_AVX2), reps)
                if avx2
                else t_i2s_sc
            )

            # ---------- tl 原生（逐 token） ----------
            got_tl = np.stack(
                [
                    k.tl_gemm_i32(
                        packed, xpad[t], N, Kp, flags=PHOS_FLAG_AVX2 if avx2 else PHOS_FLAG_SCALAR
                    )
                    for t in range(T)
                ],
                axis=1,
            )
            ok_tl = np.array_equal(got_tl, ref_lut)
            t_tl_sc = _best(
                partial(_tl_native_loop, k, packed, xpad, N, Kp, T, PHOS_FLAG_SCALAR), reps
            )
            t_tl_av = (
                _best(partial(_tl_native_loop, k, packed, xpad, N, Kp, T, PHOS_FLAG_AVX2), reps)
                if avx2
                else t_tl_sc
            )

            # ---------- numpy 参考 ----------
            # i2s: 批量矩阵乘（引擎 prefill 路径）；tl: 逐 token LUT（引擎 numpy 路径）
            t_np_i2s = _best(partial(_i2s_numpy_matmul, W, xq, rw), reps)
            np_reps = 1 if N >= 1024 else reps  # tl numpy 参考极慢，大形状只跑 1 次
            t_np_tl = _best(partial(_tl_numpy_loop, codes, xpad, Kp, T), np_reps)

            rows.append(
                dict(
                    N=N,
                    K=K,
                    T=T,
                    ok_i2s=ok_i2s,
                    ok_tl=ok_tl,
                    i2s_sc=t_i2s_sc,
                    i2s_av=t_i2s_av,
                    np_i2s=t_np_i2s,
                    tl_sc=t_tl_sc,
                    tl_av=t_tl_av,
                    np_tl=t_np_tl,
                )
            )

    print()
    print(
        "| N | K | T | i2s 正确 | i2s scalar vs numpy | i2s AVX2 vs numpy | tl 正确 | tl scalar vs numpy | tl AVX2 vs numpy |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(
            f"| {r['N']} | {r['K']} | {r['T']}"
            f" | {'✅' if r['ok_i2s'] else '❌'}"
            f" | {r['np_i2s'] / r['i2s_sc']:.2f}x | {r['np_i2s'] / r['i2s_av']:.2f}x"
            f" | {'✅' if r['ok_tl'] else '❌'}"
            f" | {r['np_tl'] / r['tl_sc']:.2f}x | {r['np_tl'] / r['tl_av']:.2f}x |"
        )
    print()
    print("注：加速比 = numpy 参考耗时 / native 耗时（同一 (N,K,T)）。i2s 的 numpy 参考是 int32")
    print("    批量矩阵乘（引擎 prefill 路径），tl 的 numpy 参考是逐 token LUT Python 循环（引擎")
    print("    numpy TL 路径）。两者都包含 numpy 调用开销，与引擎实际路径同构。")
    bad = [r for r in rows if not (r["ok_i2s"] and r["ok_tl"])]
    if bad:
        print(f"[FAIL] {len(bad)} 个形状位不一致")
        return 1
    print("[ok] 全部形状位一致（不变量 ①）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
