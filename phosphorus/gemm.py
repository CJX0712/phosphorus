"""GEMM 后端：fp32 / int8 / ternary_tl / ternary_i2s（ARCHITECTURE.md §5）。

借鉴来源：
    * bitnet.cpp 的 TL（Ternary LUT）与 I2_S（int2 + scale）两类 mpGEMM — A-3
    * 不变量 ①：TL 与 I2_S 精确计算同一个整数式 Σ_j W[i,j]·xq[j]，
      因此在 **int32 域逐位相等**；这是本项目可证明的正确性锚点。

四个后端的语义（诚实定义，不做漂亮但虚假的包装）：
    * `fp32`       —— 反量化后的 fp32 矩阵乘，量化误差的对照基线。
    * `int8`       —— 权重/激活各自 absmax 量化到 int8，int32 累加后反量化。
    * `ternary_tl` —— **真·LUT 路径**：吃 5-bit 打包流的 code5，查 14 项 LUT
                      （展开为 32 项按 code5 直查表），无乘法，int32 累加。
    * `ternary_i2s`—— **真·零点校正路径**：权重 int8 镜像 × uint8 激活，
                      末尾减 128·row_wsum；与 TL 数学恒等。

`*_i32` 系列返回**纯整数累加**（不做任何浮点缩放），是位精确断言的唯一合法入口。
"""

from __future__ import annotations

import numpy as np

from .packing import (
    _DIGIT_MINUS_ONE,
    BYTES_PER_CHUNK,
    GROUPS_PER_CHUNK,
    WEIGHTS_PER_CHUNK,
    k_pad,
    row_wsum,
    unpack_code5,
)

# --------------------------------------------------------------------------- #
# 量化基元                                                                     #
# --------------------------------------------------------------------------- #


def quantize_activation(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """BitNet per-token 激活量化：返回 (xq int8, x_scale float32)。

    `x_hat = clip(round(x · Q_b / γ), -Q_b, Q_b)`，`γ = max|x|`（per-token），
    `Q_b = 127`（8 bit）。反量化为 `x_hat · γ / Q_b`，与引擎/训练侧一致。
    """
    x = np.asarray(x, dtype=np.float32)
    qb = float((1 << (bits - 1)) - 1)  # 8 bit -> 127
    if x.ndim == 1:
        amax = np.array([np.max(np.abs(x)) if x.size else 0.0], dtype=np.float32)
    else:
        amax = np.max(np.abs(x), axis=-1).astype(np.float32)
    xs = np.where(amax > 0, amax / qb, np.float32(1.0)).astype(np.float32)
    xq = np.clip(np.round(x / xs.reshape((-1,) + (1,) * (x.ndim - 1))), -qb, qb).astype(np.int8)
    return xq, xs


def int32_reference(W_ternary: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """权威整数参考：acc[i] = Σ_j W[i,j]·xq[j]（int32）。不变量 ① 的基准。"""
    W = np.asarray(W_ternary, dtype=np.int32)
    X = np.asarray(xq, dtype=np.int32).reshape(-1)
    if W.ndim == 1:
        return np.array([int(np.dot(W, X[: W.shape[0]]))], dtype=np.int32)
    return (W @ X[: W.shape[1]]).astype(np.int32)


# --------------------------------------------------------------------------- #
# LUT 构建（TL 内核）                                                          #
# --------------------------------------------------------------------------- #


def build_lut_tables(xq: np.ndarray, ng: int) -> np.ndarray:
    """为每个 3 权重组构建「按 code5 直查」的 32 项表，返回 [ng, 32] int32。

    lut[g, c+1] = (d0-1)·a0 + (d1-1)·a1 + (d2-1)·a2（c ∈ [0,12]，d = digit(c)）；
    lut[g, 0] = 0（全零组）；idx ∈ [14,15] 为 SIMD 补齐项，恒 0。
    sign=1 时取相反数（part(26-c) == -part(c)），故只需 13 项实体 + 1 个零项。
    """
    a = np.asarray(xq, dtype=np.int32).reshape(-1)[: ng * 3].reshape(ng, 3)
    lut = np.zeros((ng, 16), dtype=np.int32)
    lut[:, 1:14] = a @ _DIGIT_MINUS_ONE[:13].T  # [ng, 13]，无 Python 循环
    idx = np.arange(32, dtype=np.int64)
    sign = (idx >> 4) & 1
    low = idx & 0x0F
    valid = low < 14
    table = lut[:, np.where(valid, low, 0)] * np.where(sign == 1, -1, 1).astype(np.int32)
    table[:, ~valid] = 0
    return table.astype(np.int32)


# --------------------------------------------------------------------------- #
# 四个后端                                                                     #
# --------------------------------------------------------------------------- #


def gemm_fp32(W_fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """fp32 基线：反量化权重 × fp32 激活。返回 [N] 或 [T, N]。"""
    W = np.asarray(W_fp, dtype=np.float32)
    X = np.asarray(x, dtype=np.float32)
    return W @ X if X.ndim == 1 else (W @ X.T).T


def gemm_int8(W_fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """真·int8 后端：权重与激活各自 absmax 量化到 int8，int32 累加后反量化。

    注意：这与三值路径**不是**同一个整数域（权重被压到 int8 而非 {-1,0,1}），
    因此它带独立的量化误差，用于「int8 vs 三值」的精度/速度对照。
    """
    W = np.asarray(W_fp, dtype=np.float32)
    X = np.asarray(x, dtype=np.float32)
    w_scale = np.max(np.abs(W), axis=-1, keepdims=True) / 127.0
    w_scale = np.where(w_scale > 0, w_scale, 1.0).astype(np.float32)
    wq = np.clip(np.round(W / w_scale), -127, 127).astype(np.int32)
    xq, xs = quantize_activation(X)
    xq = xq.astype(np.int32)
    if X.ndim == 1:
        acc = wq @ xq
        scale = w_scale.reshape(-1) * xs[0]
    else:
        acc = (wq @ xq.T).T
        scale = w_scale.reshape(1, -1) * xs.reshape(-1, 1)
    return (acc.astype(np.float64) * scale.astype(np.float64)).astype(np.float32)


def tl_acc_i32(codes: np.ndarray, xq: np.ndarray, K: int) -> np.ndarray:
    """TL 内核的整数累加（int32）：按 code5 查 LUT 后求和。不变量 ① 的 TL 侧。"""
    Kpad = k_pad(K)
    ng = Kpad // 3
    codes = np.asarray(codes, dtype=np.int64).reshape(-1, ng)
    xv = np.asarray(xq, dtype=np.int32).reshape(-1)
    if xv.size < Kpad:
        xv = np.concatenate([xv, np.zeros(Kpad - xv.size, dtype=np.int32)])
    table = build_lut_tables(xv, ng)  # [ng, 32]
    g = np.arange(ng, dtype=np.int64).reshape(1, ng)
    vals = table[g, codes]  # [N, ng]：真·查表，无乘法
    return vals.sum(axis=1, dtype=np.int32)


def i2s_acc_i32(W_i8: np.ndarray, xq: np.ndarray, row_sum: np.ndarray) -> np.ndarray:
    """I2_S 内核的整数累加（int32）：xu8·Wᵀ − 128·row_wsum。不变量 ① 的 I2_S 侧。"""
    W = np.asarray(W_i8, dtype=np.int32)
    xq = np.asarray(xq, dtype=np.int32).reshape(-1)
    xu8 = xq + 128  # uint8 激活（零点 = 128）
    acc = W @ xu8[: W.shape[1]] - 128 * np.asarray(row_sum, dtype=np.int32)
    return acc.astype(np.int32)


def gemm_ternary_tl(
    packed: np.ndarray, xq: np.ndarray, w_scale: np.ndarray, x_scale: float, N: int, K: int
) -> np.ndarray:
    """TL 后端（打包域）：返回 fp32 输出 [N] = acc_int32 · w_scale · x_scale。"""
    codes = (
        packed
        if np.asarray(packed).ndim == 2
        else unpack_code5(packed, N, K).astype(np.int64)
    )
    acc = tl_acc_i32(codes, xq, K)
    return (acc.astype(np.float64) * np.asarray(w_scale, np.float64) * float(x_scale)).astype(
        np.float32
    )


def gemm_ternary_i2s(
    W_ternary: np.ndarray,
    xq: np.ndarray,
    w_scale: np.ndarray,
    x_scale: float,
    row_sum: np.ndarray | None = None,
) -> np.ndarray:
    """I2_S 后端：返回 fp32 输出 [N] = acc_int32 · w_scale · x_scale。"""
    W = np.asarray(W_ternary, dtype=np.int8)
    if row_sum is None:
        row_sum = row_wsum(W)
    acc = i2s_acc_i32(W, xq, row_sum)
    return (acc.astype(np.float64) * np.asarray(w_scale, np.float64) * float(x_scale)).astype(
        np.float32
    )


# --------------------------------------------------------------------------- #
# 不变量 ① 断言                                                                #
# --------------------------------------------------------------------------- #


def tl_i2s_bitwise_equal(W_ternary: np.ndarray, xq: np.ndarray, row_sum: np.ndarray) -> bool:
    """不变量 ①（密集权重版）：TL 与 I2_S 的 int32 累加逐位相等。"""
    tl = int32_reference(W_ternary, xq)
    i2s = i2s_acc_i32(W_ternary, xq, row_sum)
    return bool(np.array_equal(tl.astype(np.int32), i2s.astype(np.int32)))


def tl_i2s_bitwise_equal_packed(
    packed: np.ndarray, W_ternary: np.ndarray, xq: np.ndarray, row_sum: np.ndarray, K: int
) -> bool:
    """不变量 ①（打包版，更强）：从 **5-bit 打包字节** 解出的 code5 走 LUT 路径，
    其 int32 输出须与 I2_S 路径及 int32_reference **三者逐位相等**。"""
    N = np.asarray(W_ternary).shape[0]
    codes = unpack_code5(packed, N, K)
    tl = tl_acc_i32(codes, xq, K)
    i2s = i2s_acc_i32(W_ternary, xq, row_sum)
    ref = int32_reference(W_ternary, xq)
    return bool(
        np.array_equal(tl.astype(np.int32), i2s.astype(np.int32))
        and np.array_equal(tl.astype(np.int32), ref.astype(np.int32))
    )


def packed_bytes_of(N: int, K: int) -> int:
    """打包字节数的理论值（供导出/测试核对）。"""
    return N * (k_pad(K) // WEIGHTS_PER_CHUNK) * BYTES_PER_CHUNK


__all__ = [
    "GROUPS_PER_CHUNK",
    "WEIGHTS_PER_CHUNK",
    "build_lut_tables",
    "gemm_fp32",
    "gemm_int8",
    "gemm_ternary_i2s",
    "gemm_ternary_tl",
    "i2s_acc_i32",
    "int32_reference",
    "packed_bytes_of",
    "quantize_activation",
    "row_wsum",
    "tl_acc_i32",
    "tl_i2s_bitwise_equal",
    "tl_i2s_bitwise_equal_packed",
]
