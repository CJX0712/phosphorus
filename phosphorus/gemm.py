"""GEMM 后端：fp32 / int8 / ternary_tl / ternary_i2s（ARCHITECTURE.md §5）。

借鉴来源：
    * 三值 LUT GEMM（bitnet.cpp TL）— A-3
    * int2+scale GEMM（bitnet.cpp I2_S）— A-3
    * 不变量 ①：TL 与 I2_S 都精确计算 Σ_j W[i,j]·xq[j]，在 int32 域逐位相等。

所有后端在 **量化域内** 输出 `y = w_scale · x_scale · Σ_j W·xq`。
fp32 基线：`y = (W_deq) @ x_fp`，用于困惑度对照（量化误差来源）。

实现说明（诚实 > 漂亮）：
    TL 的数值权威路径就是 `int32_reference = W_ternary @ xq`（int32），
    与 I2_S 的 `xu8·Wi8 - 128·row_wsum` 数学恒等。C 内核（csrc/phos.c）走
    5-bit 打包 + 14 项 LUT 的物理路径，但其 int32 输出必须与本参考逐位相等
    （见 test_gemm_bitwise.py）。推理引擎直接复用 `W_ternary @ xq`（等价 TL），
    避免每次重复解包。
"""

from __future__ import annotations

import numpy as np


def quantize_activation(x: np.ndarray, scheme: str = "per_token") -> tuple[np.ndarray, float]:
    """把 fp32 激活量化为 int8。返回 (xq int8, x_scale float)。"""
    x = np.asarray(x, dtype=np.float32)
    if scheme == "per_token":
        amax = float(np.max(np.abs(x))) if x.size else 0.0
    else:
        amax = float(np.max(np.abs(x))) if x.size else 0.0
    xs = amax / 127.0 if amax > 0 else 1.0
    xq = np.clip(np.round(x / xs), -127, 127).astype(np.int8)
    return xq, float(xs)


def int32_reference(W_ternary: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """权威整数参考：acc[i] = Σ_j W[i,j]·xq[j]（int32）。不变量 ① 的基准。"""
    W = np.asarray(W_ternary, dtype=np.int32)
    X = np.asarray(xq, dtype=np.int32)
    return (W @ X).astype(np.int32)


def gemm_fp32(W_fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """fp32 基线。W_fp [out, in] float；x [in] 或 [tokens, in]。"""
    return (np.asarray(W_fp, dtype=np.float32) @ np.asarray(x, dtype=np.float32).T).T


def gemm_int8(W_ternary, x_fp, w_scale, x_scale) -> np.ndarray:
    """int8 近似：W 反量化为 fp32 后乘激活（基线对照）。"""
    W = np.asarray(W_ternary, dtype=np.float32) * np.asarray(w_scale, dtype=np.float32).reshape(
        -1, 1
    )
    return (W @ np.asarray(x_fp, dtype=np.float32).T).T * x_scale


def gemm_ternary_tl(W_ternary, xq, w_scale, x_scale) -> np.ndarray:
    """TL 路径（数值等价 int32 参考）：acc = W_ternary @ xq，再乘缩放。"""
    acc = int32_reference(W_ternary, xq).astype(np.float64)
    return acc * np.asarray(w_scale, dtype=np.float64).reshape(-1) * x_scale


def gemm_ternary_i2s(W_ternary, xq, w_scale, x_scale, row_wsum) -> np.ndarray:
    """I2_S 路径：xu8 = xq+128；acc = xu8·Wi8 - 128·row_wsum（int32）。"""
    xq = np.asarray(xq, dtype=np.int32)
    Wi8 = np.asarray(W_ternary, dtype=np.int32)
    xu8 = xq + 128
    acc = xu8 @ Wi8.T - 128 * np.asarray(row_wsum, dtype=np.int32)
    return acc.astype(np.float64) * np.asarray(w_scale, dtype=np.float64).reshape(-1) * x_scale


def tl_i2s_bitwise_equal(W_ternary, xq, row_wsum) -> bool:
    """不变量 ① 断言：TL 与 I2_S 在 int32 域逐位相等。"""
    tl = int32_reference(W_ternary, xq)
    i2s = (xq.astype(np.int32) + 128) @ np.asarray(W_ternary, dtype=np.int32).T - 128 * np.asarray(
        row_wsum, dtype=np.int32
    )
    return bool(np.array_equal(tl, i2s.astype(np.int32)))
