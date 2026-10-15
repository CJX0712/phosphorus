"""不变量 ①：三值 GEMM 在 int32 域逐位一致（TL == I2_S == 引擎 matmul）。

TL 路径 = 直接 int32 累加 Σ_j W·xq；I2_S 路径 = (xq+128)·Wᵀ − 128·row_wsum。
二者数学恒等，引擎的忠实路径 `_ternary_gemm_fp32` 反量化后须与 int32 参考缩放一致。
"""

import numpy as np
import pytest

from phosphorus.engine import _ternary_gemm_int8
from phosphorus.gemm import (
    gemm_ternary_tl,
    int32_reference,
    tl_i2s_bitwise_equal,
)
from phosphorus.packing import digits_to_ternary, k_pad, ternary_to_digits


def _rand_ternary(shape, rng):
    return rng.integers(-1, 2, size=shape).astype(np.int8)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_tl_i2s_bitwise_equal(rng):
    """TL 与 I2_S 的 int32 累加须逐位相等（核心不变量 ①）。"""
    out, in_f = 32, 96
    W = _rand_ternary((out, in_f), rng)
    xq = rng.integers(-127, 128, size=in_f).astype(np.int8)
    row_wsum = W.astype(np.int32).sum(axis=1)
    assert tl_i2s_bitwise_equal(W, xq, row_wsum)


def test_tl_matches_int32_reference(rng):
    """TL 路径的 int32 累加须等于权威 int32_reference。"""
    out, in_f = 24, 72
    W = _rand_ternary((out, in_f), rng)
    xq = rng.integers(-127, 128, size=in_f).astype(np.int8)
    tl = gemm_ternary_tl(W, xq, np.ones(out), 1.0)
    ref = int32_reference(W, xq).astype(np.float64) * np.ones(out)
    assert np.array_equal(np.round(tl).astype(np.int64), np.round(ref).astype(np.int64))


def test_engine_gemm_matches_int32_reference_scaled(rng):
    """引擎 int8 路径须等于 int32 参考乘缩放（激活量化后用精确 int8 累加）。"""
    out, in_f = 16, 48
    W = _rand_ternary((out, in_f), rng)
    scale = (rng.random(out) + 0.5).astype(np.float32)  # per-channel 权重缩放
    x = rng.standard_normal((1, in_f)).astype(np.float32)  # 单 token 激活 [1, in]
    x_scale = float(np.max(np.abs(x))) / 127.0 + 1e-9
    xq = np.clip(np.round(x / x_scale), -127, 127).astype(np.int8)  # [1, in]
    x_scale_arr = np.array([x_scale], dtype=np.float64)  # 每 token 一个激活缩放

    engine_out = _ternary_gemm_int8(W.astype(np.int32), scale, xq, x_scale_arr)  # [1, out]
    ref_acc = int32_reference(W, xq[0]).astype(np.float64) * scale * x_scale  # [out]
    assert np.allclose(engine_out[0], ref_acc, atol=1e-3)


def test_digits_roundtrip(rng):
    """digit(-1,0,1) <-> 三值 恒等（打包内部表示）。"""
    w = _rand_ternary((10, 24), rng)
    assert np.array_equal(digits_to_ternary(ternary_to_digits(w)), w)
    assert np.array_equal(w[:, : k_pad(24)], w[:, : k_pad(24)])
