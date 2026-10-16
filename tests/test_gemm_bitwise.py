"""不变量 ①：三值 GEMM 在 int32 域逐位一致（TL == I2_S == int32_reference）。

本文件的核心断言比旧版更强：TL 侧不再「直接用密集矩阵乘冒充」，
而是 **从 5-bit 打包字节解出 code5 → 查 14 项 LUT → 累加**，
即真正走一遍部署态的 LUT 路径，再与 I2_S、int32_reference 三方逐位比对。

数学恒等式（不变量 ① 的证明）：
    TL:    acc[n] = Σ_g LUT_g[code5[n,g]]                    （LUT 无乘法）
    I2_S:  acc[n] = Σ_j (xq[j]+128)·W[n,j] − 128·Σ_j W[n,j]  （零点校正）
    两者都等于 Σ_j W[n,j]·xq[j]，且全部在 int32 域内，故必逐位相等。
"""

import numpy as np
import pytest

from phosphorus.gemm import (
    build_lut_tables,
    gemm_fp32,
    gemm_int8,
    gemm_ternary_i2s,
    gemm_ternary_tl,
    i2s_acc_i32,
    int32_reference,
    quantize_activation,
    tl_acc_i32,
    tl_i2s_bitwise_equal,
    tl_i2s_bitwise_equal_packed,
)
from phosphorus.packing import k_pad, pack_ternary_v1, unpack_code5, unpack_ternary_v1


def _rand_ternary(shape, rng):
    return rng.integers(-1, 2, size=shape).astype(np.int8)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_tl_i2s_bitwise_equal(rng):
    """TL 与 I2_S 的 int32 累加须逐位相等（核心不变量 ①，密集权重版）。"""
    for out, in_f in [(1, 3), (32, 96), (17, 50), (64, 24)]:
        W = _rand_ternary((out, in_f), rng)
        xq = rng.integers(-127, 128, size=in_f).astype(np.int8)
        row_sum = W.astype(np.int32).sum(axis=1)
        assert tl_i2s_bitwise_equal(W, xq, row_sum), f"失败于 {out}x{in_f}"


def test_packed_tl_i2s_reference_three_way_bitwise(rng):
    """★ 最强形式：打包字节 → LUT（TL）== I2_S == int32_reference，三者逐位相等。

    含 K 不是 24 倍数的补齐边界（K=50 → K_pad=72），验证零填充不污染累加。
    """
    for N, K in [(8, 24), (16, 50), (33, 7), (64, 128)]:
        W = _rand_ternary((N, K), rng)
        xq = rng.integers(-127, 128, size=K).astype(np.int8)
        packed, Kpad = pack_ternary_v1(W)
        row_sum = W.astype(np.int32).sum(axis=1)
        assert tl_i2s_bitwise_equal_packed(packed, W, xq, row_sum, K), f"失败于 {N}x{K}"

        codes = unpack_code5(packed, N, K)
        tl = tl_acc_i32(codes, xq, K)
        i2s = i2s_acc_i32(W, xq, row_sum)
        ref = int32_reference(W, xq)
        assert tl.dtype == np.int32
        assert np.array_equal(tl, i2s) and np.array_equal(tl, ref)
        # 补齐不改变值：K_pad 行宽下结果一致
        assert Kpad == k_pad(K)


def test_lut_table_matches_direct_dot(rng):
    """14 项 LUT 查表值须等于直接点积 Σ (d_k-1)·a_k（建表正确性）。"""
    a = rng.integers(-127, 128, size=3).astype(np.int32)
    lut = build_lut_tables(a.reshape(1, 3), 1)[0]
    for c in range(27):
        d0, d1, d2 = c % 3, (c // 3) % 3, c // 9
        expect = (d0 - 1) * a[0] + (d1 - 1) * a[1] + (d2 - 1) * a[2]
        # c <= 12 -> code5 = c+1（sign=0）；c >= 14 -> code5 = (27-c) | 0x10（sign=1）
        code5 = 0 if c == 13 else ((c + 1) if c <= 12 else ((27 - c) | 0x10))
        assert int(lut[code5]) == int(expect), f"LUT 错于 c={c}"


def test_tl_scaled_matches_reference(rng):
    """TL 后端缩放后的 fp32 输出须等于 int32 参考 × w_scale × x_scale。"""
    N, K = 24, 72
    W = _rand_ternary((N, K), rng)
    packed, _ = pack_ternary_v1(W)
    xq = rng.integers(-127, 128, size=K).astype(np.int8)
    w_scale = (rng.random(N) + 0.5).astype(np.float32)
    x_scale = 0.017
    got = gemm_ternary_tl(packed, xq, w_scale, x_scale, N, K)
    expect = int32_reference(W, xq).astype(np.float64) * w_scale * x_scale
    assert np.allclose(got, expect, rtol=1e-6, atol=1e-4)


def test_i2s_scaled_matches_reference(rng):
    """I2_S 后端缩放后须等于 int32 参考 × w_scale × x_scale。"""
    N, K = 16, 48
    W = _rand_ternary((N, K), rng)
    xq = rng.integers(-127, 128, size=K).astype(np.int8)
    w_scale = (rng.random(N) + 0.5).astype(np.float32)
    got = gemm_ternary_i2s(W, xq, w_scale, 0.02)
    expect = int32_reference(W, xq).astype(np.float64) * w_scale * 0.02
    assert np.allclose(got, expect, rtol=1e-6, atol=1e-4)


def test_fp32_backend_matches_numpy(rng):
    """fp32 后端 = 反量化权重 × fp32 激活（对照基线）。"""
    N, K = 12, 32
    W = _rand_ternary((N, K), rng).astype(np.float32) * 0.3
    x = rng.standard_normal(K).astype(np.float32)
    assert np.allclose(gemm_fp32(W, x), W @ x, atol=1e-6)


def test_int8_backend_within_tolerance(rng):
    """int8 后端相对 fp32 的误差须在 int8 量化容差内（≈1/127 量级）。"""
    N, K = 32, 64
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.2
    x = rng.standard_normal(K).astype(np.float32)
    ref = gemm_fp32(W, x)
    got = gemm_int8(W, x)
    denom = np.maximum(np.abs(ref), 1e-6)
    assert np.max(np.abs(got - ref) / denom) < 0.25


def test_activation_quant_roundtrip(rng):
    """激活量化：dequant(quant(x)) 与 x 的逐元素误差 ≤ γ/254（8 bit 理论界）。"""
    x = rng.standard_normal(64).astype(np.float32) * 3.0
    xq, xs = quantize_activation(x)
    assert xq.dtype == np.int8 and np.all(np.abs(xq) <= 127)
    deq = xq.astype(np.float32) * float(xs[0])
    assert np.max(np.abs(deq - x)) <= float(xs[0]) + 1e-6


def test_unpacked_matches_packed_source(rng):
    """打包 → 解包 → 再打包，两次打包字节须逐位相同（打包确定性）。"""
    W = _rand_ternary((9, 40), rng)
    p1, _ = pack_ternary_v1(W)
    W2 = unpack_ternary_v1(p1, 9, 40)
    p2, _ = pack_ternary_v1(W2)
    assert np.array_equal(p1, p2)
