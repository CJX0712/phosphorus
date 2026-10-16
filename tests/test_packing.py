"""不变量 ③：三值 5-bit 打包 / 解包往返**逐位恒等**，且吞吐达标。

对应 T3/C1 验收：
    * 对 ≥ 1e6 个随机三值权重（含长度 1..300 的**非 24 倍数**边界）往返逐位恒等；
    * `packed_bytes == n_rows · ceil(K/24) · 5`；
    * 纯 NumPy 向量化，打包吞吐 ≥ 50 MB/s（禁止 Python 逐元素循环）。
"""

import time

import numpy as np
import pytest

from phosphorus.packing import (
    BYTES_PER_CHUNK,
    WEIGHTS_PER_CHUNK,
    k_pad,
    pack_ternary_v1,
    packed_bytes,
    unpack_ternary_v1,
)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_pack_unpack_bitwise(rng):
    """随机三值矩阵打包后再解包，须逐位一致（含 K 非 24 倍数的补齐边界）。"""
    for N, K in [(1, 1), (3, 7), (8, 24), (12, 50), (32, 128)]:
        W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
        packed, Kpad = pack_ternary_v1(W)
        W2 = unpack_ternary_v1(packed, N, K)
        assert W2.shape == (N, K), f"shape mismatch {N}x{K}"
        assert np.array_equal(W2, W), f"bitwise mismatch at {N}x{K}"


def test_packed_size(rng):
    """打包字节数 = N · (K_pad/24) · 5。"""
    N, K = 16, 100
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
    packed, Kpad = pack_ternary_v1(W)
    assert Kpad == k_pad(K)
    assert len(packed) == N * (Kpad // WEIGHTS_PER_CHUNK) * BYTES_PER_CHUNK
    assert len(packed) == packed_bytes(N, K)


def test_all_weight_combinations():
    """穷举 3^3 = 27 种三权重组合，验证 code5 编/解码表无遗漏。"""
    combos = np.array(
        [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
        dtype=np.int8,
    )
    W = np.repeat(combos, 4, axis=0)  # 108 行 × 3 列
    packed, _ = pack_ternary_v1(W)
    assert np.array_equal(unpack_ternary_v1(packed, W.shape[0], 3), W)


def test_roundtrip_1e6_weights_bitwise(rng):
    """★ C1 硬指标：≥ 1e6 个随机三值权重、行宽 1..300（含非 24 倍数）逐位恒等。"""
    total = 0
    for K in list(range(1, 25)) + [36, 49, 50, 73, 100, 129, 192, 257, 300]:
        N = max(1, 40_000 // max(K, 1))
        W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
        packed, _ = pack_ternary_v1(W)
        assert len(packed) == packed_bytes(N, K)
        assert np.array_equal(unpack_ternary_v1(packed, N, K), W), f"K={K} 往返不一致"
        total += W.size
    assert total >= 1_000_000, f"样本量不足：{total}"


def test_packing_throughput_above_50mbps(rng):
    """★ C1 硬指标：打包吞吐 ≥ 50 MB/s（按原始 int8 权重字节数计）。"""
    W = rng.integers(-1, 2, size=(1024, 1024)).astype(np.int8)
    t0 = time.perf_counter()
    packed, _ = pack_ternary_v1(W)
    dt = time.perf_counter() - t0
    mbps = W.nbytes / dt / 1e6
    assert mbps >= 50.0, f"打包吞吐 {mbps:.1f} MB/s < 50 MB/s"
    t0 = time.perf_counter()
    W2 = unpack_ternary_v1(packed, 1024, 1024)
    dt2 = time.perf_counter() - t0
    assert np.array_equal(W, W2)
    assert W.nbytes / dt2 / 1e6 >= 20.0, f"解包吞吐 {W.nbytes / dt2 / 1e6:.1f} MB/s 过低"


def test_padding_is_zero_and_lossless(rng):
    """补齐区填 0（digit=1 / w=0），不污染解包后的有效列。"""
    K = 50
    W = rng.integers(-1, 2, size=(4, K)).astype(np.int8)
    W[:, -1] = 1  # 末列放 +1，确保补齐不是简单复制
    packed, Kpad = pack_ternary_v1(W)
    assert Kpad == 72
    assert np.array_equal(unpack_ternary_v1(packed, 4, K), W)
    # 解到补齐宽度时，多出来的 22 列必须全 0
    full = unpack_ternary_v1(packed, 4, Kpad)
    assert np.all(full[:, K:] == 0)
