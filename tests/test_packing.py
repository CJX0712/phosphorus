"""不变量 ③：三值 5-bit 打包 / 解包往返逐位恒等（与 .phos 字节布局严格一致）。"""

import numpy as np
import pytest

from phosphorus.packing import k_pad, pack_ternary_v1, unpack_ternary_v1


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
    """打包字节数 = N * (K_pad/24) * 5。"""
    N, K = 16, 100
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
    packed, Kpad = pack_ternary_v1(W)
    assert Kpad == k_pad(K)
    assert len(packed) == N * (Kpad // 24) * 5
