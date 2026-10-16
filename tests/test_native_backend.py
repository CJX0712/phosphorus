"""T4 C2：原生 C99 内核与 numpy 后端的位精确对齐（不变量 ①）。

断言层级（docs/TASKS.md T4 C2）：
    L0  native.i2s_i32 == native.tl_i32 == int32_reference          逐位
    L1  SIMD 路径 == 标量路径（PHOS_FLAG_SCALAR）                    逐位
    L2  f32 生产路径 == numpy float64 参考                           逐位
    L3  rmsnorm / softmax / rope vs numpy                            < 1e-6
    L4  pack5 / unpack5 vs packing.py                                字节级逐位
    L5  行块分块拼接 == 全量                                          逐位
    L6  端到端：PhosModel(impl=native) == PhosModel(impl=numpy)      逐位

原生后端不可用时整套 skip（降级路径由 test_backend.py 覆盖）。
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphorus.gemm import (
    gemm_ternary_i2s,
    gemm_ternary_tl,
    int32_reference,
    row_wsum,
)
from phosphorus.native import (
    PHOS_FLAG_AVX2,
    PHOS_FLAG_SCALAR,
    NativeUnavailable,
    kernels,
)
from phosphorus.packing import k_pad, pack_ternary_v1, unpack_ternary_v1

torch = pytest.importorskip("torch")  # 端到端用例需要构造 .phos；内核用例不需要
from phosphorus.config import ModelConfig, QuantConfig  # noqa: E402
from phosphorus.engine import PhosModel  # noqa: E402
from phosphorus.export import export_model  # noqa: E402
from phosphorus.loader import load_phos  # noqa: E402
from phosphorus.model import LlamaModel  # noqa: E402

try:
    KERN = kernels()
except NativeUnavailable:  # pragma: no cover - 环境无 ziglang 时整套跳过
    KERN = None

pytestmark = pytest.mark.skipif(KERN is None, reason="native 后端不可用（未装 ziglang?）")

AVX2 = PHOS_FLAG_AVX2 if (KERN is not None and KERN.cpu_has_avx2()) else 0
SCALAR = PHOS_FLAG_SCALAR

_SHAPES = [(1, 1), (1, 24), (7, 25), (128, 23), (128, 24), (192, 128), (256, 100), (384, 300)]


def _rand_weights(rng, N, K):
    W = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(N, K))
    W[rng.random((N, K)) < 0.4] = 0  # 0 占比 ~0.4 ∈ [0.2, 0.6]
    return W


def _rand_act(rng, K):
    return rng.integers(-127, 128, size=K).astype(np.int8)


# --------------------------------------------------------------------------- #
# L0 / L1：整数域逐位相等                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N,K", _SHAPES)
def test_i2s_and_tl_bitwise_equal_int32_reference(N, K):
    """L0：native 的 i2s 与 tl 整数累加都与权威参考逐位相等（AVX2 + 标量）。"""
    rng = np.random.default_rng(N * 1000 + K)
    W = _rand_weights(rng, N, K)
    xq = _rand_act(rng, K)
    ref = int32_reference(W, xq)

    xu8 = (xq.astype(np.int32) + 128).astype(np.uint8)
    rw = row_wsum(W)
    for flags in (AVX2, SCALAR):
        i2s = KERN.i2s_gemm_i32(W, xu8, rw, flags=flags)
        assert np.array_equal(i2s, ref), f"i2s != reference (N={N},K={K},flags={flags})"

    packed, Kp = pack_ternary_v1(W)
    xpad = np.zeros(Kp, dtype=np.int8)
    xpad[:K] = xq
    for flags in (AVX2, SCALAR):
        tl = KERN.tl_gemm_i32(packed, xpad, N, Kp, flags=flags)
        assert np.array_equal(tl, ref), f"tl != reference (N={N},K={K},flags={flags})"


@pytest.mark.parametrize("N,K", _SHAPES)
def test_scalar_and_simd_bitwise_equal(N, K):
    """L1：同一内核的 SIMD 与标量路径逐位相等。"""
    rng = np.random.default_rng(909 + N * 31 + K)
    W = _rand_weights(rng, N, K)
    xq = _rand_act(rng, K)
    xu8 = (xq.astype(np.int32) + 128).astype(np.uint8)
    rw = row_wsum(W)
    packed, Kp = pack_ternary_v1(W)
    xpad = np.zeros(Kp, dtype=np.int8)
    xpad[:K] = xq

    assert np.array_equal(
        KERN.i2s_gemm_i32(W, xu8, rw, flags=SCALAR), KERN.i2s_gemm_i32(W, xu8, rw, flags=AVX2)
    )
    assert np.array_equal(
        KERN.tl_gemm_i32(packed, xpad, N, Kp, flags=SCALAR),
        KERN.tl_gemm_i32(packed, xpad, N, Kp, flags=AVX2),
    )


# --------------------------------------------------------------------------- #
# L2：f32 生产路径与 numpy float64 参考逐位相等                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N,K", [(128, 128), (192, 192), (64, 384)])
def test_f32_production_bitwise_equal_numpy(N, K):
    """L2：f32 输出与 gemm.gemm_ternary_* 的 float64 参考逐位相等。"""
    rng = np.random.default_rng(4242 + N + K)
    W = _rand_weights(rng, N, K)
    xq = _rand_act(rng, K)
    w_scale = (rng.random(N).astype(np.float32) + 0.1).astype(np.float32)
    sx = float(rng.random() + 0.1)

    ref_i2s = gemm_ternary_i2s(W, xq, w_scale, sx)
    xu8 = (xq.astype(np.int32) + 128).astype(np.uint8)
    got = KERN.i2s_gemm_f32(W, xu8, row_wsum(W), w_scale, sx)
    assert np.array_equal(got, ref_i2s), "i2s f32 与 numpy 参考不逐位相等"

    packed, Kp = pack_ternary_v1(W)
    xpad = np.zeros(Kp, dtype=np.int8)
    xpad[:K] = xq
    ref_tl = gemm_ternary_tl(packed, xq, w_scale, sx, N, K)
    got_tl = KERN.tl_gemm_f32(packed, xpad, w_scale, sx, N, Kp)
    assert np.array_equal(got_tl, ref_tl), "tl f32 与 numpy 参考不逐位相等"


# --------------------------------------------------------------------------- #
# L4：pack / unpack 与 packing.py 字节级一致                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(20))
def test_pack_unpack_roundtrip_matches_packing(seed):
    """L4：C 版 pack5/unpack5 与 packing.py 往返逐位一致（含非 24 倍数 K）。"""
    rng = np.random.default_rng(seed)
    N = int(rng.integers(1, 40))
    K = int(rng.integers(1, 400))
    W = _rand_weights(rng, N, K)
    Kp = k_pad(K)
    Wpad = np.zeros((N, Kp), dtype=np.int8)
    Wpad[:, :K] = W

    pk_c = KERN.pack5(Wpad.reshape(-1))
    pk_py, Kp2 = pack_ternary_v1(W)
    assert Kp2 == Kp and pk_c.size == pk_py.size
    assert np.array_equal(pk_c.view(np.uint8), pk_py.view(np.uint8)), "pack5 与 packing.py 不一致"

    un_c = KERN.unpack5(pk_c, N * Kp // 3).reshape(N, Kp)
    un_py = unpack_ternary_v1(pk_py, N, K)
    assert np.array_equal(un_c[:, :K], un_py), "unpack5 与 packing.py 不一致"
    assert np.array_equal(un_c[:, :K], W), "往返不恒等"


# --------------------------------------------------------------------------- #
# L5：行块分块拼接 == 全量                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N,K", [(192, 192), (257, 384)])
def test_row_block_split_equals_full(N, K):
    """L5：按行块分块调用的拼接结果与全量一致（并行正确性）。"""
    rng = np.random.default_rng(N + K)
    W = _rand_weights(rng, N, K)
    xq = _rand_act(rng, K)
    xu8 = (xq.astype(np.int32) + 128).astype(np.uint8)
    rw = row_wsum(W)
    full = KERN.i2s_gemm_i32(W, xu8, rw)
    for n0, n1 in [(0, 1), (0, 64), (64, 128), (128, N), (N - 3, N)]:
        part = KERN.i2s_gemm_i32(W, xu8, rw, n0=n0, n1=n1)
        assert np.array_equal(part[n0:n1], full[n0:n1]), f"行块 [{n0},{n1}) 与全量不一致"


# --------------------------------------------------------------------------- #
# L3：ops 与 numpy 参考 < 1e-6                                                  #
# --------------------------------------------------------------------------- #


def test_ops_match_numpy_reference():
    """L3：rmsnorm / softmax / rope 与 numpy 参考误差 < 1e-6。"""
    rng = np.random.default_rng(7)
    N, D = 8, 192
    x = rng.standard_normal((N, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    eps = 1e-5

    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    ref_norm = (x * (1.0 / np.sqrt(var + eps)) * w).astype(np.float32)
    assert np.allclose(KERN.rmsnorm(x, w, eps), ref_norm, atol=1e-6, rtol=0)

    s = rng.standard_normal((4, 64)).astype(np.float32)
    e = np.exp(s - s.max(-1, keepdims=True))
    ref_sm = (e / e.sum(-1, keepdims=True)).astype(np.float32)
    assert np.allclose(KERN.softmax(s), ref_sm, atol=1e-6, rtol=0)

    T, H, Dh = 5, 3, 32
    q = rng.standard_normal((T, H, Dh)).astype(np.float32)
    got = KERN.rope(q, pos0=3, theta=10000.0)
    inv = 1.0 / (10000.0 ** (np.arange(0, Dh, 2, dtype=np.float64) / Dh))
    freqs = np.outer(np.arange(3, 3 + T, dtype=np.float64), inv)
    c = np.cos(freqs).astype(np.float32).reshape(T, 1, -1)
    sn = np.sin(freqs).astype(np.float32).reshape(T, 1, -1)
    q4 = q.reshape(T, H, Dh)
    ref = np.empty_like(q4)
    ref[..., 0::2] = q4[..., 0::2] * c - q4[..., 1::2] * sn
    ref[..., 1::2] = q4[..., 1::2] * c + q4[..., 0::2] * sn
    assert np.allclose(got, ref, atol=1e-6, rtol=0), f"rope 最大误差 {np.abs(got - ref).max()}"


# --------------------------------------------------------------------------- #
# L6：端到端 native == numpy                                                    #
# --------------------------------------------------------------------------- #


def _tiny_phos(tmp_path, name="_t_native.phos"):
    mcfg = ModelConfig(
        d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=4, qat=True)
    model = LlamaModel(mcfg, qcfg).eval()
    with torch.no_grad():
        weights = model.export_numpy()
    path = tmp_path / name
    export_model(weights, mcfg, qcfg, path=str(path))
    return str(path)


@pytest.mark.parametrize("backend", ["i2s", "tl"])
def test_engine_native_matches_numpy_bitwise(tmp_path, backend):
    """L6a：decode 形态（T=1，native 实际被调度）下，两种实现的 logits 逐位相等。"""
    path = _tiny_phos(tmp_path)
    m_np = PhosModel(load_phos(path), backend=backend, native="numpy")
    m_nat = PhosModel(load_phos(path), backend=backend, native="native")
    assert m_nat.impl == "native", "native 实现未启用"
    a = m_np.forward(np.array([3, 17, 42, 8, 99, 5], dtype=np.int64))
    b = m_nat.forward(np.array([3, 17, 42, 8, 99, 5], dtype=np.int64))
    assert np.array_equal(a, b), f"backend={backend}: prefill 阶段 native 与 numpy 不逐位相等"
    # decode：单 token，此时 _use_native 为真，native 内核真正被执行
    a1 = m_np.forward(np.array([77], dtype=np.int64))
    b1 = m_nat.forward(np.array([77], dtype=np.int64))
    assert np.array_equal(a1, b1), f"backend={backend}: decode 阶段 native 与 numpy 不逐位相等"


@pytest.mark.parametrize("backend,T", [("i2s", 1), ("i2s", 8), ("tl", 1), ("tl", 8)])
def test_proj_native_matches_numpy_directly(tmp_path, backend, T):
    """L6b：绕过调度启发式，直接比对 _proj_native 与 numpy 投影函数（逐位）。"""
    from phosphorus.engine import _quant_per_token, _ternary_gemm_i2s, _ternary_gemm_tl

    path = _tiny_phos(tmp_path)
    m = PhosModel(load_phos(path), backend=backend, native="native")
    rng = np.random.default_rng(T)
    x = rng.standard_normal((T, m.d)).astype(np.float32)
    for name in m._proj_names[:2]:
        wname = f"layers.0.{name}.weight"
        W = m.w[wname]
        w_scale = m.scales.get(wname, np.ones(W.shape[0], dtype=np.float32))
        xq, x_scale = _quant_per_token(x)
        got = m._proj_native(x, wname, W.astype(np.int32), w_scale, xq, x_scale)
        if backend == "i2s":
            ref = _ternary_gemm_i2s(W, w_scale, xq, x_scale)
        else:
            ref = _ternary_gemm_tl(m._codes_of(wname), W.shape[1], w_scale, xq, x_scale)
        assert np.array_equal(got, ref), f"{wname} T={T}: _proj_native 与 numpy 不逐位相等"


def test_engine_default_fp32_path_unchanged(tmp_path):
    """默认 fp32 路径不受 native 影响（回归保护）。"""
    path = _tiny_phos(tmp_path)
    m = PhosModel(load_phos(path))
    tokens = np.array([3, 17, 42], dtype=np.int64)
    out = m.forward(tokens)
    assert out.shape == (3, 256) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# 后端注册表                                                                   #
# --------------------------------------------------------------------------- #


def test_backend_resolve_and_available():
    from phosphorus.backend import available_backends, resolve

    assert resolve("numpy") == "numpy"
    assert resolve("fp32") == "numpy"
    got = resolve("native")
    assert got in ("native", "numpy")
    assert ("native" in available_backends()) == (got == "native")
