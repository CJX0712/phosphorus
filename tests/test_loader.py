"""不变量 ③ + 加载正确性：.phos 导出 → 加载 字节级往返，且 scale / HGF 与导出一致。

* 三值权重解包后与 export_numpy 的 ternary 逐位相等。
* per-channel scale 与导出逐元素相等（修复前的 key 错位 bug 回归护栏）。
* HGF 低秩因子 A / B / gate 与导出一致（保证引擎完整复现校正）。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from phosphorus.config import ModelConfig, QuantConfig  # noqa: E402
from phosphorus.export import export_model  # noqa: E402
from phosphorus.loader import load_phos  # noqa: E402
from phosphorus.model import LlamaModel  # noqa: E402


@pytest.fixture
def model_and_weights():
    mcfg = ModelConfig(
        d_model=64, n_layer=3, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=4, qat=True)
    torch.manual_seed(0)
    model = LlamaModel(mcfg, qcfg).eval()
    with torch.no_grad():
        weights = model.export_numpy()
    return mcfg, qcfg, weights


def test_ternary_roundtrip_bitwise(model_and_weights):
    mcfg, qcfg, weights = model_and_weights
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_loader.phos")
    pf = load_phos(blob)
    for k, v in weights.items():
        if k.endswith(".weight") and "layers." in k:
            assert np.array_equal(pf.weights[k], v), f"三值权重往返失配: {k}"


def test_scale_matches_export(model_and_weights):
    mcfg, qcfg, weights = model_and_weights
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_loader.phos")
    pf = load_phos(blob)
    for k, v in weights.items():
        if k.endswith(".scale"):
            base = k[: -len(".scale")] + ".weight"
            assert base in pf.scales, f"scale 键缺失: {base}"
            assert np.allclose(pf.scales[base], v), f"scale 失配: {base}"


def test_hgf_matches_export(model_and_weights):
    mcfg, qcfg, weights = model_and_weights
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_loader.phos")
    pf = load_phos(blob)
    # HGF 因子以 float16 落盘，故用宽松容差（~1e-2 相对）比对。
    for k, _v in weights.items():
        if k.endswith(".hgf_a"):
            base = k[: -len(".hgf_a")] + ".weight"
            assert base in pf.hgf, f"HGF 缺失: {base}"
            A, B, gate = pf.hgf[base]
            assert np.allclose(A, weights[k], atol=1e-2, rtol=1e-2), f"HGF A 失配: {base}"
            assert np.allclose(B, weights[k[: -len(".hgf_a")] + ".hgf_b"], atol=1e-2, rtol=1e-2), (
                f"HGF B 失配: {base}"
            )
            assert abs(gate - weights[k[: -len(".hgf_a")] + ".hgf_gate"]) < 1e-2, (
                f"HGF gate 失配: {base}"
            )


def test_header_fields(model_and_weights):
    mcfg, qcfg, weights = model_and_weights
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_loader.phos")
    pf = load_phos(blob)
    h = pf.header
    assert h["n_layer"] == mcfg.n_layer
    assert h["d_model"] == mcfg.d_model
    assert h["n_head"] == mcfg.n_head
    assert h["vocab"] == mcfg.vocab_size
    assert bool(h["flags"] & (1 << 0))  # F_HAS_HGF
    assert bool(h["flags"] & (1 << 1))  # F_TIED_EMBED
