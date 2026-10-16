"""HGF（Hybrid Gated Flow）低秩校正：可开关、秩可配、B 零初始化。

来源：Hybrid Gated Flow 思路（用户给定）；低秩形式参考 LoRA [arXiv:2106.09685]。
公式：y = ternary_out + gate · (alpha / r) · (B @ (A @ x))
      A ~ N(0, 1/sqrt(d_in))，B 零初始化，gate 可学习标量（初值 0.05）

关键不变量：
    * r = 0 时与「无 HGF」**数值完全一致**（不是近似，是同一条代码路径）；
    * B 零初始化 → 训练/启用起点校正量恒为 0，不破坏已有量化模型；
    * 额外参数量 = r · (d_in + d_out) + 1（gate），用于校验 §3.7 的内存开销公式。
"""

import pytest

torch = pytest.importorskip("torch")

from phosphorus.config import ModelConfig, QuantConfig  # noqa: E402
from phosphorus.model import BitLinear, LlamaModel  # noqa: E402


def test_rank_zero_has_no_hgf_params():
    """r=0 时不创建任何 HGF 参数（参数量与纯三值线性层一致）。"""
    for r in (0, 1, 2, 4, 8, 16):
        blk = BitLinear(32, 48, qat=True, hgf_rank=r)
        n = sum(p.numel() for p in blk.parameters())
        expect = 32 * 48 + (r * (32 + 48) + 1 if r > 0 else 0)
        assert n == expect, f"r={r} 参数量 {n} != {expect}"


def test_b_zero_init_means_identity_at_start():
    """B 零初始化 → 初始校正恒为 0，HGF 层的输出等于纯三值线性层输出。"""
    torch.manual_seed(0)
    in_f, out_f = 16, 24
    plain = BitLinear(in_f, out_f, qat=True, hgf_rank=0)
    with_hgf = BitLinear(in_f, out_f, qat=True, hgf_rank=4, hgf_gate_init=0.05)
    with torch.no_grad():
        with_hgf.weight.copy_(plain.weight)
    x = torch.randn(5, in_f)
    with torch.no_grad():
        y0 = plain(x)
        y1 = with_hgf(x)
    assert torch.allclose(y0, y1, atol=1e-6), "HGF 起点不是恒等映射"


def test_gate_init_value():
    """gate 初值须等于配置值（默认 0.05，小初值避免训练初期破坏三值主干）。"""
    blk = BitLinear(8, 8, qat=True, hgf_rank=4, hgf_gate_init=0.05)
    assert abs(float(blk.gate.item()) - 0.05) < 1e-9
    assert torch.count_nonzero(blk.lora_b) == 0, "B 必须零初始化"


def test_hgf_rank_zero_model_matches_quant_only():
    """模型级：hgf_rank=0 与 hgf_rank>0（但 gate=0）前向一致。"""
    torch.manual_seed(1)
    mcfg = ModelConfig(d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, vocab_size=128)
    m0 = LlamaModel(mcfg, QuantConfig(hgf_rank=0, qat=True)).eval()
    m4 = LlamaModel(mcfg, QuantConfig(hgf_rank=4, qat=True)).eval()
    # 只同步两者共有的参数（tok_emb / norm / subln / 各投影权重），
    # HGF 旁路参数保持初始化值（B=0）→ 校正量恒为 0。
    sd0, sd4 = m0.state_dict(), m4.state_dict()
    with torch.no_grad():
        for k, v in sd0.items():
            if k in sd4:
                sd4[k].copy_(v)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        y0 = m0(ids)
        y4 = m4(ids)
    assert torch.allclose(y0, y4, atol=1e-5), "r=4 但 B=0 时应与 r=0 完全一致"


def test_hgf_extra_bytes_formula():
    """§3.7 闭式开销公式：overhead% = r·(d_in+d_out)·2 / (d_in·d_out·1.667/8)·100。"""
    d_in = d_out = 512
    ternary_bytes = d_in * d_out * 1.667 / 8
    for r, expect_pct in ((1, 3.75), (2, 7.5), (4, 15.0), (8, 30.0), (16, 60.0)):
        extra = r * (d_in + d_out) * 2  # f16 = 2 bytes/param
        pct = extra / ternary_bytes * 100
        assert abs(pct - expect_pct) < 0.5, f"r={r}: {pct:.2f}% != {expect_pct}%"


def test_model_param_count_matches_config():
    """模型实际参数量须与 ModelConfig.n_params() 的闭式估算一致。"""
    mcfg = ModelConfig(d_model=64, n_layer=3, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256)
    model = LlamaModel(mcfg, QuantConfig(hgf_rank=0, qat=True))
    actual = sum(p.numel() for p in model.parameters())
    assert actual == mcfg.n_params(), f"实际 {actual} != 估算 {mcfg.n_params()}"
