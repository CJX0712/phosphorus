"""不变量 ⑥：三值量化 + 直通估计（STE）正确。

* forward：量化权重须逐元素等于 clamp(round(w/scale), -1, 1) * scale
  （真·三值，与 export_numpy / engine 部署权重一致）。
* backward：STE 梯度须为恒等（全精度权重以单位梯度更新，BitNet QAT 标准），
  否则训练侧前向与推理侧权重不一致（不变量 ⑦ 的隐患源）。
"""

import pytest

torch = pytest.importorskip("torch")
from phosphorus.model import ternary_ste  # noqa: E402


def test_forward_is_true_ternary():
    """forward 值 = w_tern * scale（∈ {-scale,0,scale}），逐元素精确。"""
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.1
    s = torch.abs(w).mean(dim=-1, keepdim=True) + 1e-8
    y = ternary_ste(w, s)
    w_tern = torch.clamp(torch.round(w / s), -1.0, 1.0)
    expected = w_tern * s
    assert torch.allclose(y, expected, atol=1e-7)


def test_ste_gradient_is_identity():
    """STE 反向梯度 = 恒等（1），不是 0（round 不可导 → 必须直通过）。"""
    torch.manual_seed(1)
    w = torch.randn(16, 48, requires_grad=True)  # 叶子张量，梯度才能写回 .grad
    with torch.no_grad():
        w.mul_(0.2)
        # 真实用法中 scale 在 no_grad 下计算（见 BitLinear.forward），故此处也 detached，
        # 否则梯度会经 scale 泄漏，破坏 STE 恒等性。
        s = torch.abs(w).mean(dim=-1, keepdim=True) + 1e-8
    y = ternary_ste(w, s)
    loss = y.sum()
    loss.backward()
    # 梯度应处处为 1（来自 (w - w.detach()) 项的恒等直通；量化项梯度为 0）
    assert w.grad is not None
    assert torch.allclose(w.grad, torch.ones_like(w), atol=1e-6)


def test_values_in_ternary_range():
    """量化权重（除以 scale 后）须落在 [-1, 1] 整数集合。"""
    torch.manual_seed(2)
    w = torch.randn(8, 32) * 5.0  # 大动态范围，压力测试 clamp
    s = torch.abs(w).mean(dim=-1, keepdim=True) + 1e-8
    y = ternary_ste(w, s)
    ratio = y / s
    assert torch.all(ratio >= -1.0 - 1e-6) and torch.all(ratio <= 1.0 + 1e-6)
    # 且 ratio 在 [-1,1] 内应恰为 -1/0/1（量化）
    uniq = torch.unique(ratio.round())
    assert set(uniq.tolist()).issubset({-1.0, 0.0, 1.0})
