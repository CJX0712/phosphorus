"""不变量 ④：RoPE 旋转位置编码的**相对性** —— 内积只依赖相对距离。

来源：Su et al., 2021, RoFormer [arXiv:2104.09864]。
RoPE 的核心性质是 <R_m·q, R_n·k> = <q, R_{n-m}·k>，即注意力分数天然
编码相对位置。若此性质被破坏，长上下文外推与 KV cache 的正确性都会失效。
"""

import math

import pytest

torch = pytest.importorskip("torch")

from phosphorus.model import apply_rope, build_rope  # noqa: E402


def _rope_dot(i: int, j: int, d_head: int = 16, T: int = 12, theta: float = 10000.0):
    """把同一个向量放到位置 i 与 j，返回旋转后的内积。"""
    torch.manual_seed(0)
    v = torch.randn(1, 1, 1, d_head)
    q = v.repeat(1, 1, T, 1)  # [1,1,T,dh]，每个位置都是同一个向量
    k = q.clone()
    cos, sin = build_rope(theta, d_head, T, "cpu")
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    return float((qr[0, 0, i] * kr[0, 0, j]).sum())


def test_dot_depends_only_on_relative_distance():
    """相同相对距离的位置对，内积必须相同（不变量 ④）。"""
    for d in (1, 2, 3, 5, 7):
        base = _rope_dot(0, d)
        for i in (1, 2, 4):
            if i + d < 12:
                assert abs(_rope_dot(i, i + d) - base) < 1e-4, f"距离 {d} 在起点 {i} 处不一致"


def test_different_distances_give_different_scores():
    """反例检查：不同相对距离应给出不同内积（否则说明旋转没生效）。"""
    vals = [_rope_dot(0, d) for d in (0, 1, 2, 3)]
    assert len(set(round(v, 5) for v in vals)) >= 3, f"内积未随距离变化: {vals}"


def test_rope_preserves_norm():
    """旋转是正交变换，范数不变。"""
    torch.manual_seed(1)
    d_head, T = 16, 6
    q = torch.randn(1, 1, T, d_head)
    cos, sin = build_rope(10000.0, d_head, T, "cpu")
    qr = apply_rope(q, cos, sin)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_pairwise_rotation_formula():
    """与闭式公式对照：out[2t] = q1·cos − q2·sin，out[2t+1] = q2·cos + q1·sin。"""
    torch.manual_seed(2)
    d_head, T = 8, 3
    q = torch.randn(1, 1, T, d_head)
    cos, sin = build_rope(10000.0, d_head, T, "cpu")
    out = apply_rope(q, cos, sin)
    for t in range(T):
        for p in range(d_head // 2):
            c, s = float(cos[t, p]), float(sin[t, p])
            q1, q2 = float(q[0, 0, t, 2 * p]), float(q[0, 0, t, 2 * p + 1])
            assert abs(float(out[0, 0, t, 2 * p]) - (q1 * c - q2 * s)) < 1e-5
            assert abs(float(out[0, 0, t, 2 * p + 1]) - (q2 * c + q1 * s)) < 1e-5


def test_engine_rope_matches_torch():
    """engine（numpy）的 RoPE 须与 torch 侧逐元素一致（不变量 ⑦ 的前置）。"""
    import numpy as np

    from phosphorus.engine import _apply_rope, _build_rope

    d_head, T = 16, 5
    torch.manual_seed(3)
    q_t = torch.randn(1, 1, T, d_head)
    cos_t, sin_t = build_rope(10000.0, d_head, T, "cpu")
    ref = apply_rope(q_t, cos_t, sin_t).numpy()

    cos_n, sin_n = _build_rope(10000.0, d_head, T)
    got = _apply_rope(q_t.numpy(), cos_n[:T], sin_n[:T])
    assert np.allclose(got, ref, atol=1e-5)
    assert not math.isnan(float(got.sum()))
