"""T5 · C4 O2 自适应投机控制器测试。

覆盖：状态机全转换、batch 门控、α 门控、抖动抑制、K* 决策公式正确性。
（wall-clock 加速比 C3 由 KV-cache 热路径保证；此处校验控制器决策逻辑与公式。）
"""

from __future__ import annotations

import numpy as np

from phosphorus.speculate.controller import SpecController, SpecPlan


def _run(ctrl: SpecController, alphas, cs, batch=1, steps=None):
    steps = steps or len(alphas)
    states = []
    for i in range(steps):
        ctrl.observe(alphas[i], cs[i] if cs is not None else 1.0, batch)
        states.append(ctrl.state)
    return states


def test_batch_gate_disables():
    ctrl = SpecController()
    # 高 α、低成本比、batch 小 → 应能开启
    for _ in range(10):
        ctrl.observe(0.8, 0.3, 1)
    assert ctrl.plan().enabled is True
    # batch 超过 B_max → 强制关闭
    ctrl.observe(0.8, 0.3, 16)
    assert ctrl.plan().enabled is False
    assert ctrl.plan().K == 0


def test_low_alpha_shuts_off():
    ctrl = SpecController()
    for _ in range(10):
        ctrl.observe(0.9, 0.2, 1)
    assert ctrl.state in ("ON", "WARMUP")
    # α 骤降 → 最终 OFF
    for _ in range(12):
        ctrl.observe(0.05, 0.2, 1)
    assert ctrl.plan().enabled is False


def test_state_machine_transitions():
    ctrl = SpecController(N_warm=2, N_dwell=2, eps_on=0.1, eps_off=0.05)
    states = _run(ctrl, [0.9] * 20, [0.2] * 20, batch=1, steps=20)
    assert "ON" in states, f"应进入 ON，实际：{states}"
    assert states[0] == "OFF" or states[0] == "WARMUP"


def test_no_chatter_under_random():
    rng = np.random.default_rng(0)
    ctrl = SpecController(N_warm=3, N_dwell=4)
    alphas = rng.uniform(0.5, 0.95, 100)
    cs = rng.uniform(0.1, 0.4, 100)
    states = _run(ctrl, alphas, cs, batch=1, steps=100)
    flips = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    assert flips <= 3, f"100 轮随机输入状态翻转 {flips} 次（应 ≤ 3），{states}"


def test_kstar_maximizes_S():
    ctrl = SpecController(K_max=8, alpha_min=0.0)
    ctrl.observe(0.7, 0.3, 1)
    plan = ctrl.plan()
    assert isinstance(plan, SpecPlan)
    # 选出的 K* 应使 S(K*) 近似最大：与暴力枚举比对
    best_K, best_S = 0, -1.0
    for K in range(0, 9):
        a = ctrl.alpha_hat
        c = ctrl.c
        S = (
            ((1 - a ** (K + 1)) / (1 - a)) / (1 + c * K)
            if 0 < a < 1
            else (K + 1 if a >= 1 else 1 / (1 + c * K))
        )
        if S > best_S:
            best_S, best_K = S, K
    assert plan.K == best_K, f"K*={plan.K} 不等于暴力最优 {best_K}"


def test_predicted_speedup_formula():
    ctrl = SpecController(K_max=8, alpha_min=0.0)
    ctrl.observe(0.7, 0.3, 1)
    plan = ctrl.plan()
    if plan.K > 0:
        expected = (1.0 + ctrl.alpha_hat * plan.K) / (1.0 + ctrl.c * plan.K)
        assert abs(plan.speedup - expected) < 1e-9
