"""O2 自适应投机控制器（ARCHITECTURE.md §6）。

职责：每轮投机后在线拟合接受率 α 与成本比 c，决定「是否开启投机」与「草稿长度 K*」，
并对 batch / α 做门控与双阈值滞回，规避「盲目开启反模式」（CPU 上 batch 大时投机必亏）。

决策公式（精确模型）：
    S(K) = ((1 - α^(K+1)) / (1 - α)) / (1 + c·K)      (α→1 取 K+1)
简化汇报式：speedup ≈ (1 + α·K) / (1 + c·K)            （ε 吸收到 c 里）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecPlan:
    enabled: bool
    K: int
    state: str
    alpha_hat: float
    speedup: float  # 控制器预测的加速比（(1+αK)/(1+cK)）


class SpecController:
    def __init__(
        self,
        K_max: int = 8,
        W_max: int = 4,
        B_max: int = 8,
        alpha_min: float = 0.15,
        eps_on: float = 0.15,
        eps_off: float = 0.05,
        N_warm: int = 4,
        N_dwell: int = 8,
        budget_factor: float = 1.0,
        smooth: float = 0.8,
    ):
        self.K_max = K_max
        self.W_max = W_max
        self.B_max = B_max
        self.alpha_min = alpha_min
        self.eps_on = eps_on
        self.eps_off = eps_off
        self.N_warm = N_warm
        self.N_dwell = N_dwell
        self.budget_factor = budget_factor
        self.smooth = smooth

        self.alpha_hat: float | None = None
        self.c: float = 1.0
        self.batch_size: int = 0
        self.kv_util: float | None = None
        self.state = "OFF"
        self._warm = 0
        self._dwell = 0
        self.K_star = 0

    # ---- 在线拟合 ---------------------------------------------------------- #
    def observe(
        self, alpha: float, c: float | None, batch_size: int, kv_util: float | None = None
    ) -> None:
        a = max(0.0, min(1.0, float(alpha)))
        if self.alpha_hat is None:
            self.alpha_hat = a
        else:
            self.alpha_hat = self.smooth * self.alpha_hat + (1 - self.smooth) * a
        if c is not None:
            c = max(0.0, float(c))
            self.c = self.smooth * self.c + (1 - self.smooth) * c
        self.batch_size = int(batch_size)
        self.kv_util = kv_util
        self._transition()

    # ---- 决策 -------------------------------------------------------------- #
    def _S(self, K: int, alpha: float, c: float) -> float:
        if alpha >= 1.0:
            return float(K + 1)
        if alpha <= 0.0:
            return 1.0 / (1.0 + c * K)
        num = (1.0 - alpha ** (K + 1)) / (1.0 - alpha)
        return num / (1.0 + c * K)

    def _best_K(self) -> tuple[int, float]:
        alpha = self.alpha_hat if self.alpha_hat is not None else 0.0
        best_K, best_S = 0, -1.0
        for K in range(0, self.K_max + 1):
            s = self._S(K, alpha, self.c)
            if s > best_S:
                best_S, best_K = s, K
        self.K_star = best_K
        return best_K, best_S

    def _set(self, s: str) -> None:
        if s != self.state:
            self.state = s
            self._warm = 0
            self._dwell = 0

    def _transition(self) -> None:
        alpha = self.alpha_hat if self.alpha_hat is not None else 0.0
        _, S = self._best_K()
        # 硬门控：batch 过大或 α 过低 → 直接 OFF（CPU 反模式规避）
        if self.batch_size > self.B_max or alpha < self.alpha_min:
            self._set("OFF")
            return
        if self.state in ("OFF", "WARMUP"):
            if 1.0 + self.eps_on <= S:
                self._warm += 1
                if self._warm >= self.N_warm:
                    self._set("ON")
            else:
                self._warm = 0
                self._set("OFF")
        elif self.state == "ON":
            if 1.0 + self.eps_off <= S:
                self._dwell = 0
            else:
                self._dwell += 1
                if self._dwell >= self.N_dwell:
                    self._set("DEGRADED")
        elif self.state == "DEGRADED":
            if 1.0 + self.eps_off <= S:
                self._dwell += 1
                if self._dwell >= self.N_dwell:
                    self._set("ON")
            else:
                self._dwell += 1
                if self._dwell >= self.N_dwell:
                    self._set("OFF")

    # ---- 对外接口 ---------------------------------------------------------- #
    def plan(self) -> SpecPlan:
        enabled = (
            self.state in ("WARMUP", "ON", "DEGRADED")
            and self.alpha_hat is not None
            and self.alpha_hat >= self.alpha_min
            and self.batch_size <= self.B_max
        )
        K = self.K_star if enabled else 0
        alpha = self.alpha_hat if self.alpha_hat is not None else 0.0
        speedup = (1.0 + alpha * K) / (1.0 + self.c * K) if K > 0 else 1.0
        return SpecPlan(enabled=enabled, K=K, state=self.state, alpha_hat=alpha, speedup=speedup)

    def predicted_speedup(self) -> float:
        return self.plan().speedup
