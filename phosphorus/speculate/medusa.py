"""Medusa 多头部并行草稿器（参考实现）。

文献：Medusa (Cai et al. 2024, arXiv:2401.10774)。思路：在基座模型上挂载 K 个「Medusa
头」，每个头在给定同一上下文时并行预测「未来第 i 个 token」，一次性产出多条候选。

本项目的现实约束：未训练 Medusa 头。此处 MedusaDrafter 以 LayerSkipDrafter 的
early-exit 分布为多头部提供「同一上下文、不同前瞻深度」的候选近似（head i 用前
`draft_layers` 层预测，等价于浅层自草稿）。真实多头部需训练独立投影，接口已预留。
"""

from __future__ import annotations

from .layerskip import LayerSkipDrafter


class MedusaDrafter(LayerSkipDrafter):
    """Medusa 风格多头部草稿器（当前 = 浅层自草稿占位）。"""

    def __init__(self, model, draft_layers: int | None = None, heads: int = 1):
        super().__init__(model, draft_layers=draft_layers, name="medusa")
        self.heads = heads
