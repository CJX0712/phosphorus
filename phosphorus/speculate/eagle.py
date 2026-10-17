"""EAGLE / EAGLE-2 草稿器（参考实现）。

文献：EAGLE (Han et al. 2024, arXiv:2401.15077) / EAGLE-2 (arXiv:2406.16858)。
思路：训练一个轻量「草稿头」基于目标模型倒数第二层的隐藏状态自回归预测下一个
token 与下一隐藏状态，形成树状草稿。

本项目的现实约束：当前未训练独立草稿头（O1 的超越点正是「零额外权重」）。因此
EagleDrafter 在此直接复用 LayerSkipDrafter 的 early-exit 自草稿作为等价实现——同样
不引入新权重、同样产出草稿分布 q。若后续训练了真正 EAGLE 草稿头，只需替换
`draft_logits` 即可接入同一套 verify / controller。
"""

from __future__ import annotations

from .layerskip import LayerSkipDrafter


class EagleDrafter(LayerSkipDrafter):
    """EAGLE 风格草稿器（当前 = early-exit 自草稿的别名 / 占位）。"""

    def __init__(self, model, draft_layers: int | None = None):
        super().__init__(model, draft_layers=draft_layers, name="eagle")
