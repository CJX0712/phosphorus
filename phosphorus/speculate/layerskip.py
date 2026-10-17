"""O1 自草稿（LayerSkip 风格）：复用同一份三值权重做 early-exit 草稿。

核心超越点（ALGORITHMS.md A-15）：草稿器**不引入任何新权重**（extra_bytes == 0），
仅用目标网络的前 `draft_layers` 层输出作为提议分布 q。这是与 EAGLE/Medusa（需训练
独立草稿头）最根本的区别——零额外参数、零额外训练即可获得草稿分布。

接受率 α 取决于「浅层网络预测」与「完整网络预测」的一致程度；模型越深、早期层与
最终层语义越接近，α 越高（TASKS.md C2 要求自训练语料上 α ≥ 0.5）。
"""

from __future__ import annotations

import numpy as np

from .base import Drafter, DrafterSpec


class LayerSkipDrafter(Drafter):
    """early-exit 自草稿器：用目标模型前 `draft_layers` 层 logits 作提议分布。"""

    def __init__(self, model, draft_layers: int | None = None, name: str = "layerskip"):
        n_layer = int(getattr(model, "n_layer", 1))
        if draft_layers is None:
            draft_layers = max(1, n_layer - 1)  # 默认跳过最后 1 层
        if not (1 <= draft_layers < n_layer):
            raise ValueError(
                f"draft_layers={draft_layers} 必须 ∈ [1, {n_layer}); "
                f"单/双层模型无法做有意义的 early-exit 自草稿"
            )
        self.model = model
        self.draft_layers = draft_layers
        self.spec = DrafterSpec(
            name=name,
            extra_bytes=0,  # ★ O1 硬约束：不新增权重字节
            draft_layers=draft_layers,
        )

    def draft_logits(self, seq: list[int]) -> np.ndarray:
        # 调用方负责在每次草稿/验证前向前 reset KV（无状态推测），这里只取末位 logits。
        out = self.model.forward(np.asarray(seq, dtype=np.int64), stop_at_layer=self.draft_layers)
        return np.asarray(out[-1], dtype=np.float64)
