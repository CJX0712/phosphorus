"""NGramDrafter —— 零成本 n-gram 对照草稿器（C2 对照基线）。

思路：在已生成序列内查找「最后 n-1 元语法」的历史出现，把最常见的后继 token 作为
确定性草稿（one-hot 提议分布 q）。命中时**零模型开销**；未命中时回退到目标模型完整
前向（q ≈ p），保证 Leviathan 验证后的输出分布仍**严格等于**目标 p（不变量 ②）。

与 LayerSkipDrafter 并列报告 α，用于量化「自草稿 vs 检索式草稿」的接受率差异。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import Drafter, DrafterSpec


class NGramDrafter(Drafter):
    def __init__(self, model=None, order: int = 3, name: str = "ngram"):
        # model 仅用于「未命中」回退；纯 n-gram 命中路径完全不调用模型。
        self.model = model
        self.order = max(1, int(order))
        self.draft_layers = 0  # n-gram 不改写网络深度，仅作占位
        # 回退时也复用目标权重，故不新增字节
        self.spec = DrafterSpec(name=name, extra_bytes=0, draft_layers=0)

    def _predict(self, seq: list[int]) -> int | None:
        n = self.order
        if len(seq) < n:
            return None
        ctx = tuple(seq[-n:])  # 最后 n 个 token 构成的上下文
        # 在 seq[:-1] 中统计 ctx 的后继 token 频次
        follow: Counter = Counter()
        for i in range(len(seq) - n):
            if tuple(seq[i : i + n]) == ctx:
                follow[seq[i + n]] += 1
        if not follow:
            return None
        return int(follow.most_common(1)[0][0])

    def draft_logits(self, seq: list[int]) -> np.ndarray:
        V = int(getattr(self.model, "vocab", 0)) or 32000
        y = self._predict(seq)
        if y is not None:
            # one-hot 提议：q[y] = 1，其余 0（贪心草稿特例，Leviathan 仍精确等价）
            logits = np.full(V, -1e9, dtype=np.float64)
            logits[y] = 0.0
            return logits
        # 未命中：回退目标模型完整前向（q ≈ p，验证接受率≈1，分布不变）
        if self.model is not None:
            out = self.model.forward(np.asarray(seq, dtype=np.int64))
            return np.asarray(out[-1], dtype=np.float64)
        # 极端兜底：均匀（仅在既无命中也无模型时，验证会退化为随机——不推荐）
        return np.zeros(V, dtype=np.float64)
