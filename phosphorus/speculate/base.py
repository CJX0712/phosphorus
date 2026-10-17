"""投机解码：抽象接口与通用类型。

设计（见 ARCHITECTURE.md §6 / TASKS.md T5）：
    * 草稿器（Drafter）只负责「给出一个 token 的提议分布 q」，不关心验证。
    * 验证（verify.py）严格按 Leviathan et al. 2023 的残差拒绝采样，保证输出
      分布与目标模型 p **精确相等**（不变量 ②）。
    * 自草稿（O1, layerskip.py）复用同一份三值权重做 early-exit，extra_bytes == 0。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DrafterSpec:
    """草稿器元信息（用于报告与断言）。"""

    name: str
    extra_bytes: int  # 相对目标模型新增的权重字节数（O1 必须为 0）
    draft_layers: int  # early-exit 层数（O1）；非 early-exit 草稿器可忽略


class Drafter(ABC):
    """草稿器抽象：给定已生成序列，返回下一个 token 的提议分布 q（logits 或概率）。"""

    spec: DrafterSpec

    @abstractmethod
    def draft_logits(self, seq: list[int]) -> np.ndarray:
        """对序列 `seq` 做一次前向，返回**末位** token 的 logits（提议分布 q 的来源）。"""
        raise NotImplementedError

    @property
    def extra_bytes(self) -> int:
        return self.spec.extra_bytes

    def reset(self) -> None:  # 默认无状态；有内部缓存的草稿器可重写
        return None
