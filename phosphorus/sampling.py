"""采样器（temperature / top-k / top-p）。

借鉴来源：
    * Categorical 采样与 top-p（nucleus）来自 Holtzman et al., 2020,
      "The Curious Case of Neural Text Degeneration" — A-19
    * 投机解码的「验证分布」采样遵循 Leviathan et al., 2023 的残差采样
      （见 speculate.py）— A-14
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """从单个位置 logits 采样一个 token id。"""
    if rng is None:
        rng = np.random.default_rng(0)
    logits = np.asarray(logits, dtype=np.float32).astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    # top-k
    if top_k and top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    # top-p
    if top_p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = softmax(logits[order])
        cum = np.cumsum(probs)
        keep = cum <= top_p
        keep[0] = True
        mask = np.zeros_like(logits, dtype=bool)
        mask[order[keep]] = True
        logits = np.where(mask, logits, -np.inf)
    probs = softmax(logits)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))
