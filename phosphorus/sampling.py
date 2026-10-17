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


def sampling_probs(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """把 logits 变换为**采样器实际使用的**概率分布（与 `sample()` 同分布）。

    投机解码的等价性（不变量 ②）要求：验证用的目标分布 p 必须是「目标采样器
    真正采样时所依据的分布」——即 temperature + top-k + top-p **截断之后**的分布，
    而非原始 softmax。若用原始 softmax 验证，则在 top_p<1 时输出分布与串行采样
    不等价（退化分布下二者碰巧一致，掩盖了缺陷）。故本函数与 `sample()` 共享
    同一变换，供 `speculate` 构造 q/p 分布使用。
    """
    logits = np.asarray(logits, dtype=np.float32).astype(np.float64)
    if temperature <= 0:
        p = np.zeros_like(logits)
        p[int(np.argmax(logits))] = 1.0
        return p
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
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
    return probs / probs.sum()


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
