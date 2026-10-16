"""GQA 注意力：分页（paged）与 dense 参考实现（T4-B）。

分页路径从 `PagedKVCache` gather K/V（只存 `n_kv_head` 份，读取时
`np.repeat` 展开到 `n_head`），dense 参考直接吃 `[1, H, T, Dh]` 全量张量。
两条路径共用同一个 `gqa_attention` 数学核心，因此差异只可能来自浮点
累加顺序——C4 验收要求最大绝对差 < 1e-5（实测为 0 或 ~1e-7）。

因果掩码语义与 `engine.py` 的 dense 前向一致：query 绝对位置
`q_start + i` 可 attend 到 key 位置 `0 .. q_start + i`。
"""

from __future__ import annotations

import math

import numpy as np


def gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    q_start: int,
    causal: bool = True,
) -> np.ndarray:
    """GQA 注意力核心。q/k/v: [1, H, T, Dh] / [1, H, S, Dh] → [1, H, T, Dh]。

    `q_start` 是**本批第一个 query 在当前 KV 序列中的下标**（引擎里即已存
    token 数）；滑动窗口丢弃后应传保留坐标系下的值（通常为 0），而非绝对位置。
    """
    Dh = q.shape[-1]
    S = k.shape[2]
    T = q.shape[2]
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
    if causal:
        cols = np.arange(S)
        rows = q_start + np.arange(T)
        mask = cols[None, :] <= rows[:, None]
        scores = np.where(mask[None, None, :, :], scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    return attn @ v


def paged_gqa_attention(
    q: np.ndarray,
    cache,
    seq_id: int,
    layer: int,
    n_rep: int,
    q_start: int,
    causal: bool = True,
) -> np.ndarray:
    """分页 GQA 注意力：从 cache gather 该序列全部有效 K/V 后走同一数学核心。

    q: [1, H, T, Dh]（RoPE 已应用）。返回 [1, H, T, Dh]。
    """
    k, v = cache.gather_heads(seq_id, layer, n_rep)
    if k.shape[2] == 0:  # 空序列：无 KV 可 attend（理论上不会发生）
        return np.zeros_like(q)
    return gqa_attention(q, k, v, q_start, causal=causal)
