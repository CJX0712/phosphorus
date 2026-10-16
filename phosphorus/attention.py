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
    token 数）；滑动窗口/驱逐丢弃后应传保留坐标系下的值（通常为 0），而非绝对位置。
    """
    return attention_weights(q, k, q_start, causal=causal) @ v


def attention_weights(
    q: np.ndarray,
    k: np.ndarray,
    q_start: int,
    causal: bool = True,
) -> np.ndarray:
    """softmax 后的注意力权重（不乘 V）：q/k [1,H,T,Dh] · [1,H,S,Dh] → [1,H,T,S]。

    与 `gqa_attention` 的前半完全同构（同算子、同顺序），故
    `gqa_attention(q,k,v,s,c) == attention_weights(q,k,s,c) @ v` 逐位成立——
    T4-B 已定的「分页 == dense」等价性不被本次重构破坏。S==0 时返回零权重。
    """
    Dh = q.shape[-1]
    S = k.shape[2]
    T = q.shape[2]
    if S == 0:
        return np.zeros((q.shape[0], q.shape[1], T, 0), dtype=q.dtype)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
    if causal:
        cols = np.arange(S)
        rows = q_start + np.arange(T)
        mask = (cols[None, :] <= rows[:, None])[None, None, :, :]
        scores = np.where(mask, scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    return attn / np.sum(attn, axis=-1, keepdims=True)


def kv_importance_scores(
    q_win: np.ndarray, k_past: np.ndarray, causal_within: bool = True
) -> np.ndarray:
    """KV 驱逐用的重要性分数：观测窗口 query 对 past key 的注意力，逐 token 求和。

    复刻 PyramidKV/SnapKV 官方计分（pyramidkv_utils.py）：取**末尾 W 个 query**
    对「窗口之外全部历史 key」的注意力，窗口内施加因果 mask（影响 softmax 归一化，
    不可省略），softmax 后对 query 维求和 → [H, T_past]。H2O 走全矩阵变体
    （见 `paged_gqa_attention` 的 `return_weights`）。

    q_win: [1, H, w, Dh]；k_past: [1, H, T_past, Dh]（已 GQA 展开到 n_head）。
    """
    H = q_win.shape[1]
    w = q_win.shape[2]
    S = k_past.shape[2]
    if S == 0 or w == 0:
        return np.zeros((H, S), dtype=np.float32)
    Dh = q_win.shape[-1]
    s_past = (q_win @ k_past.transpose(0, 1, 3, 2)) / math.sqrt(Dh)  # [1,H,w,S]
    s_win = (q_win @ q_win.transpose(0, 1, 3, 2)) / math.sqrt(Dh)  # [1,H,w,w]
    if causal_within and w > 1:
        cols = np.arange(w)
        rows = np.arange(w)
        m = (cols[None, :] <= rows[:, None])[None, None, :, :]
        s_win = np.where(m, s_win, -np.inf)
    s = np.concatenate([s_past, s_win], axis=-1)  # [1,H,w,S+w]
    s = s - np.max(s, axis=-1, keepdims=True)
    a = np.exp(s)
    a = a / np.sum(a, axis=-1, keepdims=True)
    return a[..., :S].sum(axis=2)[0]  # [H,S]


def paged_gqa_attention(
    q: np.ndarray,
    cache,
    seq_id: int,
    layer: int,
    n_rep: int,
    q_start: int,
    causal: bool = True,
    return_weights: bool = False,
):
    """分页 GQA 注意力：从 cache gather 该序列全部有效 K/V 后走同一数学核心。

    q: [1, H, T, Dh]（RoPE 已应用）。返回 [1, H, T, Dh]；`return_weights=True`
    时同时返回 softmax 权重 [1, H, T, S]（供 T4-C H2O 策略做分数累积）。
    """
    k, v = cache.gather_heads(seq_id, layer, n_rep)
    if k.shape[2] == 0:  # 空序列：无 KV 可 attend
        out = np.zeros_like(q)
        if return_weights:
            return out, np.zeros((q.shape[0], q.shape[1], q.shape[2], 0), dtype=q.dtype)
        return out
    aw = attention_weights(q, k, q_start, causal=causal)
    out = aw @ v
    return (out, aw) if return_weights else out
