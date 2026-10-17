"""Leviathan et al. (2023) 投机解码验证 —— 残差拒绝采样（不变量 ② 基石）。

算法（A-12）：给定草稿提出的 K 个 token `y_0..y_{K-1}`，以及对应的
草稿分布 `q_0..q_{K-1}` 与目标分布 `p_0..p_K`（p_K 为草稿之后「bonus」位置的目标分布）：

    for i in 0..K-1:
        q, p = softmax(q_i), softmax(p_i)
        if rng.random() < min(1, p[y_i] / q[y_i]):  接受 y_i
        else: 从 r = normalize(max(0, p - q)) 采样 x，以 [y_0..y_{i-1}, x] 收尾并终止
    if 全部接受: 从 softmax(p_K) 采样 bonus token 追加（保证每轮至少前进 K+1? 至少 1 步）

关键性质：无论草稿分布 q 多差，输出分布的边际**严格等于**目标 p；且每轮至少前进
1 个 token（accept 0 个时也由残差补 1 个），这是 speedup 的下界保证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


@dataclass
class VerifyResult:
    accepted: list[int]  # 本轮实际接受的 token（含 bonus 或残差补位）
    n_proposed: int  # K（草稿长度）

    @property
    def n_new(self) -> int:
        return len(self.accepted)

    @property
    def n_accepted_drafts(self) -> int:
        # 末位要么是 bonus 要么是残差补位，均算「额外前进 1 步」，草稿接受数 = 总长 - 1
        return len(self.accepted) - 1

    @property
    def alpha_step(self) -> float:
        return self.n_accepted_drafts / max(self.n_proposed, 1)


def _spec_verify_greedy(draft_tokens: list[int], p_dists: list[np.ndarray]) -> VerifyResult:
    """确定性贪心验证：保证投机与串行贪心序列逐位相同（TASKS.md C1 第二条款）。

    接受规则：草稿 token y_i 被接受 iff y_i == argmax(p_i)。若在某位置 i 被拒绝，
    则以 argmax(p_i) 收尾（这正是串行贪心在该位置的输出），并终止本轮。全部接受时
    追加 argmax(p_K) 作为 bonus。由于草稿 token 本身也是由贪心（对 q）产生，且被接受
    的 token 恒等于目标模型的贪心 token，故整条序列与逐 token 贪心解码严格一致。
    """
    K = len(draft_tokens)
    accepted: list[int] = []
    for i in range(K):
        p = np.asarray(p_dists[i], dtype=np.float64)
        g = int(np.argmax(p))
        y = int(draft_tokens[i])
        if y == g:
            accepted.append(y)
            continue
        # 拒绝：输出目标贪心 token 收尾（与串行贪心严格一致）
        return VerifyResult(accepted + [g], K)
    # 全部接受：追加 bonus = 目标在末位的贪心 token
    pb = np.asarray(p_dists[K], dtype=np.float64)
    g = int(np.argmax(pb))
    return VerifyResult(accepted + [g], K)


def spec_verify(
    draft_tokens: list[int],
    q_dists: list[np.ndarray],
    p_dists: list[np.ndarray],
    rng: np.random.Generator,
    greedy: bool = False,
) -> VerifyResult:
    """执行一轮 Leviathan 验证。

    参数
    ----
    draft_tokens : 草稿 token y_0..y_{K-1}
    q_dists      : 长度 K，q_dists[i] = 第 i 个**提议分布（概率向量，Σ=1）**
    p_dists      : 长度 K+1，p_dists[i] = 第 i 个**目标分布（概率向量，Σ=1）**；
                   p_dists[K] = bonus 位置目标分布
    rng          : numpy 随机生成器

    注意：q/p 必须是**采样器真正采样时所依据的分布**（temperature + top-k + top-p
    截断之后），即 `sampling.sampling_probs(logits, ...)` 的输出——否则在 top_p<1
    时输出分布与串行采样不等价（不变量 ② 失效）。本函数不再内部 softmax。
    greedy       : 若 True 走确定性贪心验证（temperature=0 专用）：接受 iff y_i 是
                   p_i 的 argmax；拒绝则补 argmax(p_i)；bonus = argmax(p_K)。此时输出
                   序列与目标模型贪心解码**逐位相同**（不变量 ② 的贪心特例）。

    返回 VerifyResult。
    """
    K = len(draft_tokens)
    if len(q_dists) != K:
        raise ValueError(f"q_dists 长度 {len(q_dists)} != K {K}")
    if len(p_dists) != K + 1:
        raise ValueError(f"p_dists 长度 {len(p_dists)} != K+1 {K + 1}")

    if greedy:
        return _spec_verify_greedy(draft_tokens, p_dists)

    accepted: list[int] = []
    for i in range(K):
        y = int(draft_tokens[i])
        q = np.asarray(q_dists[i], dtype=np.float64)
        p = np.asarray(p_dists[i], dtype=np.float64)
        qy = float(q[y])
        if qy <= 0.0:
            # 草稿分布未给该 token 任何质量：用残差补位（退化为纯目标采样）
            r = np.maximum(0.0, p - q)
            s = r.sum()
            dist = (r / s) if s > 0 else p
            x = int(rng.choice(len(dist), p=dist))
            return VerifyResult(accepted + [x], K)
        accept_prob = min(1.0, float(p[y]) / qy)
        if rng.random() < accept_prob:
            accepted.append(y)
            continue
        # 拒绝：从残差分布采样一个 token 收尾
        r = np.maximum(0.0, p - q)
        s = r.sum()
        if s <= 0.0:
            # p 处处 <= q：几乎完全重叠，直接接受 y 以保证前进
            return VerifyResult(accepted + [y], K)
        r = r / s
        x = int(rng.choice(len(r), p=r))
        return VerifyResult(accepted + [x], K)

    # 全部接受：追加 bonus token（来自 p_K）
    pb = np.asarray(p_dists[K], dtype=np.float64)
    x = int(rng.choice(len(pb), p=pb))
    return VerifyResult(accepted + [x], K)
