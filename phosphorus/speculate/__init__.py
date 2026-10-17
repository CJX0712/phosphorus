"""投机解码包：O1 自草稿 + Leviathan 验证 + O2 自适应控制器 + 树草稿 + n-gram 对照。

对外主入口：`generate_tokens(model, ctx_ids, ..., drafter, K, controller)` —— 把多步
自回归「串行」替换为「草稿 + 一轮验证」，输出分布与目标模型**严格相等**（不变量 ②），
且每轮至少前进 ≥ 1 个 token。详见各子模块与 ARCHITECTURE.md §6 / TASKS.md T5。
"""

from __future__ import annotations

import time

import numpy as np

from ..sampling import sampling_probs
from .base import Drafter, DrafterSpec
from .controller import SpecController, SpecPlan
from .layerskip import LayerSkipDrafter
from .ngram import NGramDrafter
from .tree import TreeNode, TreeSpec, build_balanced_tree, prune_by_prob
from .verify import VerifyResult, spec_verify

__all__ = [
    "Drafter",
    "DrafterSpec",
    "SpecController",
    "SpecPlan",
    "LayerSkipDrafter",
    "NGramDrafter",
    "TreeSpec",
    "TreeNode",
    "build_balanced_tree",
    "prune_by_prob",
    "VerifyResult",
    "spec_verify",
    "generate_tokens",
]


def generate_tokens(
    model,
    ctx_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    seed: int = 0,
    drafter: Drafter | None = None,
    K: int = 4,
    controller: SpecController | None = None,
    stream: bool = False,
    tok=None,
    draft_model=None,
) -> tuple[list[int], float, dict]:
    """投机解码生成（热 KV-cache 增量解码）：返回 (生成 token 列表, 平均接受率 α, 统计信息)。

    设计（ARCHITECTURE.md §6 / TASKS.md T5-C3）：
    * 目标模型与草稿器各持有一条**独立且热**的 KV cache（草稿器用 `model.shadow()`
      共享权重的镜像，零额外权重字节）。
    * 每轮：草稿以单 token 热追加（仅 early-exit 层）给出 K 个提议；目标模型一次性
      前向覆盖 K 个新位置（BLAS 批量，权重只加载一次）；残差拒绝采样保证输出分布与
      目标**严格相等**（不变量 ②）。
    * reject 时以 `truncate_tail` 回滚被拒草稿 token（O(T) 而非 O(L)），再追加最终
      token——整个序列的 KV 始终热，故吞吐显著优于逐 token 冷重填。
    * `controller` 处于 auto 模式且判定关闭时，退化为逐 token 串行（cache 仍热），
      分布等价不变。`drafter` 为 None 时强制串行。
    """
    rng = np.random.default_rng(seed)
    greedy = temperature <= 0
    seq = [int(x) for x in ctx_ids]
    generated: list[int] = []
    total_acc = 0
    total_prop = 0
    steps = 0

    # 草稿器独占一条热 KV cache（目标模型 `model` 自带一条），二者互不污染。
    # 调用方可传入已建好的 `draft_model`（避免每轮重建镜像）；否则惰性 shadow。
    if drafter is not None and draft_model is None:
        draft_model = model.shadow()

    # 一次性热起：预填 ctx 到两条 cache，并缓存末位 logits（下一轮的 q_0 / p_0）。
    model.reset()
    if draft_model is not None:
        draft_model.reset()
    Pt = model.forward(np.asarray(seq, dtype=np.int64))
    last_tgt = np.asarray(Pt[-1], dtype=np.float64)
    if draft_model is not None:
        Pd = draft_model.forward(
            np.asarray(seq, dtype=np.int64), stop_at_layer=drafter.draft_layers
        )
        last_dft = np.asarray(Pd[-1], dtype=np.float64)

    def _dist_of(logits: np.ndarray) -> np.ndarray:
        """logits → 采样器实际使用的分布（temperature/top-k/top-p 截断后）。"""
        return sampling_probs(logits, temperature, top_k, top_p)

    def _sample_logits(logits: np.ndarray) -> int:
        """按目标采样分布取一个 token（与 `sampling.sample` 同分布）。"""
        if greedy:
            return int(np.argmax(logits))
        p = _dist_of(logits)
        return int(rng.choice(len(p), p=p))

    def _serial_step() -> int:
        """逐 token 串行单步（cache 热，两条 cache 同步）。"""
        nonlocal last_tgt, last_dft
        y = _sample_logits(last_tgt)
        out = model.forward(np.asarray([y], dtype=np.int64))
        last_tgt = np.asarray(out[-1], dtype=np.float64)
        if draft_model is not None:
            od = draft_model.forward(
                np.asarray([y], dtype=np.int64), stop_at_layer=drafter.draft_layers
            )
            last_dft = np.asarray(od[-1], dtype=np.float64)
        generated.append(y)
        seq.append(y)
        if stream and tok:
            print(tok.decode([y]), end="", flush=True)
        return y

    while len(generated) < max_new_tokens:
        # ---- 控制器决策（auto 模式）----
        enabled, cur_K = True, max(1, int(K))
        if controller is not None:
            plan = controller.plan()
            enabled, cur_K = plan.enabled, max(1, plan.K if plan.enabled else 1)

        # 退化条件：无草稿器 / 关闭 / K<2 → 串行单步（cache 仍热）
        if drafter is None or draft_model is None or not enabled or cur_K < 2:
            y = _serial_step()
            if controller is not None:
                controller.observe(1.0, 1.0, 1, None)
            if model.eos_id is not None and y == model.eos_id:
                break
            steps += 1
            continue

        # ---- 草稿：K 个单 token 热追加到 draft_model cache（仅 early-exit 层）----
        # 注意：每个草稿 token（含最后一个）都必须前向写入草稿 cache——否则
        # 草稿 cache 与目标 cache 每轮脱同步 1 个 token，后续 q 将基于缺失
        # 尾部的前缀计算，接受率崩塌（实测 round3 α→0）。末次输出的 q_K 不
        # 参与验证（bonus 来自目标 p_K），但缓存同步是硬前提。
        t_draft = time.time()
        draft_tokens: list[int] = []
        q_dists: list[np.ndarray] = []
        q = last_dft  # q_0 = 上一轮末位草稿 logits
        for _k in range(cur_K):
            q_dists.append(_dist_of(q))
            y = _sample_logits(q)
            draft_tokens.append(y)
            od = draft_model.forward(
                np.asarray([y], dtype=np.int64),
                stop_at_layer=drafter.draft_layers,
            )
            q = np.asarray(od[-1], dtype=np.float64)
        dt_draft = time.time() - t_draft

        # ---- 验证：目标模型一次性前向覆盖 K 个新位置（热追加，BLAS 批量）----
        t_verify = time.time()
        p_0 = last_tgt
        P = model.forward(np.asarray(draft_tokens, dtype=np.int64))
        # P[i] = 目标在位置 L+i 的 logits = p_{i+1}；故 p_dists = [p_0] + P[0..K-1]
        p_dists = [_dist_of(p_0)] + [_dist_of(P[i]) for i in range(cur_K)]
        res = spec_verify(draft_tokens, q_dists, p_dists, rng, greedy=greedy)
        dt_verify = time.time() - t_verify

        accepted = res.accepted
        n_acc = len(accepted) - 1  # 被接受的草稿数
        nb = max(0, cur_K - n_acc)  # 需回滚的草稿 token 数（含被拒位置的后续）
        if nb > 0:
            model.truncate_tail(0, nb)
            draft_model.truncate_tail(0, nb)
        final = accepted[-1]  # 新 token（bonus 或残差补位），不在 cache 中
        # 追加 final 到两条 cache，并拿取下一轮 q_0 / p_0
        of = model.forward(np.asarray([final], dtype=np.int64))
        last_tgt = np.asarray(of[-1], dtype=np.float64)
        od = draft_model.forward(
            np.asarray([final], dtype=np.int64), stop_at_layer=drafter.draft_layers
        )
        last_dft = np.asarray(od[-1], dtype=np.float64)

        seq.extend(accepted)
        generated.extend(accepted)
        total_acc += res.n_accepted_drafts
        total_prop += cur_K
        if stream and tok:
            print(tok.decode(accepted), end="", flush=True)
        if controller is not None:
            c = dt_draft / max(dt_verify, 1e-9) if dt_verify > 0 else 1.0
            controller.observe(res.alpha_step, c, 1, None)
        if model.eos_id is not None and final == model.eos_id:
            break
        steps += 1

    if stream and tok:
        print()

    alpha = total_acc / max(total_prop, 1)
    info = {
        "alpha": alpha,
        "steps": steps,
        "n_new": len(generated),
        "drafts_accepted": total_acc,
        "drafts_proposed": total_prop,
    }
    return generated, alpha, info
