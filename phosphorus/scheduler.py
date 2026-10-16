"""连续批处理调度器（T4-D · C6/C7 验收）。

把多条序列放进**同一个**融合前向（`PhosModel.forward_step`）里并发解码，
从而把 Python 层循环开销与 BLAS 批处理收益最大化；相比逐条串行 `generate`，
在 8 条序列下要求吞吐 ≥ 2×（C6）。采样器复用 `sampling.sample`（top-k /
top-p / temperature，同 seed 可复现，C7）。

设计要点（与 ARCHITECTURE.md §5.5 对齐）：
* **KV 隔离**：paged 路径下每条序列用独立 `seq_id`，只 attend 自己的 KV、
  用各自的绝对 RoPE 位置。因此 batch 内任意两条序列**互不干扰**——调度器
  跑出的 token id 与「逐条串行、各序列 KV 独立」的结果**逐位相同**（这点由
  `tests/test_scheduler.py` 的 C6 测试硬验证）。
* **贪心等价（C6）**：`temperature=0` 时 `sample` 退化为 argmax，与串行
  `generate` 产出的 token id 完全一致。
* **波次（wave）切分**：当 prompt 数 > `max_seqs` 时，按 `max_seqs` 分组，
  每组 `reset()` 后独立跑；跨组不共享 KV，故仍等价于串行。
* **驱逐不混用**：`forward_step` 要求 `eviction == "none"`（T4-D 范围）；
  若模型带驱逐则 `generate` 会透传 NotImplementedError。
"""

from __future__ import annotations

import numpy as np

from .sampling import sample as _sample


class Scheduler:
    """连续批处理调度器：一次融合前向服务多条序列。

    >>> sched = Scheduler(model, max_seqs=8)
    >>> texts = sched.generate(["Hello", "World"], max_new_tokens=32)
    """

    def __init__(self, model, max_seqs: int | None = None) -> None:
        self.model = model
        # 调度并发上限取模型构造时的 max_seqs（物理池按它分配）。
        self.max_seqs = (
            int(max_seqs) if max_seqs is not None else int(getattr(model, "max_seqs", 1))
        )
        if model.kv_mode != "paged":
            # dense 路径 forward_step 退化为逐序列串行，无融合收益但仍可跑。
            pass

    # ----------------------------- 公开 API ----------------------------- #
    def generate_ids(
        self,
        prompt_ids: list[list[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.95,
        seed: int = 0,
    ) -> list[list[int]]:
        """对一批 prompt（已是 token id）做连续批处理解码，返回各条**新生成**的 id。

        返回顺序与 `prompt_ids` 严格对齐（按 group 内 seq_id 回填）。
        """
        model = self.model
        n = len(prompt_ids)
        if n == 0:
            return []
        results: list[list[int] | None] = [None] * n

        for w0 in range(0, n, self.max_seqs):
            batch = list(range(w0, min(w0 + self.max_seqs, n)))
            size = len(batch)
            model.reset()

            # 组内局部 seq_id = 在 batch 中的位置（0..size-1），必 < max_seqs。
            seq_of: dict[int, int] = {local: batch[local] for local in range(size)}
            ctx_of: dict[int, list[int]] = {
                local: list(prompt_ids[seq_of[local]]) for local in range(size)
            }

            # 预填充：各组 prompt 一次性写入（融合并行）。
            prefill = {local: np.array(ctx_of[local], dtype=np.int64) for local in range(size)}
            last_logits = model.forward_step(prefill)

            # 每序列独立 rng：seed + 全局 prompt 下标 → 与「逐条串行」同 seed 可复现。
            rngs = {local: np.random.default_rng(seed + seq_of[local]) for local in range(size)}
            gens: dict[int, list[int]] = {local: list(ctx_of[local]) for local in range(size)}
            next_ids: dict[int, int] = {}
            done: dict[int, bool] = {}
            for local in range(size):
                lg = last_logits[local]
                nid = _sample(lg, temperature, top_k, top_p, rngs[local])
                gens[local].append(nid)
                done[local] = self._finished(model, gens[local], ctx_of[local], nid, max_new_tokens)
                if not done[local]:
                    next_ids[local] = nid

            # 解码循环：每步只喂「未完成」的序列（已完成的不再进 batch，不影响他人）。
            for _ in range(max_new_tokens - 1):
                if all(done.values()):
                    break
                active = {
                    local: np.array([next_ids[local]], dtype=np.int64)
                    for local in range(size)
                    if not done[local]
                }
                if not active:
                    break
                lls = model.forward_step(active)
                for local in active:
                    nid = _sample(lls[local], temperature, top_k, top_p, rngs[local])
                    gens[local].append(nid)
                    done[local] = self._finished(
                        model, gens[local], ctx_of[local], nid, max_new_tokens
                    )
                    if not done[local]:
                        next_ids[local] = nid

            for local in range(size):
                g = seq_of[local]
                results[g] = gens[local][len(ctx_of[local]) :]  # 去掉 prompt 部分

        return [r if r is not None else [] for r in results]

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.95,
        seed: int = 0,
    ) -> list[str]:
        """对一批 prompt（文本）做连续批处理解码，返回各条生成的文本。"""
        tok = self.model.tok
        if tok is None:
            raise RuntimeError("该 .phos 不含 tokenizer，无法做文本调度（请用 generate_ids）")
        bos = self.model.bos_id
        prompt_ids: list[list[int]] = []
        for p in prompts:
            ctx = tok.encode(p)
            if not ctx:
                ctx = [bos] if (bos is not None and 0 <= bos < self.model.vocab) else [0]
            prompt_ids.append(ctx)
        gen_ids = self.generate_ids(prompt_ids, max_new_tokens, temperature, top_k, top_p, seed)
        return [tok.decode(ids) for ids in gen_ids]

    # ----------------------------- 内部 ----------------------------- #
    @staticmethod
    def _finished(model, gen: list[int], ctx: list[int], last_id: int, max_new_tokens: int) -> bool:
        if model.eos_id is not None and last_id == model.eos_id:
            return True
        return (len(gen) - len(ctx)) >= max_new_tokens
