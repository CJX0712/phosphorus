"""KV 驱逐策略族（T4-C）：SnapKV / PyramidKV / H2O / StreamingLLM。

四种策略共享同一个接口（`EvictionPolicy.select`）与同一套「强制保留」规则，
差异只在**重要性分数怎么来**与**每层预算怎么分**：

| 策略 | 分数来源 | 预算 | 池化 |
|---|---|---|---|
| SnapKV | 观测窗口（末尾 W 个 query）对 past key 的注意力，softmax 后对 query 维求和 | 逐层相同 | avg/max pool（kernel 5） |
| PyramidKV | 同 SnapKV | **逐层金字塔**（`BudgetAllocator`） | 同 SnapKV |
| H2O | **全部** query 对 past key 的注意力（累积分数） | 逐层相同 | 无（论文原版即不池化） |
| StreamingLLM | 不用分数 | 逐层相同 | — |

**强制保留**（所有策略一致，对应 `ARCHITECTURE.md` §A-8~A-11）：
序列最前 `sink_size`（默认 4）个 attention sink + 末尾 `window_size`（默认 32）
个最近 token 永不被驱逐——这使任意预算下的输出都有语义锚点。

来源（详见 `docs/ALGORITHMS.md` A-8~A-11 与 `NOTICE`）：
* SnapKV · [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)（Apache-2.0）
* PyramidKV · [arXiv:2406.02069](https://arxiv.org/abs/2406.02069)（Apache-2.0）
* H2O · [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)（MIT）
* StreamingLLM · [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)（MIT）

参考实现：<https://github.com/Zefan-Cai/PyramidKV>（`pyramidkv/pyramidkv_utils.py`，
同一份代码里四种策略并列实现，本模块的分数/池化/预算公式以其为准）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 跨 head 聚合：物理块为全 head 共享，无法 per-head 保留（vLLM 滑动窗口同款约束），
# 故把 [H, T] 的 per-head 分数聚合到 token 粒度。均值对离群 head 更鲁棒。
HEAD_AGG = ("mean", "max")


def _pool1d(x: np.ndarray, kernel: int, mode: str) -> np.ndarray:
    """SnapKV 的 avg/max pool1d（kernel_size=k, padding=k//2, stride=1）。

    与 `F.avg_pool1d` / `F.max_pool1d` 语义一致（含奇数核的右侧多补一）。
    """
    if kernel <= 1:
        return x
    pad = kernel // 2
    xp = np.concatenate([np.full(pad, x[0], dtype=x.dtype), x, np.full(kernel - 1 - pad, x[-1], dtype=x.dtype)])
    if mode == "avgpool":
        c = np.convolve(xp, np.ones(kernel, dtype=x.dtype) / kernel, mode="valid")
    elif mode == "maxpool":
        # 滑窗最大值（朴素实现，长度 < 1e5 时可接受）
        c = np.stack([np.max(xp[i : i + kernel]) for i in range(len(xp) - kernel + 1)])
    else:
        raise ValueError(f"未知池化 {mode!r}")
    return c[: len(x)]


@dataclass
class EvictionConfig:
    """驱逐配置（默认值对应论文常用设置）。"""

    budget_ratio: float = 0.5  # 保留比例（0.5 = 保留一半）
    window_size: int = 32  # 观测窗口 / 强制保留的最近窗口 W
    sink_size: int = 4  # 强制保留的 attention sink 数
    kernel_size: int = 5  # SnapKV 池化核
    pooling: str = "avgpool"  # avgpool | maxpool | none
    head_agg: str = "mean"  # 跨 head 聚合：mean | max
    beta: int = 20  # PyramidKV 的 min:max 比例
    m_min: int = 8  # 每层保留下限（防止深层被压到 0）


class EvictionPolicy:
    """策略基类：把「分数 + 预算」变成「保留掩码」。"""

    def __init__(self, cfg: EvictionConfig | None = None) -> None:
        self.cfg = cfg or EvictionConfig()
        if self.cfg.head_agg not in HEAD_AGG:
            raise ValueError(f"head_agg 必须是 {HEAD_AGG}，收到 {self.cfg.head_agg!r}")

    name = "base"

    # ----------------------------- 预算 ----------------------------- #
    def layer_budget(self, layer_idx: int, n_layer: int, seq_len: int) -> int:
        """该层保留的 KV 总数（含 sink + window）。

        注意**不能**用 `max(sink+window, ...)` 兜底：那会让小预算时缓存
        永远停在 sink+window 上，每一步 decode 都触发一次全量压缩。
        有效保留下限由 `select` 自然形成（预算低于 sink+window 时只保留
        这两段），这里只保证不小于 1。
        """
        total = int(round(seq_len * self.cfg.budget_ratio))
        return max(1, min(total, seq_len))

    # ----------------------------- 选择 ----------------------------- #
    def _aggregate(self, scores: np.ndarray) -> np.ndarray:
        """[H, T_past] -> [T_past]（跨 head 聚合）。"""
        return scores.mean(axis=0) if self.cfg.head_agg == "mean" else scores.max(axis=0)

    def _pool(self, per_tok: np.ndarray) -> np.ndarray:
        if self.cfg.pooling in ("avgpool", "maxpool"):
            return _pool1d(per_tok, self.cfg.kernel_size, self.cfg.pooling)
        return per_tok  # "none"（H2O / StreamingLLM）

    def select(self, scores: np.ndarray, budget: int, n_total: int) -> np.ndarray:
        """返回 keep_mask: [n_total] bool。

        `scores`: [H, T_past] 候选 past token 的重要性（T_past = n_total - sink - window，
        由调用方按 `n_past_of(n_total)` 切好）；`budget`: 该层总保留数。
        """
        sink = min(self.cfg.sink_size, n_total)
        window = min(self.cfg.window_size, max(0, n_total - sink))
        n_past = n_total - sink - window
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != n_past:
            raise ValueError(f"scores 形状应为 [H, {n_past}]，收到 {scores.shape}")
        mask = np.zeros(n_total, dtype=bool)
        mask[:sink] = True
        if window:
            mask[n_total - window :] = True
        if n_past == 0:
            return mask
        b = int(budget) - sink - window
        if b >= n_past:
            mask[sink : sink + n_past] = True
            return mask
        if b <= 0:
            return mask
        per_tok = self._pool(self._aggregate(scores))
        # top-b：分数相同取更早的 token（稳定可复现）
        idx = np.argpartition(-per_tok, b - 1)[:b]
        idx = np.sort(idx)
        mask[sink + idx] = True
        return mask

    def n_past_of(self, n_total: int) -> int:
        """可被驱逐的候选 token 数（= n_total - sink - window）。"""
        sink = min(self.cfg.sink_size, n_total)
        window = min(self.cfg.window_size, max(0, n_total - sink))
        return max(0, n_total - sink - window)


class SnapKVPolicy(EvictionPolicy):
    """SnapKV：观测窗口打分 + 池化 + 固定逐层预算。"""

    name = "snapkv"


class H2OPolicy(EvictionPolicy):
    """H2O：全矩阵累积分数，不池化（论文原版）。"""

    name = "h2o"

    def __init__(self, cfg: EvictionConfig | None = None) -> None:
        super().__init__(cfg)
        self.cfg.pooling = "none"


class StreamingLLMPolicy(EvictionPolicy):
    """StreamingLLM：不用分数，只保留 sink + 最近窗口。"""

    name = "streamingllm"

    def select(self, scores: np.ndarray, budget: int, n_total: int) -> np.ndarray:
        """不用分数，只保留 sink 段 + 最近窗口。

        与内容型策略一致：窗口随预算伸缩（`window = budget - sink`），使
        `kv_budget` 对 StreamingLLM 同样有效（不再是固定 sink+window_size）。
        `scores` 形参保留以复用统一接口，本策略不使用。
        """
        sink = min(self.cfg.sink_size, n_total)
        budget = max(sink, min(int(budget), n_total))
        window = budget - sink  # 最近窗口随预算收窄/放宽
        mask = np.zeros(n_total, dtype=bool)
        mask[:sink] = True
        if window > 0:
            mask[n_total - window :] = True
        return mask


class PyramidKVPolicy(EvictionPolicy):
    """PyramidKV：观测窗口打分 + 池化 + 逐层金字塔预算。

    预算公式照抄官方实现（`pyramidkv/pyramidkv_utils.py`）::

        min_num = (C - W) // beta
        max_num = (C - W) * 2 - min_num          # C = 均匀预算, W = window
        steps   = (max_num - min_num) // (L - 1)
        budget_l = max_num - layer_idx * steps

    该公式保证 Σ_l budget_l ≈ L·(C-W)（总量守恒）；整数除法的残差用
    **最大余数法**补到最早的层，使 Σ 精确等于 L·(C-W)（C5 验收 ±1）。
    """

    name = "pyramidkv"

    def layer_budgets(self, n_layer: int, seq_len: int) -> list[int]:
        """逐层总保留数（含 sink + window），Σ == L·C ± 1（C 为均匀预算）。

        金字塔只作用在**可驱逐候选段** c = C - W - sink 上：候选容量逐层递减，
        强制保留段（sink + window）每层相同，故总量与 SnapKV 的均匀分配守恒。

        对齐 PyramidKV 官方实现（pyramidkv_utils.py）的两条分支：
        * 短序列（`seq_len < 2·c`）：各层固定预算 c（不套金字塔），对应官方
          `elif q_len < (max_capacity_prompt - window_size) * 2` 分支；
        * 长序列：用金字塔公式，并做 `max_num` 钳制（候选数 ≤ 可用候选段）。
        整数残差用**最大余数法**补到精确等于 L·c（C5 验收要求 Σ == L·C ± 1）。
        """
        C = int(round(seq_len * self.cfg.budget_ratio))
        W = self.cfg.window_size
        sink = min(self.cfg.sink_size, seq_len)
        L = max(n_layer, 1)
        c = max(C - W - sink, 0)  # 候选段总量目标 = L·c
        if L == 1 or c == 0:
            return [C] * L
        # 短序列分支：各层统一保留 c（官方短序列分支）
        if seq_len < 2 * c:
            return [c + W + sink] * L
        # 长序列分支：金字塔
        min_num = max(c // self.cfg.beta, self.cfg.m_min, 0)
        max_num = c * 2 - min_num
        cap = seq_len - W - sink  # 可用候选段上限
        if max_num > cap:
            max_num = cap
            min_num = max(c * 2 - max_num, 0)
        if max_num < min_num:
            max_num = min_num
        steps = (max_num - min_num) // (L - 1)
        base = [max_num - i * steps for i in range(L)]
        # 最大余数法修正：把 base 之和精确补齐到 L*c
        target = L * c
        diff = target - sum(base)
        order = sorted(range(L), key=lambda i: (-(base[i]), i))
        i = 0
        while diff != 0 and order:
            j = order[i % len(order)]
            if diff > 0:
                base[j] += 1
                diff -= 1
            elif base[j] > self.cfg.m_min:
                base[j] -= 1
                diff += 1
            else:
                order.remove(j)
                if not order:
                    break
            i += 1
        # 每层总保留 = 金字塔候选容量 + 强制保留（sink + window）
        return [b + W + sink for b in base]

    def layer_budget(self, layer_idx: int, n_layer: int, seq_len: int) -> int:
        return self.layer_budgets(n_layer, seq_len)[layer_idx]


class BudgetAllocator:
    """逐层预算分配器（金字塔 / 均匀），供 PyramidKV 与基准对照使用。"""

    def __init__(self, cfg: EvictionConfig | None = None) -> None:
        self.cfg = cfg or EvictionConfig()

    def layer_budgets(self, n_layer: int, seq_len: int, mode: str = "pyramid") -> list[int]:
        if mode == "uniform":
            C = int(round(seq_len * self.cfg.budget_ratio))
            return [max(C, 1)] * n_layer
        if mode == "pyramid":
            return PyramidKVPolicy(self.cfg).layer_budgets(n_layer, seq_len)
        raise ValueError(f"未知分配模式 {mode!r}")


# 策略注册表（CLI --eviction 的取值；"none" 表示不驱逐，由引擎层处理）
POLICIES: dict[str, type[EvictionPolicy]] = {
    "snapkv": SnapKVPolicy,
    "pyramidkv": PyramidKVPolicy,
    "h2o": H2OPolicy,
    "streamingllm": StreamingLLMPolicy,
}
EVICTIONS: tuple[str, ...] = ("none",) + tuple(sorted(POLICIES))


def make_policy(name: str, cfg: EvictionConfig | None = None) -> EvictionPolicy:
    if name not in POLICIES:
        raise ValueError(f"未知驱逐策略 {name!r}，可选 {sorted(POLICIES)}")
    return POLICIES[name](cfg)
