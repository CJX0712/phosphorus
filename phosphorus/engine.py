"""纯 numpy 推理引擎（运行时零 torch 依赖，ARCHITECTURE.md §4）。

加载 `.phos` -> 复现 LLaMA 变体的前向（Pre-RMSNorm / RoPE / GQA / ReLU² / SubLN），
使用三值 TL GEMM（W_ternary @ xq，等价 int32 参考）与可选 HGF 低秩校正，
支持 KV cache 的 prefill / decode 与 temperature / top-k / top-p 采样。

KV cache 两种模式（`kv_mode`）：
* `dense` —— 每层一份 `[1, n_head, T, d_head]` 全量张量（参考实现）；
* `paged` —— 分页物理池 + block_tables（vLLM 语义，T4-B 默认），
  只存 `n_kv_head` 份，读取时 GQA 展开；两种模式数值等价（不变量 ⑨）。

借鉴来源：
    * llama2.c 的 Llama.run / kv cache 设计（Karpathy）— A-16
    * BitNet 三值 GEMM（见 gemm.py，TL 路径数值权威）— A-1/A-3
    * RoPE 旋转编码（Su et al., 2021）— A-16
    * HGF 低秩 FP 校正（Pizzo, 2026）— A-11
    * vLLM PagedAttention 块管理语义（Apache-2.0）— A-7 / A-25

不变量（详见 docs/ARCHITECTURE.md §7）：
    ① engine 的三值 GEMM 与 gemm.int32_reference 在 int32 域逐位一致
    ③ pack/unpack 往返逐位恒等（由 loader+pkg 保证）
    ④ RoPE 与 model.apply_rope 数值一致（同一公式）
    ⑤ encode->decode 恒等（tokenizer）
    ⑦ 同一输入 engine 与 torch 模型 logits 相对误差 < 5%
    ⑨ paged 注意力与 dense 参考最大绝对差 < 1e-5（T4-B）
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .attention import attention_weights, kv_importance_scores, paged_gqa_attention
from .eviction import EVICTIONS, EvictionConfig, make_policy
from .gemm import gemm_int8, tl_acc_i32
from .loader import PhosFile, load_phos
from .packing import k_pad, unpack_code5
from .sampling import sample as _sample

logger = logging.getLogger(__name__)

# 四种 GEMM 后端（ARCHITECTURE.md §5）。
#   fp32 —— 反量化后的 fp32 基线，与 torch QAT 前向逐元素等价（不变量 ⑦ 的对照面）
#   int8 —— 权重/激活各自 absmax 量化到 int8，int32 累加（精度对照用）
#   tl   —— 真·LUT：直接吃 5-bit 打包 code5，查 14 项 LUT，无乘法
#   i2s  —— 真·零点校正：int8 镜像权重 × uint8 激活 − 128·row_wsum
BACKENDS: tuple[str, ...] = ("fp32", "int8", "tl", "i2s")

# KV cache 模式（ARCHITECTURE.md §5.4 / TASKS.md C4）。
KV_MODES: tuple[str, ...] = ("dense", "paged")
# KV 驱逐策略（T4-C）：none = 不驱逐；其余见 eviction.EVICTIONS。
DEFAULT_BLOCK_SIZE = 16


def _quant_per_token(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per-token 激活量化：返回 (xq int8 [T,in], x_scale float32 [T])。"""
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x), axis=1, keepdims=True)
    xs = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    xq = np.clip(np.round(x / xs), -127, 127).astype(np.int8)
    return xq, xs.reshape(-1)


def _ternary_gemm_fp32(W: np.ndarray, w_scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    """忠实路径（默认）：out[T,out] = (W_ternary * w_scale) @ x_fp.T。

    与 torch QAT 前向逐元素等价（权重三值、激活 fp32），用于不变量 ⑦ 验证。
    """
    Wd = W.astype(np.float32) * w_scale.astype(np.float32)[:, None]
    return (Wd @ x.T).T  # [T, out]


def _ternary_gemm_i2s(W: np.ndarray, w_scale: np.ndarray, xq: np.ndarray, x_scale: np.ndarray):
    """I2_S 路径（向量化）：out[t,n] = (Σ_j W[n,j]·(xq[t,j]+128) − 128·row_wsum[n]) · s。"""
    W32 = np.asarray(W, dtype=np.int32)
    xu8 = np.asarray(xq, dtype=np.int32) + 128
    row_sum = W32.sum(axis=1)
    acc = (W32 @ xu8.T) - (128 * row_sum).reshape(-1, 1)  # [out, T]
    return (
        (
            acc.astype(np.float64)
            * np.asarray(w_scale, dtype=np.float64).reshape(-1, 1)
            * np.asarray(x_scale, dtype=np.float64).reshape(1, -1)
        )
        .astype(np.float32)
        .T
    )  # [T, out]


def _ternary_gemm_tl(codes: np.ndarray, K: int, w_scale: np.ndarray, xq, x_scale) -> np.ndarray:
    """TL 路径（真·查表）：逐 token 用 14 项 LUT 累加。

    NumPy 无法把「按 code5 查表」向量化到 batch 维（每个 token 的激活不同，
    LUT 也不同），故逐 token 调用 `tl_acc_i32`——这正是 decode（T=1）的形态。
    prefill 走 i2s 更快，但两条路径在 int32 域逐位相等（不变量 ①）。
    """
    xq = np.asarray(xq, dtype=np.int8)
    out = np.zeros((xq.shape[0], codes.shape[0]), dtype=np.float32)
    for t in range(xq.shape[0]):
        acc = tl_acc_i32(codes, xq[t], K)
        out[t] = (acc.astype(np.float64) * w_scale.astype(np.float64) * float(x_scale[t])).astype(
            np.float32
        )
    return out


class _KVCache:
    def __init__(self, n_layer: int, n_head: int, d_head: int) -> None:
        self.n_layer = n_layer
        self.k: list[np.ndarray] = [
            np.zeros((1, n_head, 0, d_head), dtype=np.float32) for _ in range(n_layer)
        ]
        self.v: list[np.ndarray] = [
            np.zeros((1, n_head, 0, d_head), dtype=np.float32) for _ in range(n_layer)
        ]

    def reset(self) -> None:
        for i in range(self.n_layer):
            self.k[i] = self.k[i][:, :, :0, :]
            self.v[i] = self.v[i][:, :, :0, :]

    def len(self) -> int:
        return int(self.k[0].shape[2])


def _build_rope(theta: float, d_head: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float64) / d_head))
    t = np.arange(max_len, dtype=np.float64)
    freqs = np.outer(t, inv)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _apply_rope(q: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """q: [1, H, T, dh]；与 model.apply_rope 完全一致（相邻配对旋转）。"""
    q1 = q[..., 0::2]
    q2 = q[..., 1::2]
    T = cos.shape[0]
    c = cos.reshape(1, 1, T, -1)  # [1,1,T,dh/2]
    s = sin.reshape(1, 1, T, -1)
    out = np.empty_like(q)
    out[..., 0::2] = q1 * c - q2 * s
    out[..., 1::2] = q2 * c + q1 * s
    return out


class PhosModel:
    """加载 .phos 并执行推理的纯 numpy 模型。"""

    def __init__(
        self,
        phos: PhosFile,
        backend: str = "fp32",
        native: str | None = None,
        kv_mode: str = "paged",
        block_size: int = DEFAULT_BLOCK_SIZE,
        eviction: str = "none",
        kv_budget: float = 0.5,
        window_size: int = 32,
        sink_size: int = 4,
    ) -> None:
        """构造推理模型。

        backend ∈ {"fp32", "int8", "tl", "i2s"}（见模块顶 `BACKENDS`）。
        默认 `fp32`：反量化后做 fp32 矩阵乘，与 torch QAT 前向逐元素等价，
        是不变量 ⑦（engine ↔ torch 相对误差 < 5%）的对照面。
        `tl` / `i2s` 是部署态的整数路径，二者在 int32 域逐位相等。

        kv_mode ∈ {"dense", "paged"}：`paged`（默认，T4-B）用分页 KV 池 +
        block_tables；`dense` 是全量张量参考实现。二者数值等价（不变量 ⑨）。

        eviction ∈ {"none", "snapkv", "pyramidkv", "h2o", "streamingllm"}（T4-C）：
        KV 驱逐策略，仅在 `kv_mode="paged"` 下生效，把长序列 KV 压到预算内
        （逐层预算 + 逐层保留掩码，不变量 ⑦——相对 L2 误差 < 10%）。`none`
        表示完全不驱逐。`kv_budget` 为保留比例（默认 0.5 = 保留一半）。
        `window_size`/`sink_size` 为强制保留的「最近窗口/注意力锚点」大小
        （StreamingLLM 的发现；其余三策略原版不保留 sink，本项目统一保留，
        属组合改进，已在 `docs/ALGORITHMS.md` A-8~A-11 标注）。
        """
        if backend not in BACKENDS:
            raise ValueError(f"未知后端 {backend!r}，可选 {BACKENDS}")
        if kv_mode not in KV_MODES:
            raise ValueError(f"未知 kv_mode {kv_mode!r}，可选 {KV_MODES}")
        self.backend = backend
        self.kv_mode = kv_mode
        self.phos = phos
        h = phos.header
        self.d = int(h["d_model"])
        self.n_layer = int(h["n_layer"])
        self.n_head = int(h["n_head"])
        self.n_kv_head = int(h["n_kv_head"])
        self.d_head = int(h["d_head"])
        self.d_ff = int(h["d_ff"])
        self.vocab = int(h["vocab"])
        self.max_seq_len = int(h["max_seq_len"])
        self.rope_theta = float(h["rope_theta"])
        self.norm_eps = float(h["norm_eps"])
        self.act_relu2 = bool(h["act"])
        self.hgf_alpha = float(h["hgf_alpha"])
        self.early_exit_layer = int(h["early_exit_layer"])
        self.n_rep = self.n_head // self.n_kv_head
        self.tied = bool(h["flags"] & (1 << 1))

        self.w = phos.weights
        self.scales = phos.scales
        self.hgf = phos.hgf
        self.tok = phos.tokenizer

        self.tok_emb = self.w["tok_emb"].astype(np.float32)
        self.lm_head = None if self.tied else self.w["lm_head"].astype(np.float32)
        self.norm_final = self.w["norm_final"].astype(np.float32)

        # 预计算 RoPE 表（覆盖最大序列）
        self._cos, self._sin = _build_rope(self.rope_theta, self.d_head, self.max_seq_len + 16)
        self.block_size = int(block_size)
        if self.kv_mode == "paged":
            from .block_manager import BlockManager
            from .kv_cache import PagedKVCache

            n_blocks = (
                self.max_seq_len + self.block_size - 1
            ) // self.block_size + 1  # +1 = null_block
            self._mgr = BlockManager(
                n_blocks=n_blocks,
                block_size=self.block_size,
                n_seqs=1,
                max_blocks_per_seq=n_blocks - 1,
            )
            self.cache = PagedKVCache(self._mgr, self.n_layer, self.n_kv_head, self.d_head)
        else:
            self._mgr = None
            self.cache = _KVCache(self.n_layer, self.n_head, self.d_head)

        # ---- T4-C：KV 驱逐（不变量 ⑦）------------------------------------- #
        # 仅在 paged 模式下启用；层数/长度以物理池为准。
        if eviction not in EVICTIONS:
            raise ValueError(f"未知驱逐策略 {eviction!r}，可选 {EVICTIONS}")
        self.eviction = eviction
        self.kv_budget = float(kv_budget)
        self._abs_dropped = 0  # 历史上被丢弃的 token 总数（RoPE 绝对位置重基准）
        self._evict = None
        if eviction != "none":
            if kv_mode != "paged":
                raise ValueError("驱逐策略仅在 kv_mode='paged' 下可用")
            self._evict = make_policy(
                eviction,
                EvictionConfig(
                    budget_ratio=self.kv_budget,
                    window_size=window_size,
                    sink_size=sink_size,
                ),
            )
            # 高水位触发：达到 max_seq_len×budget 才压缩（vLLM 同款策略），
            # 压缩目标 = 压缩瞬间序列长度的 ratio（对 50% 预算即压到一半）。
            self._evict_threshold = max(
                sink_size + window_size + 1,
                int(round(self.max_seq_len * self.kv_budget)),
            )
            # 每层本步窗口打分（SnapKV/PyramidKV）；H2O 用跨步累积分数。
            self._win_scores: list[np.ndarray] = [
                np.zeros((self.n_head, 0), dtype=np.float32) for _ in range(self.n_layer)
            ]
            self._h2o: list[np.ndarray] = [
                np.zeros((self.n_head, 0), dtype=np.float32) for _ in range(self.n_layer)
            ]

        # 特殊 token id
        self.bos_id = self.tok.specials.get("<|bos|>") if self.tok else None
        self.eos_id = self.tok.specials.get("<|eos|>") if self.tok else None

        self._proj_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        # TL 后端需要 5-bit 打包流的 code5；按张量惰性解出并缓存。
        self._packed = phos.raw
        self._codes: dict[str, np.ndarray] = {}

        # ---- 实现后端（T4）：native C99 内核 / numpy ---------------------- #
        # 与数学后端（self.backend）正交：tl/i2s 可由原生内核计算。
        # 默认 auto：native 可用则启用，不可用静默回落 numpy（不改变默认数值路径）。
        self._row_wsum: dict[str, np.ndarray] = {}
        self._packed_arr: dict[str, np.ndarray] = {}
        self._kern = None
        self.impl = "numpy"
        try:
            from .backend import resolve as _resolve_impl

            self.impl = _resolve_impl(native)
            if self.impl == "native":
                from .native import kernels as _kernels

                self._kern = _kernels()
        except Exception as exc:  # noqa: BLE001 - 原生不可用绝不阻断推理
            logger.warning("phosphorus: native 实现后端不可用，回退 numpy（%s）", exc)
            self.impl = "numpy"
            self._kern = None

    def _codes_of(self, wname: str) -> np.ndarray:
        """取某三值权重的 code5 数组 [out, ng]（惰性构建 + 缓存）。"""
        if wname not in self._codes:
            if wname not in self._packed:
                raise KeyError(f"{wname} 无打包字节（该 .phos 不含原始 5-bit 流）")
            W = self.w[wname]
            self._codes[wname] = unpack_code5(self._packed[wname], W.shape[0], W.shape[1]).astype(
                np.int64
            )
        return self._codes[wname]

    # ----------------------------- 工厂 ----------------------------- #
    @classmethod
    def from_file(
        cls,
        path,
        backend: str = "fp32",
        native: str | None = None,
        kv_mode: str = "paged",
        block_size: int = DEFAULT_BLOCK_SIZE,
        eviction: str = "none",
        kv_budget: float = 0.5,
        window_size: int = 32,
        sink_size: int = 4,
    ) -> PhosModel:
        return cls(
            load_phos(path),
            backend=backend,
            native=native,
            kv_mode=kv_mode,
            block_size=block_size,
            eviction=eviction,
            kv_budget=kv_budget,
            window_size=window_size,
            sink_size=sink_size,
        )

    # ----------------------------- 工具 ----------------------------- #
    def _rmsnorm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        return (x * (1.0 / np.sqrt(var + self.norm_eps)) * w).astype(np.float32)

    def _row_wsum_of(self, wname: str) -> np.ndarray:
        """row_wsum[i] = Σ_j W[i,j]（.phos 未序列化，从解包权重现算并缓存）。"""
        if wname not in self._row_wsum:
            self._row_wsum[wname] = self.w[wname].astype(np.int32).sum(axis=1)
        return self._row_wsum[wname]

    def _packed_of(self, wname: str) -> np.ndarray:
        """5-bit 打包字节的 uint8 视图（TL 原生内核直接消费，零拷贝）。"""
        if wname not in self._packed_arr:
            self._packed_arr[wname] = np.frombuffer(self._packed[wname], dtype=np.uint8)
        return self._packed_arr[wname]

    def _use_native(self, T: int, N: int | None = None) -> bool:
        """按实测 crossover 决定是否走原生内核（数字见 docs/BENCHMARKS.md §T4-A）。

        * `tl`：numpy 侧 `_ternary_gemm_tl` 是**逐 token Python 循环**，
          native 在全部实测形状都快 2.8x~6.8x → 总是走 native。
        * `i2s`：numpy 侧是**批量矩阵乘**，小 N（≤512）且 T>1 时占优
          （192x192@T=8 仅 0.74x）；但实测 N>=1024 时即使 T=8 原生仍快
          4.6x~9.4x → decode（T=1）与大 N 的 prefill 走 native。
        两条路径在 int32 域逐位相等（不变量 ①），故该调度不改变数值结果。
        """
        if self._kern is None:
            return False
        if self.backend == "tl":
            return True
        return T == 1 or (N is not None and N >= 1024)

    def _proj_native(self, x, wname, W, w_scale, xq, x_scale):
        """原生内核投影：与 numpy 路径在 int32 域逐位相等（不变量 ①）。

        i2s 走预解包 int8（P0 主路径，实测 3x~15x）；tl 走 5-bit 打包流
        （P1，内存受限场景）。逐 token 调用（GEMV 形态）。
        """
        k = self._kern
        N, K = W.shape
        T = x.shape[0]
        out = np.empty((T, N), dtype=np.float32)
        if self.backend == "i2s":
            rw = self._row_wsum_of(wname)
            for t in range(T):
                xu8 = (xq[t].astype(np.int32) + 128).astype(np.uint8)
                out[t] = k.i2s_gemm_f32(W, xu8, rw, w_scale, float(x_scale[t]))
        else:  # tl
            packed = self._packed_of(wname)
            Kp = k_pad(K)
            xpad = np.zeros(Kp, dtype=np.int8)
            for t in range(T):
                xpad[:K] = xq[t]
                out[t] = k.tl_gemm_f32(packed, xpad, w_scale, float(x_scale[t]), N, Kp)
        return out

    def _proj(self, x: np.ndarray, name: str) -> np.ndarray:
        """三值线性投影 + 可选 HGF 校正。x: [T, in] -> [T, out]。

        name 为投影基名（如 `layers.0.q_proj`）；权重/scale/aux 实际键为
        `name + ".weight"`（与 loader 保持一致）。量化激活与否由 self.quant_act 决定。
        """
        wname = name + ".weight"
        W = self.w[wname].astype(np.int32)
        w_scale = self.scales.get(wname, np.ones(W.shape[0], dtype=np.float32))
        if self.backend == "fp32":
            y = _ternary_gemm_fp32(W, w_scale, x)
        elif self.backend == "int8":
            # int8 后端：权重先反量化为 fp32，再与激活各自 absmax 量化到 int8
            y = gemm_int8(W.astype(np.float32) * w_scale.reshape(-1, 1), x)
        else:
            xq, x_scale = _quant_per_token(x)
            if self._use_native(x.shape[0], W.shape[0]):
                y = self._proj_native(x, wname, W, w_scale, xq, x_scale)
            elif self.backend == "i2s":
                y = _ternary_gemm_i2s(W, w_scale, xq, x_scale)
            else:  # "tl"：真·LUT，直接吃打包流解出的 code5
                y = _ternary_gemm_tl(self._codes_of(wname), W.shape[1], w_scale, xq, x_scale)
        if wname in self.hgf:
            A, B, gate = self.hgf[wname]
            corr = (B @ (A @ x.T)).T  # [T, out]
            y = y + gate * (self.hgf_alpha / max(A.shape[0], 1)) * corr.astype(np.float32)
        return y

    # ----------------------------- 前向 ----------------------------- #
    def forward(self, tokens: np.ndarray, return_hidden: bool = False):
        """对 [T] token 做一次前向，更新 KV cache，返回 [T, V] logits。

        return_hidden=True 时额外返回各层残差流 hidden（debug 用），便于与 torch 对齐。
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        T = int(tokens.shape[0])
        # 单批超过 max_seq_len：只保留最近 max_seq_len 个 token（标准滑动窗口语义）。
        # 否则 RoPE 位置会超出预计算表（既存 bug：T > max_seq_len 时越界）。
        if self.max_seq_len < T:
            tokens = tokens[T - self.max_seq_len :]
            T = self.max_seq_len
        start = self.cache_len()
        # 超过最大「逻辑」长度（cache_len）时做滑动窗口截断——这是 RoPE 表边界
        # 守卫，必须用逻辑坐标，绝不能用绝对坐标（否则驱逐后仍会触发，破坏
        # layer_len / mgr.seq_len 一致性）。绝对坐标只用于取 RoPE 表。
        if start + T > self.max_seq_len:
            drop = start + T - self.max_seq_len
            if self.kv_mode == "paged":
                self.cache.drop_front(0, drop)
            else:
                for i in range(self.n_layer):
                    self.cache.k[i] = self.cache.k[i][:, :, drop:, :]
                    self.cache.v[i] = self.cache.v[i][:, drop:, :]
            self._abs_dropped += drop
            start = self.cache_len()
        # 绝对位置 = 已丢弃总数 + 当前缓存长度（对齐 RoPE，驱逐/截断后依然正确）
        abs_start = self._abs_dropped + start

        x = self.tok_emb[tokens].astype(np.float32)  # [T, d]
        self._hiddens = []
        self._dbg = {}
        cos, sin = self._rope_slice(abs_start, T)
        Tk = self.cache_len()

        for i in range(self.n_layer):
            p = f"layers.{i}."
            h = self._rmsnorm(x, self.w[p + "attn_norm"])
            q = self._proj(h, p + "q_proj")  # [T, d]
            k = self._proj(h, p + "k_proj")  # [T, dkv]
            v = self._proj(h, p + "v_proj")  # [T, dkv]

            # reshape -> [1, n_head, T, dh]；K/V 保持 n_kv_head（paged 省 n_rep 倍内存）
            q = q.reshape(1, T, self.n_head, self.d_head).transpose(0, 2, 1, 3)
            kkv = k.reshape(1, T, self.n_kv_head, self.d_head).transpose(0, 2, 1, 3)
            vkv = v.reshape(1, T, self.n_kv_head, self.d_head).transpose(0, 2, 1, 3)
            if i == 0 and self._dbg is not None:
                self._dbg["q_raw"] = q.copy()
                self._dbg["h_norm"] = h.copy()

            # RoPE（逐头独立 → 先旋转后展开与 dense 路径逐位相同）
            q = _apply_rope(q, cos, sin)
            kkv = _apply_rope(kkv, cos, sin)
            if i == 0 and self._dbg is not None:
                self._dbg["q_rope"] = q.copy()

            if self.kv_mode == "paged":
                # ---- 分页 KV：预留 -> 逐层写 -> gather + GQA 注意力（不变量 ⑨）----
                if i == 0 and not self.cache.begin_append(0, T):
                    raise RuntimeError(
                        f"分页 KV 容量不足：需要追加 {T} token 但池已满"
                        f"（max_seq_len={self.max_seq_len}, block_size={self.block_size}）"
                    )
                self.cache.write(0, i, kkv[0].transpose(1, 0, 2), vkv[0].transpose(1, 0, 2))
                # q_start = 本批 query 在 KV 序列中的逻辑起点 = 写入前缓存长度（start）。
                # 注意：绝不能取 layer_len[i]（write 已 +T，会错位 T 并破坏因果掩码）；
                # 各层 query 起点相同，逐层差异只在「key 数量」（已由 gather 按 layer_len 读取）。
                q_start = int(start)
                out = paged_gqa_attention(q, self.cache, 0, i, self.n_rep, q_start)
                if self._evict is not None:
                    self._score_layer(i, q)
            else:
                k = np.repeat(kkv, self.n_rep, axis=1)
                v = np.repeat(vkv, self.n_rep, axis=1)
                # 拼接 cache
                k_full = np.concatenate([self.cache.k[i], k], axis=2)
                v_full = np.concatenate([self.cache.v[i], v], axis=2)
                self.cache.k[i] = k_full
                self.cache.v[i] = v_full

                # 注意力
                scores = (q @ k_full.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
                if T > 1:
                    mask = np.tril(np.ones((T, Tk + T), dtype=bool), k=Tk)
                    scores = np.where(mask[None, None, :, :], scores, -np.inf)
                attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn = attn / np.sum(attn, axis=-1, keepdims=True)
                out = attn @ v_full  # [1, H, T, dh]
            out = out.transpose(0, 2, 1, 3).reshape(T, self.d)
            if i == 0 and self._dbg is not None:
                self._dbg["attn_out"] = out.copy()

            o = self._proj(out, p + "o_proj")
            x = x + self.w[p + "subln_attn"].astype(np.float32) * o

            # FFN
            h2 = self._rmsnorm(x, self.w[p + "ffn_norm"])
            if self.act_relu2:
                g = np.clip(self._proj(h2, p + "gate_proj"), 0, None) ** 2
                u = self._proj(h2, p + "up_proj")
                ffn = self._proj(g * u, p + "down_proj")
            else:
                g = self._proj(h2, p + "gate_proj")
                u = self._proj(h2, p + "up_proj")
                ff = g * (1.0 / (1.0 + np.exp(-u)))
                ffn = self._proj(ff, p + "down_proj")
            x = x + self.w[p + "subln_ffn"].astype(np.float32) * ffn
            self._hiddens.append(x.astype(np.float32).copy())

        if self.kv_mode == "paged":
            self.cache.end_append(0)
            # T4-C：达到高水位则按策略压缩 KV（逐层预算 + 逐层保留掩码）
            if self._evict is not None and self.cache_len() > self._evict_threshold:
                self._run_eviction()

        x = x * self.norm_final
        emb = self.tok_emb if self.tied else self.lm_head
        logits = x @ emb.T  # [T, V]
        if return_hidden:
            return logits.astype(np.float32), self._hiddens
        return logits.astype(np.float32)

    # ----------------------------- 生成 ----------------------------- #
    def cache_len(self) -> int:
        """当前 KV cache 中的 token 数（dense 与 paged 统一接口）。"""
        return self.cache.mgr.seq_len(0) if self.kv_mode == "paged" else self.cache.len()

    def _rope_slice(self, start: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        """取 RoPE 表 [start:start+T]，必要时惰性扩容（绝对位置可超过 max_seq_len）。"""
        need = start + T
        if need > self._cos.shape[0]:
            new_len = max(need, 2 * self._cos.shape[0])
            self._cos, self._sin = _build_rope(self.rope_theta, self.d_head, new_len)
        return self._cos[start : start + T], self._sin[start : start + T]

    # ----------------------------- T4-C 驱逐 ----------------------------- #
    def _score_layer(self, i: int, q: np.ndarray) -> None:
        """为第 i 层计算驱逐分数（SnapKV 窗口打分 / H2O 跨步累积）。

        q: [1, H, T, Dh]（RoPE 已应用）。分数对应「窗口之外全部历史 token」
        （即前 `n_past = layer_len - sink - window` 个），与 `EvictionPolicy.select`
        的候选段切分严格对齐。
        """
        pol = self._evict
        n = int(self.cache.layer_len[i])
        n_past = pol.n_past_of(n)
        W = pol.cfg.window_size
        if n_past <= 0:
            self._win_scores[i] = np.zeros((self.n_head, 0), dtype=np.float32)
            return
        k, _ = self.cache.gather_heads(0, i, self.n_rep)  # [1, H, n, Dh]
        kp = k[:, :, :n_past, :]
        if pol.name in ("snapkv", "pyramidkv"):
            w = min(W, q.shape[2])
            self._win_scores[i] = kv_importance_scores(q[:, :, q.shape[2] - w :, :], kp)
        elif pol.name == "h2o":
            aw = attention_weights(q, k, n - q.shape[2], causal=True)
            cur = aw[:, :, :, :n_past].sum(axis=2)[0]  # [H, n_past]，本步对各历史 token 的注意力
            # 跨步累积（位置对齐）：保留段累加本步注意力、新段（窗口滑出的 token）
            # 从本步起算。压缩后 _h2o 已按 keep_mask 聚到保留 token，r = 保留数 ≤ n_past。
            r = self._h2o[i].shape[1]
            new = np.zeros((self.n_head, n_past), dtype=np.float32)
            if r > 0:
                new[:, :r] = self._h2o[i] + cur[:, :r]
            new[:, r:] = cur[:, r:]
            self._h2o[i] = new
        # streamingllm：不依赖分数（只保留 sink + 最近窗口），无需打分

    def _run_eviction(self) -> None:
        """按策略逐层选 token 并压缩 KV；更新 `_abs_dropped` 与 RoPE 基准。"""
        pol = self._evict
        n_total = self.cache_len()
        if n_total <= 0:
            return
        masks = np.zeros((self.n_layer, n_total), dtype=bool)
        for i in range(self.n_layer):
            kl = int(self.cache.layer_len[i])
            if kl <= 0:
                continue
            if pol.name in ("snapkv", "pyramidkv"):
                scores = self._win_scores[i]
            else:
                scores = self._h2o[i]
            budget = pol.layer_budget(i, self.n_layer, n_total)
            masks[i, :kl] = pol.select(scores, budget, kl)
        if pol.name == "h2o":
            for i in range(self.n_layer):
                kl = int(self.cache.layer_len[i])
                sink = min(pol.cfg.sink_size, kl)
                scored_hi = kl - min(pol.cfg.window_size, max(0, kl - sink))
                n_past = scored_hi - sink
                if n_past > 0 and self._h2o[i].shape[1] == n_past:
                    self._h2o[i] = np.ascontiguousarray(self._h2o[i][:, masks[i, sink:scored_hi]])
        before = n_total
        self.cache.compact(0, masks)
        self._abs_dropped += before - self.cache_len()

    def reset(self) -> None:
        self.cache.reset()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.95,
        seed: int = 0,
        stream: bool = False,
    ) -> str:
        if self.tok is None:
            raise RuntimeError("该 .phos 不含 tokenizer，无法做文本生成（请用 token id 接口）")
        rng = np.random.default_rng(seed)
        ctx = self.tok.encode(prompt)
        if not ctx:
            # 特殊 token（如 <|bos|>）的 id 可能 >= vocab（词表外），嵌入表无对应行；
            # 训练语料也未前置特殊 token，故回退到 id 0（与训练行为一致）。
            ctx = (
                [self.bos_id]
                if (self.bos_id is not None and 0 <= self.bos_id < self.vocab)
                else [0]
            )
        self.reset()
        logits = self.forward(np.array(ctx, dtype=np.int64))
        next_id = _sample(logits[-1], temperature, top_k, top_p, rng)
        gen: list[int] = [next_id]
        if stream:
            print(self.tok.decode([next_id]), end="", flush=True)
        for _ in range(max_new_tokens - 1):
            logits = self.forward(np.array([next_id], dtype=np.int64))
            next_id = _sample(logits[-1], temperature, top_k, top_p, rng)
            gen.append(next_id)
            if stream:
                print(self.tok.decode([next_id]), end="", flush=True)
            if self.eos_id is not None and next_id == self.eos_id:
                break
        if stream:
            print()
        return self.tok.decode(ctx + gen)
