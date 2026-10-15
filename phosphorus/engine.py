"""纯 numpy 推理引擎（运行时零 torch 依赖，ARCHITECTURE.md §4）。

加载 `.phos` -> 复现 LLaMA 变体的前向（Pre-RMSNorm / RoPE / GQA / ReLU² / SubLN），
使用三值 TL GEMM（W_ternary @ xq，等价 int32 参考）与可选 HGF 低秩校正，
支持 KV cache 的 prefill / decode 与 temperature / top-k / top-p 采样。

借鉴来源：
    * llama2.c 的 Llama.run / kv cache 设计（Karpathy）— A-16
    * BitNet 三值 GEMM（见 gemm.py，TL 路径数值权威）— A-1/A-3
    * RoPE 旋转编码（Su et al., 2021）— A-16
    * HGF 低秩 FP 校正（Pizzo, 2026）— A-11

不变量（详见 docs/ARCHITECTURE.md §7）：
    ① engine 的三值 GEMM 与 gemm.int32_reference 在 int32 域逐位一致
    ③ pack/unpack 往返逐位恒等（由 loader+pkg 保证）
    ④ RoPE 与 model.apply_rope 数值一致（同一公式）
    ⑤ encode->decode 恒等（tokenizer）
    ⑦ 同一输入 engine 与 torch 模型 logits 相对误差 < 5%
"""

from __future__ import annotations

import math

import numpy as np

from .loader import PhosFile, load_phos
from .sampling import sample as _sample


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


def _ternary_gemm_int8(
    W: np.ndarray, w_scale: np.ndarray, xq: np.ndarray, x_scale: np.ndarray
) -> np.ndarray:
    """真·BitNet 路径：out[T,out] = (W_int8 @ xq_int8.T) * w_scale * x_scale。

    int8 激活量化是部署态的近似（量化误差来源，对应风险 R6 的 PPL 退化）。
    """
    acc = W.astype(np.int32) @ xq.astype(np.int32).T  # [out, T]
    out = (
        acc.astype(np.float64)
        * w_scale.astype(np.float64)[:, None]
        * x_scale.astype(np.float64)[None, :]
    )
    return out.astype(np.float32).T  # [T, out]


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

    def __init__(self, phos: PhosFile, quant_act: bool = False) -> None:
        self.phos = phos
        self.quant_act = quant_act  # False=忠实 fp32 激活（匹配 torch）；True=真·BitNet int8 激活
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
        self.cache = _KVCache(self.n_layer, self.n_head, self.d_head)

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

    # ----------------------------- 工厂 ----------------------------- #
    @classmethod
    def from_file(cls, path) -> PhosModel:
        return cls(load_phos(path))

    # ----------------------------- 工具 ----------------------------- #
    def _rmsnorm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        return (x * (1.0 / np.sqrt(var + self.norm_eps)) * w).astype(np.float32)

    def _proj(self, x: np.ndarray, name: str) -> np.ndarray:
        """三值线性投影 + 可选 HGF 校正。x: [T, in] -> [T, out]。

        name 为投影基名（如 `layers.0.q_proj`）；权重/scale/aux 实际键为
        `name + ".weight"`（与 loader 保持一致）。量化激活与否由 self.quant_act 决定。
        """
        wname = name + ".weight"
        W = self.w[wname].astype(np.int32)
        w_scale = self.scales.get(wname, np.ones(W.shape[0], dtype=np.float32))
        if self.quant_act:
            xq, x_scale = _quant_per_token(x)
            y = _ternary_gemm_int8(W, w_scale, xq, x_scale)
        else:
            y = _ternary_gemm_fp32(W, w_scale, x)
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
        start = self.cache.len()
        # 超过最大长度时做滑动窗口截断（简易 SnapKV 风格）
        if start + T > self.max_seq_len:
            drop = start + T - self.max_seq_len
            for i in range(self.n_layer):
                self.cache.k[i] = self.cache.k[i][:, :, drop:, :]
                self.cache.v[i] = self.cache.v[i][:, :, drop:, :]
            start = self.cache.len()

        x = self.tok_emb[tokens].astype(np.float32)  # [T, d]
        self._hiddens = []
        self._dbg = {}
        cos = self._cos[start : start + T]
        sin = self._sin[start : start + T]
        Tk = self.cache.len()

        for i in range(self.n_layer):
            p = f"layers.{i}."
            h = self._rmsnorm(x, self.w[p + "attn_norm"])
            q = self._proj(h, p + "q_proj")  # [T, d]
            k = self._proj(h, p + "k_proj")  # [T, dkv]
            v = self._proj(h, p + "v_proj")  # [T, dkv]

            # reshape -> [1, n_head, T, dh]（k/v 经 GQA 重复）
            q = q.reshape(1, T, self.n_head, self.d_head).transpose(0, 2, 1, 3)
            kkv = k.reshape(1, T, self.n_kv_head, self.d_head).transpose(0, 2, 1, 3)
            vkv = v.reshape(1, T, self.n_kv_head, self.d_head).transpose(0, 2, 1, 3)
            k = np.repeat(kkv, self.n_rep, axis=1)
            v = np.repeat(vkv, self.n_rep, axis=1)
            if i == 0 and self._dbg is not None:
                self._dbg["q_raw"] = q.copy()
                self._dbg["h_norm"] = h.copy()

            # RoPE
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            if i == 0 and self._dbg is not None:
                self._dbg["q_rope"] = q.copy()
                self._dbg["k_rope"] = k.copy()

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

        x = x * self.norm_final
        emb = self.tok_emb if self.tied else self.lm_head
        logits = x @ emb.T  # [T, V]
        if return_hidden:
            return logits.astype(np.float32), self._hiddens
        return logits.astype(np.float32)

    # ----------------------------- 生成 ----------------------------- #
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
            ctx = [self.bos_id] if self.bos_id is not None else [0]
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
