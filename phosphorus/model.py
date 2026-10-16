"""LLaMA 变体模型（PyTorch，训练侧；lazy import torch）。

借鉴来源：
    * LLaMA / llama2.c 架构（Pre-RMSNorm, RoPE, SwiGLU/ReLU², GQA, 无 bias）— A-16
    * BitNet b1.58（absmean 三值, 8-bit 激活, SubLN, ReLU²）Wang et al. 2024/2025 — A-1
    * HGF 低秩 FP 校正（Pizzo, 2026）— A-11
    * LayerSkip 自草稿 / 早退（arXiv:2404.16710）— A-15（仅暴露早退头，草稿复用目标前层）

设计：本模块是训练入口；导出后由 `engine.py`（纯 numpy，零 torch）推理。
权重命名 / 形状与 ARCHITECTURE.md §3.5 严格一致，便于 `.phos` 互操作。
每层用 `Layer(nn.Module)` 承载子模块与 SubLN 标量增益（nn.ModuleDict 不接受
nn.Parameter，故用属性访问）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, QuantConfig

# --------------------------------------------------------------------------- #
# 量化基元                                                                     #
# --------------------------------------------------------------------------- #


def absmean(w: torch.Tensor) -> torch.Tensor:
    """BitNet absmean 缩放（per-channel，按输出行）。返回 scale [out, 1]。

    为什么是 absmean 而不是 absmax（BitNet b1.58 §3.1）：absmean 对离群值鲁棒，
    且无需校准集。scale = mean(|W|) 后 round(W/scale) 的值域天然落在 [-1,1]。

    注：`.phos` 的 scale 布局按 per-channel 设计（每行一个 float32）；
    per_tensor / per_block 粒度尚未落地（会改变 .phos 的 scale 张量形状，
    需要格式版本升级）。故此处不暴露未实现的粒度选项——不做「幽灵参数」。
    """
    return w.abs().mean(dim=-1, keepdim=True)


def ternary_ste(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """三值量化 + 直通估计（STE）。

    forward: w_hat = clamp(round(w/scale), -1, 1) * scale  （值 ∈ {-1,0,1}*scale）
             这是「真·三值权重」，须与 export_numpy / engine 的部署权重逐元素一致，
             否则训练侧前向与推理侧不一致（不变量 ⑦ 失败）。
    backward: 直通估计——round/clamp 不可导，故用 (w - w.detach()) 让梯度恒等流过
             全精度权重（BitNet QAT 标准做法），学习到干净映射到 {-1,0,1}。
    不变量 ⑥：数值梯度应与本函数解析梯度一致（见 test_qat_ste.py）。
    """
    w_scaled = w / scale
    w_tern = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
    # forward = w_tern * scale；(w - w.detach()) 仅提供反向恒等梯度（前向抵消为 0）。
    return w_tern * scale + (w - w.detach())


class BitLinear(nn.Module):
    """线性层 + 可选三值 QAT + 可选 HGF 低秩校正。

    QAT 模式下：权重被量化到 {-1,0,1}*scale（STE），激活在推理侧走 int8。
    HGF：前向加 `gate * (alpha/r) * B @ (A @ x)`（可学习低秩 FP 校正）。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        qat: bool = False,
        hgf_rank: int = 0,
        hgf_alpha: float = 1.0,
        hgf_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qat = qat
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.hgf_rank = hgf_rank
        self.hgf_alpha = hgf_alpha
        if hgf_rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(hgf_rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, hgf_rank))
            nn.init.normal_(self.lora_a, std=0.01)
            nn.init.zeros_(self.lora_b)
            self.gate = nn.Parameter(torch.full((1,), float(hgf_gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qat:
            with torch.no_grad():
                s = absmean(self.weight)
            w = ternary_ste(self.weight, s)
        else:
            w = self.weight
        y = F.linear(x, w)
        if self.hgf_rank > 0:
            corr = self.lora_b @ (self.lora_a @ x.transpose(-1, -2))  # [out, T]
            y = y + self.gate * (self.hgf_alpha / max(self.hgf_rank, 1)) * corr.transpose(-1, -2)
        return y


# --------------------------------------------------------------------------- #
# 归一化 / 位置编码                                                            #
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def build_rope(theta: float, d_head: int, max_len: int, device):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    t = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)  # [T, d_head/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """q: [B, H, T, dh]。Rotary 位置编码（相邻两两配对，Su et al. 2021）。不变量 ④。"""
    q1 = q[..., 0::2]
    q2 = q[..., 1::2]
    T = cos.shape[0]
    c = cos.reshape(1, 1, T, -1)
    s = sin.reshape(1, 1, T, -1)
    out = torch.empty_like(q)
    out[..., 0::2] = q1 * c - q2 * s
    out[..., 1::2] = q2 * c + q1 * s
    return out


# --------------------------------------------------------------------------- #
# 单层                                                                         #
# --------------------------------------------------------------------------- #


class Layer(nn.Module):
    """一个 Transformer 块（含 SubLN 标量增益）。"""

    def __init__(self, mcfg: ModelConfig, qcfg: QuantConfig) -> None:
        super().__init__()
        d = mcfg.d_model
        dkv = mcfg.d_kv
        dff = mcfg.d_ff
        qat = qcfg.qat
        r = qcfg.hgf_rank if qcfg.hgf_rank > 0 else 0
        self.attn_norm = RMSNorm(d, mcfg.norm_eps)
        self.ffn_norm = RMSNorm(d, mcfg.norm_eps)
        self.q_proj = BitLinear(
            d, d, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha, hgf_gate_init=qcfg.hgf_gate_init
        )
        self.k_proj = BitLinear(d, dkv, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.v_proj = BitLinear(d, dkv, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.o_proj = BitLinear(d, d, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.gate_proj = BitLinear(d, dff, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.up_proj = BitLinear(d, dff, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.down_proj = BitLinear(dff, d, qat=qat, hgf_rank=r, hgf_alpha=qcfg.hgf_alpha)
        self.subln_attn = nn.Parameter(torch.ones(d))
        self.subln_ffn = nn.Parameter(torch.ones(d))


# --------------------------------------------------------------------------- #
# 模型                                                                        #
# --------------------------------------------------------------------------- #


class LlamaModel(nn.Module):
    def __init__(self, mcfg: ModelConfig, qcfg: QuantConfig):
        super().__init__()
        self.m = mcfg
        self.q = qcfg
        d = mcfg.d_model
        self.tok_emb = nn.Parameter(torch.zeros(mcfg.vocab_size, d))
        nn.init.normal_(self.tok_emb, std=0.02)
        self.layers = nn.ModuleList([Layer(mcfg, qcfg) for _ in range(mcfg.n_layer)])
        self.norm_final = nn.Parameter(torch.ones(d))
        if not mcfg.tied_embed:
            self.lm_head = nn.Parameter(torch.zeros(mcfg.vocab_size, d))
        self.early_exit_layer = mcfg.early_exit_layer
        if mcfg.early_head_rank > 0:
            r = mcfg.early_head_rank
            self.early_lora_a = nn.Parameter(torch.zeros(r, d))
            self.early_lora_b = nn.Parameter(torch.zeros(mcfg.vocab_size, r))
            nn.init.normal_(self.early_lora_a, std=0.01)
            nn.init.zeros_(self.early_lora_b)
            self.early_gate = nn.Parameter(torch.full((1,), 0.05))

    # ----------------------------- 前向 ----------------------------- #
    def forward(self, idx: torch.Tensor, cos=None, sin=None, return_logits: bool = True):
        B, T = idx.shape
        d = self.m.d_model
        dev = idx.device
        if cos is None:
            cos, sin = build_rope(self.m.rope_theta, self.m.d_head, T, dev)
        cos, sin = cos[:T].to(dev), sin[:T].to(dev)
        x = self.tok_emb[idx]  # [B,T,d]
        early_logits_list = []
        for i, ly in enumerate(self.layers):
            h = ly.attn_norm(x)
            q = ly.q_proj(h)
            k = ly.k_proj(h)
            v = ly.v_proj(h)
            n_rep = self.m.n_head // self.m.n_kv_head
            k = k.view(B, T, self.m.n_kv_head, self.m.d_head).transpose(1, 2)
            v = v.view(B, T, self.m.n_kv_head, self.m.d_head).transpose(1, 2)
            q = q.view(B, T, self.m.n_head, self.m.d_head).transpose(1, 2)
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.m.d_head)
            mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v  # [B,H,T,dh]
            out = out.transpose(1, 2).reshape(B, T, d)
            o = ly.o_proj(out)
            x = x + ly.subln_attn * o
            h2 = ly.ffn_norm(x)
            if self.m.act == self.m.act.RELU2:
                g = torch.relu(ly.gate_proj(h2)) ** 2
                u = ly.up_proj(h2)
                ffn = ly.down_proj(g * u)
            else:
                g = ly.gate_proj(h2)
                u = ly.up_proj(h2)
                ff = F.silu(g) * u
                ffn = ly.down_proj(ff)
            x = x + ly.subln_ffn * ffn
            if self.m.early_head_rank > 0 and i == self.early_exit_layer:
                e = self.early_gate * (
                    self.early_lora_b @ (self.early_lora_a @ x.transpose(-1, -2))
                )
                early_logits_list.append(e.transpose(-1, -2))
        x = x * self.norm_final
        if return_logits:
            emb = self.tok_emb if self.m.tied_embed else self.lm_head
            logits = x @ emb.t()  # [B,T,V]
            if early_logits_list:
                logits = logits + early_logits_list[0]
            return logits
        return x

    # ----------------------------- 导出 ----------------------------- #
    def export_numpy(self) -> dict[str, np.ndarray]:
        """返回纯 numpy 权重字典（engine 推理用）。三值权重已量化。"""
        m = self.m
        out: dict[str, np.ndarray] = {}
        out["tok_emb"] = self.tok_emb.detach().cpu().numpy().astype(np.float32)
        out["norm_final"] = self.norm_final.detach().cpu().numpy().astype(np.float32)
        if not m.tied_embed:
            out["lm_head"] = self.lm_head.detach().cpu().numpy().astype(np.float32)
        if m.early_head_rank > 0:
            out["early_lora_a"] = self.early_lora_a.detach().cpu().numpy().astype(np.float32)
            out["early_lora_b"] = self.early_lora_b.detach().cpu().numpy().astype(np.float32)
            out["early_gate"] = float(self.early_gate.detach().cpu())
        for i, ly in enumerate(self.layers):
            for name in ("attn_norm", "ffn_norm"):
                out[f"layers.{i}.{name}"] = (
                    getattr(ly, name).weight.detach().cpu().numpy().astype(np.float32)
                )
            for name in ("subln_attn", "subln_ffn"):
                out[f"layers.{i}.{name}"] = (
                    getattr(ly, name).detach().cpu().numpy().astype(np.float32)
                )
            for proj in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                blk: BitLinear = getattr(ly, proj)
                if not blk.qat:
                    # fp32 检查点无法导出：.phos 的张量 dtype 是 TERNARY_PACKED，
                    # 把 fp32 权重 astype(int8) 会静默截断（丢掉全部小数位），
                    # 属于「看起来成功、实则毁掉模型」的陷阱，故显式报错。
                    raise ValueError(
                        f"layers.{i}.{proj} 未走 QAT（qat=False），无法导出 .phos；"
                        "引擎只支持三值权重。请改用 --mode qat-hgf 训练。"
                    )
                w = blk.weight.detach().cpu().numpy().astype(np.float32)
                s = (
                    absmean(blk.weight)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                ternary = np.clip(np.round(w / s.reshape(-1, 1)), -1, 1).astype(np.int8)
                out[f"layers.{i}.{proj}.weight"] = ternary
                out[f"layers.{i}.{proj}.scale"] = s
                if blk.hgf_rank > 0:
                    out[f"layers.{i}.{proj}.hgf_a"] = (
                        blk.lora_a.detach().cpu().numpy().astype(np.float32)
                    )
                    out[f"layers.{i}.{proj}.hgf_b"] = (
                        blk.lora_b.detach().cpu().numpy().astype(np.float32)
                    )
                    out[f"layers.{i}.{proj}.hgf_gate"] = float(blk.gate.detach().cpu())
        return out
