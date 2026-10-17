"""Phosphorus 忠实 HF 模型（T5-C8 增强）：逐位保真于原生 Phosphorus 引擎。

为什么需要本文件
----------------
标准 `LlamaForCausalLM` 省略了 BitNet 的两个关键结构：

1. **SubLN 标量增益**：Phosphorus 在注意力输出与 FFN 输出、加入残差之前，
   各乘一个逐通道标量增益 `subln_attn` / `subln_ffn`
   （`x = x + subln_attn * o_proj(attn)`；`x = x + subln_ffn * mlp`）。
   这是数据相关的，无法折叠进静态权重。
2. **ReLU² 激活**：FFN 用 `relu(gate)²` 而非 HF Llama 的 SiLU（SwiGLU）。

标准 Llama 导出只能做到「架构近似等价」。本文件复刻
`phosphorus/model.py::LlamaModel.forward` 的**逐位前向**（Pre-RMSNorm + RoPE +
GQA + 标量 SubLN 增益 + ReLU² FFN + norm_final），使
`transformers.AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)`
得到的 logits 与 `phosphorus generate`（原生引擎）**逐位一致**（不变量 ⑦）。

兼容性（集大成：复用而非重写）
---------------------------------
* 装有 `transformers` 时：导出 `PhosphorusForCausalLM(PreTrainedModel)` 与
  `PhosphorusConfig(PretrainedConfig)`，`from_pretrained` 经 `config.json` 的
  `auto_map` 自动加载本文件。
* 未装 `transformers` 时：回退为纯 torch `nn.Module`（同名类、同前向），
  供仓库内测试在不依赖 `transformers` 的环境验证逐位等价性（不引入幽灵依赖）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PretrainedConfig, PreTrainedModel

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - 仅在无 transformers 的测试环境走此分支
    PretrainedConfig = object  # type: ignore[assignment, misc]
    PreTrainedModel = nn.Module  # type: ignore[assignment, misc]
    _HAS_TRANSFORMERS = False


# --------------------------------------------------------------------------- #
# 复刻的归一化 / 位置编码（与 model.py 逐位一致）                              #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def _build_rope(theta: float, d_head: int, max_len: int, device):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    t = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
# 层与前向（与 model.py:201-254 逐位一致）；被两个外壳共享                    #
# --------------------------------------------------------------------------- #
class _PhosphorusBase:
    def _build(self, cfg) -> None:
        d = int(cfg.hidden_size)
        self.d = d
        self.m = cfg
        self.act = getattr(cfg, "act", "relu2")
        self.tok_emb = nn.Parameter(torch.zeros(int(cfg.vocab_size), d))
        self.layers = nn.ModuleList()
        d_head = d // int(cfg.num_attention_heads)
        d_kv = int(cfg.num_key_value_heads) * d_head
        for _ in range(int(cfg.num_hidden_layers)):
            ly = nn.Module()
            ly.attn_norm = RMSNorm(d, float(cfg.rms_norm_eps))
            ly.ffn_norm = RMSNorm(d, float(cfg.rms_norm_eps))
            ly.q_proj = nn.Linear(d, d, bias=False)
            ly.k_proj = nn.Linear(d, d_kv, bias=False)
            ly.v_proj = nn.Linear(d, d_kv, bias=False)
            ly.o_proj = nn.Linear(d, d, bias=False)
            ly.gate_proj = nn.Linear(d, int(cfg.intermediate_size), bias=False)
            ly.up_proj = nn.Linear(d, int(cfg.intermediate_size), bias=False)
            ly.down_proj = nn.Linear(int(cfg.intermediate_size), d, bias=False)
            ly.subln_attn = nn.Parameter(torch.ones(d))
            ly.subln_ffn = nn.Parameter(torch.ones(d))
            self.layers.append(ly)
        self.norm_final = nn.Parameter(torch.ones(d))
        if not getattr(cfg, "tie_word_embeddings", True):
            self.lm_head = nn.Parameter(torch.zeros(int(cfg.vocab_size), d))

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        cfg = self.m
        B, T = input_ids.shape
        d = self.d
        dev = input_ids.device
        d_head = d // int(cfg.num_attention_heads)
        n_rep = int(cfg.num_attention_heads) // int(cfg.num_key_value_heads)
        cos, sin = _build_rope(float(cfg.rope_theta), d_head, T, dev)
        cos, sin = cos[:T].to(dev), sin[:T].to(dev)
        x = self.tok_emb[input_ids]
        for ly in self.layers:
            h = ly.attn_norm(x)
            q = ly.q_proj(h).view(B, T, int(cfg.num_attention_heads), d_head).transpose(1, 2)
            k = ly.k_proj(h).view(B, T, int(cfg.num_key_value_heads), d_head).transpose(1, 2)
            v = ly.v_proj(h).view(B, T, int(cfg.num_key_value_heads), d_head).transpose(1, 2)
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
            mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, T, d)
            o = ly.o_proj(out)
            x = x + ly.subln_attn * o  # SubLN 增益（注意力分支）
            h2 = ly.ffn_norm(x)
            if self.act == "relu2":
                g = torch.relu(ly.gate_proj(h2)) ** 2
                u = ly.up_proj(h2)
                ffn = ly.down_proj(g * u)
            else:
                g = ly.gate_proj(h2)
                u = ly.up_proj(h2)
                ffn = ly.down_proj(F.silu(g) * u)
            x = x + ly.subln_ffn * ffn  # SubLN 增益（FFN 分支）
        x = x * self.norm_final
        if return_logits:
            emb = self.tok_emb if getattr(cfg, "tie_word_embeddings", True) else self.lm_head
            return x @ emb.t()
        return x


# --------------------------------------------------------------------------- #
# 双外壳：有 transformers 用 PreTrainedModel（可被 from_pretrained 加载）；
# 否则用纯 nn.Module（同前向，供本地验证）。                                  #
# --------------------------------------------------------------------------- #
if _HAS_TRANSFORMERS:

    class PhosphorusConfig(PretrainedConfig):
        model_type = "phosphorus"

        def __init__(
            self,
            vocab_size: int = 4096,
            hidden_size: int = 192,
            num_hidden_layers: int = 3,
            num_attention_heads: int = 3,
            num_key_value_heads: int = 1,
            intermediate_size: int = 384,
            max_position_embeddings: int = 256,
            rms_norm_eps: float = 1e-5,
            rope_theta: float = 10000.0,
            act: str = "relu2",
            hidden_act: str = "relu2",
            tie_word_embeddings: bool = True,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            **kwargs,
        ) -> None:
            super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.intermediate_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.act = act
            self.hidden_act = hidden_act
            self.tie_word_embeddings = tie_word_embeddings

    class PhosphorusForCausalLM(_PhosphorusBase, PreTrainedModel):
        config_class = PhosphorusConfig

        def __init__(self, config: PhosphorusConfig) -> None:
            super().__init__(config)
            self._build(config)

else:

    class PhosphorusForCausalLM(_PhosphorusBase, nn.Module):  # type: ignore[no-redef]
        def __init__(self, config) -> None:
            super().__init__()
            self._build(config)
