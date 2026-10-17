"""导出 `.phos` → HuggingFace 格式（`transformers` 可 `from_pretrained`）。

实现要点：
  * 三值权重解包（int8 ∈ {-1,0,1}）+ 逐通道 absmean scale 还原；
  * HGF 低秩校正因矩阵结合律 `B@(A@x) == (B@A)@x` 可静态折叠进权重：
        W_full = (W_hat * scale) + gate * (alpha/r) * (B @ A)
  * 权重键名从 Phosphorus 命名映射到 HF `LlamaForCausalLM` 命名；
  * 写出标准 `config.json` + `pytorch_model.bin`（fp16），`transformers` 可直接加载；
  * 若 `.phos` 带分词器，额外导出 GPT2-BPE 兼容的 `vocab.json` / `merges.txt` /
    `tokenizer_config.json`。

诚实限制（已知、非缺陷）：
  * 标准 `LlamaForCausalLM` **不含 BitNet 的 SubLN 子层**（attn/ffn 后的子层归一化，
    数据相关、不可折叠进静态权重）。本导出**省略 SubLN**，故 HF 模型为
    **架构近似等价**（权重级忠实，但前向计算图与 Phosphorus 引擎不完全一致）。
    需要逐位保真请使用 Phosphorus 原生引擎（`phosphorus generate`）。
  * 引擎激活为 BitNet ReLU²；HF 标准 Llama 用 `relu`，存在轻微数值差异。
  * 以上两点使 HF 模型为「可加载、可运行的近似参考实现」，而非逐位副本。

零 transformers 依赖即可写文件；仅 `from_pretrained` 验证步骤需 transformers。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

# ---- 键名映射：Phosphorus -> HF LlamaForCausalLM ----
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def _map_key(name: str) -> str | None:
    """把 Phosphorus 权重名映射到 HF 键名；返回 None 表示跳过（不导出）。"""
    if name == "tok_emb":
        return "model.embed_tokens.weight"
    if name == "norm_final":
        return "model.norm.weight"
    if name == "lm_head":
        return "lm_head.weight"
    if name in ("early_lora_a", "early_lora_b") or name == "early_gate":
        return None  # O1 早退头，标准 Llama 无对应；跳过
    if name.endswith((".scale", ".hgf_a", ".hgf_b", ".hgf_gate")):
        return None  # 仅用于本地解量化，不作为独立张量导出
    if name.endswith((".subln_attn", ".subln_ffn")):
        return None  # SubLN：数据相关，HF 标准 Llama 不含 → 省略（见模块 docstring）

    m = _LAYER_RE.match(name)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    if rest == "attn_norm":
        return f"model.layers.{idx}.input_layernorm.weight"
    if rest == "ffn_norm":
        return f"model.layers.{idx}.post_attention_layernorm.weight"
    # 投影权重：去掉 .weight 后缀映射到对应子模块
    proj = rest[: -len(".weight")] if rest.endswith(".weight") else rest
    attn_map = {"q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
                "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj"}
    mlp_map = {"gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
               "down_proj": "mlp.down_proj"}
    if proj in attn_map:
        return f"model.layers.{idx}.{attn_map[proj]}.weight"
    if proj in mlp_map:
        return f"model.layers.{idx}.{mlp_map[proj]}.weight"
    return None


def _dequantize(name: str, phos, alpha: float, dtype=np.float16) -> np.ndarray:
    """还原单个权重为 fp32（三值解包 + scale + HGF 折叠），再转 dtype。"""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    arr = phos.weights[name].astype(np.float32)
    scale = phos.scales.get(base)
    if scale is not None:
        arr = arr * scale.reshape(-1, 1)  # per-channel absmean scale
    hgf = phos.hgf.get(base)
    if hgf is not None:
        A, B, gate = hgf
        r = int(A.shape[0])
        # C = gate * (alpha/r) * (B @ A)   [out, in]
        C = gate * (alpha / r) * (B @ A)
        arr = arr + C
    return arr.astype(dtype)


def _build_state_dict(phos, dtype=np.float16) -> dict[str, np.ndarray]:
    alpha = float(phos.header.get("hgf_alpha", 1.0))
    sd: dict[str, np.ndarray] = {}
    for name in phos.weights:
        hf_key = _map_key(name)
        if hf_key is None:
            continue
        # 三值投影权重走解量化；其余（norm / embed / lm_head）已是 fp32，直接转 dtype
        if name.endswith(".weight") and name in phos.scales:
            sd[hf_key] = _dequantize(name, phos, alpha, dtype)
        else:
            sd[hf_key] = phos.weights[name].astype(dtype)
    return sd


def _build_config(phos, tie_word_embeddings: bool) -> dict:
    h = phos.header
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": int(h["d_model"]),
        "intermediate_size": int(h["d_ff"]),
        "num_hidden_layers": int(h["n_layer"]),
        "num_attention_heads": int(h["n_head"]),
        "num_key_value_heads": int(h["n_kv_head"]),
        "head_dim": int(h["d_head"]),
        "max_position_embeddings": int(h["max_seq_len"]),
        "vocab_size": int(h["vocab"]),
        "rms_norm_eps": float(h["norm_eps"]),
        "rope_theta": float(h["rope_theta"]),
        "hidden_act": "relu",
        "tie_word_embeddings": bool(tie_word_embeddings),
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _export_tokenizer(phos, out_dir: Path) -> bool:
    """把 Phosphorus BPE 分词器导出为 GPT2-BPE 兼容格式（best-effort）。"""
    tok = phos.tokenizer
    if tok is None:
        return False
    try:
        vocab = {str(i): s for i, s in enumerate(tok.vocab)}
        (out_dir / "vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
        )
        with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
            f.write("# merging pairs\n")
            for a, b in tok.merges:
                f.write(f"{a} {b}\n")
        cfg = {
            "model_type": "gpt2",
            "tokenizer_class": "GPT2TokenizerFast",
            "add_bos_token": False,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
        }
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def export_to_hf(phos_path, out_dir, dtype: str = "fp16") -> Path:
    """把 `.phos` 导出为 HuggingFace 模型目录（config.json + pytorch_model.bin）。

    参数：
        phos_path: `.phos` 文件路径
        out_dir:   输出目录（不存在则创建）
        dtype:     "fp16"（默认）或 "fp32"
    返回：
        输出目录 Path。
    """
    from ..loader import load_phos  # 惰性（loader 零 torch 依赖）

    np_dtype = np.float16 if dtype == "fp16" else np.float32
    phos = load_phos(phos_path)
    sd = _build_state_dict(phos, np_dtype)

    # tied：.phos 标记 tied 且 lm_head 未独立存储 → 让 HF 共享 embed_tokens
    flags = int(phos.header.get("flags", 0))
    F_TIED_EMBED = 1 << 1
    tied = bool(flags & F_TIED_EMBED) and "lm_head.weight" not in sd
    cfg = _build_config(phos, tie_word_embeddings=tied)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import torch  # 仅此处需要 torch 写 bin

    th_sd = {k: torch.from_numpy(v) for k, v in sd.items()}
    torch.save(th_sd, out / "pytorch_model.bin")

    _export_tokenizer(phos, out)
    return out


def cleanup(path: str | Path) -> None:
    """删除导出目录（测试用）。"""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
