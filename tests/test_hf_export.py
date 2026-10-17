"""T5-C8：`.phos` → HuggingFace 导出验证。

分层守卫：
  * 模块级 `importorskip("torch")` —— 无 torch 环境（纯 numpy 推理 venv）整体跳过。
  * `test_hf_from_pretrained` 内 `importorskip("transformers")` —— 仅当环境装了
    transformers 才做真实 `from_pretrained` 加载；否则跳过但导出结构仍被前序用例验证。
"""

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
PHOS = REPO / "ckpt" / "tiny.phos"

from phosphorus.convert import hf_export  # noqa: E402
from phosphorus.loader import load_phos  # noqa: E402


def _recompute_q_proj(phos):
    """按导出公式手动还原 layers.0 q_proj 的 fp32 权重（三值解包 + scale + HGF 折叠）。

    注意：`.phos` 的 `scales` / `hgf` 以**全名**（含 `.weight` 后缀）为键，
    必须以 `name` 原样查表。
    """
    name = "layers.0.q_proj.weight"  # .phos 原始键名（HF 的 self_attn. 前缀是导出时才映射的）
    w = phos.weights[name].astype(np.float32)
    scale = phos.scales.get(name)
    if scale is not None:
        w = w * scale.reshape(-1, 1)
    hgf = phos.hgf.get(name)
    if hgf is not None:
        A, B, gate = hgf
        r = int(A.shape[0])
        alpha = float(phos.header.get("hgf_alpha", 1.0))
        w = w + gate * (alpha / r) * (B @ A)
    return w


@pytest.fixture
def exported(tmp_path):
    if not PHOS.exists():
        pytest.skip(f"{PHOS} 不存在（需先训练 tiny 模型）")
    out = tmp_path / "hf_model"
    hf_export.export_to_hf(PHOS, out, dtype="fp16")
    yield out
    shutil.rmtree(out, ignore_errors=True)


def test_map_key_coverage():
    # 关键命名映射必须存在且不漏掉投影权重
    assert hf_export._map_key("tok_emb") == "model.embed_tokens.weight"
    assert hf_export._map_key("norm_final") == "model.norm.weight"
    assert hf_export._map_key("lm_head") == "lm_head.weight"
    assert hf_export._map_key("layers.0.attn_norm") == "model.layers.0.input_layernorm.weight"
    assert (
        hf_export._map_key("layers.0.ffn_norm") == "model.layers.0.post_attention_layernorm.weight"
    )
    assert hf_export._map_key("layers.3.q_proj.weight") == "model.layers.3.self_attn.q_proj.weight"
    assert hf_export._map_key("layers.3.down_proj.weight") == "model.layers.3.mlp.down_proj.weight"
    # SubLN / scale / hgf 附属张量应被跳过
    assert hf_export._map_key("layers.0.subln_attn") is None
    assert hf_export._map_key("layers.0.q_proj.scale") is None
    assert hf_export._map_key("layers.0.q_proj.hgf_a") is None
    assert hf_export._map_key("early_lora_a") is None


def test_config_and_bin(exported):
    cfg = json.loads((exported / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "llama"
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert cfg["hidden_size"] > 0 and cfg["num_hidden_layers"] > 0
    # pytorch_model.bin 可加载且含标准键
    sd = torch.load(exported / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert "model.embed_tokens.weight" in sd
    assert "model.norm.weight" in sd
    assert "lm_head.weight" in sd or cfg["tie_word_embeddings"] is True
    assert "model.layers.0.self_attn.q_proj.weight" in sd
    assert "model.layers.0.mlp.down_proj.weight" in sd
    # SubLN 在 HF 标准键中不应出现（已知省略）
    assert not any("subln" in k for k in sd)


def test_dequant_matches_formula(exported):
    phos = load_phos(PHOS)
    sd = torch.load(exported / "pytorch_model.bin", map_location="cpu", weights_only=True)
    exported_w = sd["model.layers.0.self_attn.q_proj.weight"].numpy()
    recomputed = _recompute_q_proj(phos).astype(np.float16).astype(np.float32)
    assert exported_w.shape == recomputed.shape
    # fp16 还原 + HGF 折叠，容差放宽到相对 2% / 绝对 0.1
    assert np.allclose(exported_w, recomputed, rtol=2e-2, atol=1e-1), (
        f"max abs diff = {np.abs(exported_w - recomputed).max():.4f}"
    )


def test_hf_from_pretrained(exported):
    """真实 from_pretrained 加载（需 transformers；无则跳过）。"""
    pytest.importorskip("transformers")
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(str(exported), local_files_only=True)
    # 跑一次前向，确认 logits 有限且形状正确
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits
    assert logits.shape[0] == 1 and logits.shape[2] == model.config.vocab_size
    assert torch.isfinite(logits).all()
