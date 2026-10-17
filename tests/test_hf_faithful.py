"""T5-C8 忠实导出验证：忠实模式产出的 PhosphorusForCausalLM 必须与原生引擎逐位等价。

不变量 ⑦（引擎 ↔ torch logits 相对误差 < 5%）的强化版：忠实模式应做到**近逐位**
一致（同数学前向），远优于近似模式。

本测试不依赖 transformers：modeling_phosphorus 在无 transformers 时回退为纯 torch
nn.Module，可直接加载导出的 pytorch_model.bin 并比对原生引擎。transformers 的
`from_pretrained` 集成路径用 importorskip 守卫（本机未装 transformers 时跳过）。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phosphorus.convert import hf_export
from phosphorus.convert.modeling_phosphorus import PhosphorusForCausalLM
from phosphorus.engine import PhosModel

TINY = "ckpt/tiny.phos"
IDS = [1, 2, 3, 4, 5, 6, 7, 8]


def _load_faithful_model(out_dir: Path, dtype=torch.float32):
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    ns = type("Cfg", (), cfg)()  # 用 SimpleNamespace 承载字段（无需 transformers）
    model = PhosphorusForCausalLM(ns)
    sd = torch.load(out_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(dtype)
    model.eval()
    return model


def test_faithful_matches_engine_fp32(tmp_path):
    """忠实 fp32 导出必须与原生引擎逐位等价（相对误差 ≪ 1e-2）。"""
    out = tmp_path / "hf_faithful"
    hf_export.export_to_hf(TINY, out, dtype="fp32", faithful=True)
    assert (out / "modeling_phosphorus.py").exists(), "忠实模式必须复制建模文件"
    assert (out / "config.json").exists()

    cfg_json = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["model_type"] == "phosphorus"
    assert cfg_json["architectures"] == ["PhosphorusForCausalLM"]
    assert "auto_map" in cfg_json

    model = _load_faithful_model(out, dtype=torch.float32)
    ids_t = torch.tensor([IDS], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids_t)[0].numpy()  # [T, V]

    eng = PhosModel.from_file(TINY, backend="fp32")
    eng.reset()
    ref = eng.forward(np.array(IDS, dtype=np.int64))  # [T, V]

    assert logits.shape == ref.shape, f"shape {logits.shape} vs {ref.shape}"
    rel = np.max(np.abs(logits - ref)) / (np.max(np.abs(ref)) + 1e-8)
    # 近逐位：同数学前向，仅浮点累加顺序差异 → 期望 < 1e-3
    assert rel < 1e-2, f"忠实模式与原生引擎偏离过大：rel={rel:.3e}"
    # 同时记录实际量级，便于后续收紧阈值
    print(f"[faithful] max rel err vs engine = {rel:.3e}")


def test_faithful_subln_keys_present(tmp_path):
    """忠实 state_dict 必须包含 SubLN 增益张量（近似模式会跳过）。"""
    out = tmp_path / "hf_faithful2"
    hf_export.export_to_hf(TINY, out, dtype="fp32", faithful=True)
    sd = torch.load(out / "pytorch_model.bin", map_location="cpu", weights_only=True)
    keys = set(sd.keys())
    assert any(k.endswith("subln_attn") for k in keys), "缺失 subln_attn"
    assert any(k.endswith("subln_ffn") for k in keys), "缺失 subln_ffn"
    # 近似模式对照：默认导出应跳过 SubLN
    out2 = tmp_path / "hf_approx"
    hf_export.export_to_hf(TINY, out2, dtype="fp32", faithful=False)
    sd2 = torch.load(out2 / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert not any("subln" in k for k in sd2), "近似模式不应含 subln"


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers") is None,
    reason="需要 transformers 以验证 from_pretrained 集成路径",
)
def test_faithful_from_pretrained(tmp_path):
    """有 transformers 时，from_pretrained(trust_remote_code=True) 应加载并逐位等价。"""
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM

    out = tmp_path / "hf_faithful3"
    hf_export.export_to_hf(TINY, out, dtype="fp32", faithful=True)
    model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    model.eval()
    ids_t = torch.tensor([IDS], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids_t).logits[0].numpy()

    eng = PhosModel.from_file(TINY, backend="fp32")
    eng.reset()
    ref = eng.forward(np.array(IDS, dtype=np.int64))
    rel = np.max(np.abs(logits - ref)) / (np.max(np.abs(ref)) + 1e-8)
    assert rel < 1e-2, f"from_pretrained 忠实模型偏离引擎：rel={rel:.3e}"
