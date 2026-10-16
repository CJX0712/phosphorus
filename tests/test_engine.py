"""不变量 ⑦：engine（纯 numpy）与 torch 参考模型 logits 相对误差 < 5%。

默认走忠实路径（backend="fp32"，激活 fp32、权重三值），与 torch QAT
前向逐元素等价。i2s 整数路径是部署态近似（激活量化误差来源），仅校验有限且
分布相近，不作为 <5% 红线（对应风险 R6）。
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from phosphorus.config import ModelConfig, QuantConfig  # noqa: E402
from phosphorus.engine import PhosModel  # noqa: E402
from phosphorus.export import export_model  # noqa: E402
from phosphorus.loader import load_phos  # noqa: E402
from phosphorus.model import LlamaModel, apply_rope, build_rope  # noqa: E402
from phosphorus.tokenizer import Tokenizer  # noqa: E402


def _torch_logits(mcfg, qcfg, ids):
    model = LlamaModel(mcfg, qcfg).eval()
    with torch.no_grad():
        x = model.tok_emb[torch.from_numpy(ids)]
        B, T = 1, ids.shape[0]
        cos, sin = build_rope(mcfg.rope_theta, mcfg.d_head, T, "cpu")
        for ly in model.layers:
            h = ly.attn_norm(x)
            q = ly.q_proj(h)
            k = ly.k_proj(h)
            v = ly.v_proj(h)
            n_rep = mcfg.n_head // mcfg.n_kv_head
            k = k.view(B, T, mcfg.n_kv_head, mcfg.d_head).transpose(1, 2)
            v = v.view(B, T, mcfg.n_kv_head, mcfg.d_head).transpose(1, 2)
            q = q.view(B, T, mcfg.n_head, mcfg.d_head).transpose(1, 2)
            k = k.repeat_interleave(n_rep, 1)
            v = v.repeat_interleave(n_rep, 1)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(mcfg.d_head)
            scores = scores.masked_fill(
                torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")
            )
            attn = torch.softmax(scores, -1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, mcfg.d_model)
            o = ly.o_proj(out)
            x = x + ly.subln_attn * o
            h2 = ly.ffn_norm(x)
            g = torch.relu(ly.gate_proj(h2)) ** 2
            u = ly.up_proj(h2)
            x = x + ly.subln_ffn * ly.down_proj(g * u)
        x = x * model.norm_final
        emb = model.tok_emb if mcfg.tied_embed else model.lm_head
        return (x @ emb.t())[0].numpy().astype(np.float32), model


def test_engine_vs_torch_logits():
    """忠实路径下 logits 相对误差 < 5%，且 argmax 完全一致。"""
    mcfg = ModelConfig(
        d_model=64, n_layer=3, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=4, qat=True)
    ids = np.array([3, 17, 42, 8, 99, 5], dtype=np.int64)

    tlogits, model = _torch_logits(mcfg, qcfg, ids)
    weights = model.export_numpy()
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_engine.phos")
    pf = load_phos(blob)
    eng = PhosModel(pf, backend="fp32")
    elogits = eng.forward(ids)

    rel = np.abs(elogits - tlogits).sum() / (np.abs(tlogits).sum() + 1e-9)
    agree = float(np.mean(np.argmax(elogits, -1) == np.argmax(tlogits, -1)))
    assert rel < 0.05, f"不变量 ⑦ 失败: rel_err={rel:.4%}"
    assert agree == 1.0, f"argmax 一致率 {agree:.2%}"


def test_engine_int8_path_finite_and_close():
    """int8 激活路径须有限，且与忠实路径分布相近（cos_sim > 0.9）。"""
    mcfg = ModelConfig(
        d_model=64, n_layer=3, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=4, qat=True)
    ids = np.array([3, 17, 42, 8, 99, 5], dtype=np.int64)

    _, model = _torch_logits(mcfg, qcfg, ids)
    weights = model.export_numpy()
    blob = export_model(weights, mcfg, qcfg, path="ckpt/_t_engine_i8.phos")
    pf = load_phos(blob)
    eng_fp = PhosModel(pf, backend="fp32")
    eng_i8 = PhosModel(pf, backend="i2s")
    lf = eng_fp.forward(ids)
    li = eng_i8.forward(ids)
    assert np.all(np.isfinite(li))
    cos = float(
        np.dot(lf.ravel(), li.ravel())
        / (np.linalg.norm(lf.ravel()) * np.linalg.norm(li.ravel()) + 1e-12)
    )
    assert cos > 0.9, f"int8 路径与忠实路径分布偏离过大 cos={cos:.4f}"


def test_engine_generate_runs():
    """engine.generate 在纯 numpy 环境可端到端运行（不加载 torch）。"""
    mcfg = ModelConfig(
        d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=0, qat=True)
    model = LlamaModel(mcfg, qcfg).eval()
    with torch.no_grad():
        weights = model.export_numpy()
    tok = Tokenizer.train("hello world abc def", vocab_size=200)
    blob = export_model(weights, mcfg, qcfg, tokenizer=tok, path="ckpt/_t_gen.phos")
    pf = load_phos(blob)
    eng = PhosModel(pf, backend="fp32")
    out = eng.generate("abc", max_new_tokens=4, temperature=0.0, seed=0)
    assert isinstance(out, str)


def test_engine_generate_empty_prompt_falls_back_when_bos_out_of_vocab():
    """回归：tokenizer 特殊 token（如 <|bos|>）id 可能 >= vocab（词表外），
    generate 空 prompt 须回退到 id 0 而非 IndexError（复现 bench 的崩溃）。"""
    mcfg = ModelConfig(
        d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, vocab_size=256, max_seq_len=128
    )
    qcfg = QuantConfig(hgf_rank=0, qat=True)
    model = LlamaModel(mcfg, qcfg).eval()
    with torch.no_grad():
        weights = model.export_numpy()
    # vocab_size=256 -> 基础 256 字节 + 0 个 merge；4 个特殊 token 追加到 id 256+（词表外）
    tok = Tokenizer.train("hello world abc def", vocab_size=256)
    assert tok.specials.get("<|bos|>", -1) >= mcfg.vocab_size, "需复现 special id 越界"
    blob = export_model(weights, mcfg, qcfg, tokenizer=tok, path="ckpt/_t_spec.phos")
    pf = load_phos(blob)
    eng = PhosModel(pf, backend="fp32")
    out = eng.generate("", max_new_tokens=2, temperature=0.0, seed=0)
    assert isinstance(out, str)
