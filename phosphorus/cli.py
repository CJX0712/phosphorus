"""命令行入口（train / generate / tokenize / bench / export）。

设计：
    * 顶层只导入零重依赖模块（config / tokenizer / engine）；需要 torch 的
      train / export 子命令惰性导入 torch，保证 `generate`/`bench` 在纯
      numpy 环境也能跑（ARCHITECTURE.md §4 运行时零 torch 依赖约束）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import ModelConfig, PhosphorusConfig, TrainConfig


def build_config_from_args(a: argparse.Namespace) -> PhosphorusConfig:
    """根据命令行参数构造 PhosphorusConfig。"""
    cfg = PhosphorusConfig()
    if getattr(a, "config", None):
        cfg = PhosphorusConfig.load(a.config)
    if getattr(a, "scale", None):
        cfg.model = ModelConfig.from_scale(a.scale)
    if getattr(a, "vocab", None):
        cfg.model.vocab_size = a.vocab
    if getattr(a, "steps", None):
        cfg.train.steps = a.steps
    if getattr(a, "mode", None):
        cfg.train.mode = TrainConfig.mode.__class__(a.mode)  # TrainMode 枚举
    if getattr(a, "max_new_tokens", None):
        cfg.decode.max_new_tokens = a.max_new_tokens
    if getattr(a, "temperature", None):
        cfg.decode.temperature = a.temperature
    if getattr(a, "top_k", None):
        cfg.decode.top_k = a.top_k
    if getattr(a, "top_p", None):
        cfg.decode.top_p = a.top_p
    return cfg


# --------------------------------------------------------------------------- #
# 子命令实现                                                                   #
# --------------------------------------------------------------------------- #


def cmd_train(a: argparse.Namespace) -> None:
    from .train import export_trained, run_train

    cfg = build_config_from_args(a)
    model, tok, stats = run_train(cfg, a.corpus, verbose=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save({"model": model.state_dict(), "stats": stats}, out.with_suffix(".pt"))
    if cfg.quant.scheme.value == "ternary":
        export_trained(model, tok, cfg, out.with_suffix(".phos"))
        print(f"[ok] checkpoint -> {out.with_suffix('.pt')}")
        print(f"[ok] model     -> {out.with_suffix('.phos')}")
    else:
        print(f"[ok] checkpoint -> {out.with_suffix('.pt')}（非 ternary，不导出 .phos）")


def cmd_export(a: argparse.Namespace) -> None:
    """从 .pt 检查点导出 .phos。"""
    import torch

    from .export import export_model
    from .model import LlamaModel
    from .tokenizer import Tokenizer

    cfg = build_config_from_args(a)
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    model = LlamaModel(cfg.model, cfg.quant)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.load(a.tokenizer) if a.tokenizer else None
    weights = model.export_numpy()
    blob = export_model(weights, cfg.model, cfg.quant, tokenizer=tok, path=a.out)
    print(f"[ok] exported .phos ({len(blob):,} bytes) -> {a.out}")


def cmd_generate(a: argparse.Namespace) -> None:
    from .engine import PhosModel

    cfg = build_config_from_args(a)
    model = PhosModel.from_file(a.model)
    t0 = time.time()
    text = model.generate(
        a.prompt,
        max_new_tokens=cfg.decode.max_new_tokens,
        temperature=cfg.decode.temperature,
        top_k=cfg.decode.top_k,
        top_p=cfg.decode.top_p,
        seed=cfg.decode.seed,
    )
    print(text)
    if a.bench:
        print(f"\n[bench] 生成 {cfg.decode.max_new_tokens} token 用时 {time.time() - t0:.2f}s")


def cmd_bench(a: argparse.Namespace) -> None:
    import numpy as np

    from .engine import PhosModel

    model = PhosModel.from_file(a.model)
    n = a.tokens
    start = model.bos_id if model.bos_id is not None else 0
    prompt_ids = [start] * min(8, model.max_seq_len)
    model.reset()
    t0 = time.time()
    logits = model.forward(np.array(prompt_ids, dtype=np.int64))
    prefill_t = time.time() - t0
    next_id = int(np.argmax(logits[-1]))
    t1 = time.time()
    for _ in range(n):
        lg = model.forward(np.array([next_id], dtype=np.int64))
        next_id = int(np.argmax(lg[-1]))
    decode_t = time.time() - t1
    print(
        f"[bench] prefill {len(prompt_ids)} tok: {prefill_t:.3f}s ({len(prompt_ids) / max(prefill_t, 1e-9):.1f} tok/s)"
    )
    print(f"[bench] decode  {n} tok: {decode_t:.3f}s ({n / max(decode_t, 1e-9):.1f} tok/s)")


def cmd_tokenize(a: argparse.Namespace) -> None:
    from .tokenizer import Tokenizer, train_from_corpus

    if a.action == "train":
        tok = train_from_corpus(a.corpus, vocab_size=a.vocab)
        tok.save(a.out)
        print(f"[ok] tokenizer ({tok.vocab_size} tokens) -> {a.out}")
    elif a.action == "encode":
        tok = Tokenizer.load(a.tokenizer)
        ids = tok.encode(a.text)
        print(ids[:200], "..." if len(ids) > 200 else "")
        print(f"[ok] {len(ids)} tokens")
    elif a.action == "decode":
        tok = Tokenizer.load(a.tokenizer)
        ids = [int(x) for x in a.text.split()]
        print(repr(tok.decode(ids)))
    elif a.action == "roundtrip":
        tok = Tokenizer.load(a.tokenizer)
        ids = tok.encode(a.text)
        print("[ok] encode->decode 恒等:", tok.decode(ids) == a.text)


# --------------------------------------------------------------------------- #
# 主解析器                                                                     #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phosphorus", description="Phosphorus：三值原生 LLM 训练与推理"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="训练并导出 .phos")
    pt.add_argument("--corpus", required=True)
    pt.add_argument("--config", default=None)
    pt.add_argument("--out", default="ckpt/tiny")
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--mode", default=None, choices=["fp32", "qat-hgf"])
    pt.add_argument("--vocab", type=int, default=None)
    pt.add_argument("--scale", default=None, choices=["tiny", "small", "base"])
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("export", help="从 .pt 检查点导出 .phos")
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--config", default=None)
    pe.add_argument("--tokenizer", default=None)
    pe.add_argument("--out", required=True)
    pe.add_argument("--scale", default=None, choices=["tiny", "small", "base"])
    pe.add_argument("--vocab", type=int, default=None)
    pe.set_defaults(func=cmd_export)

    pg = sub.add_parser("generate", help="用 .phos 生成文本")
    pg.add_argument("--model", required=True)
    pg.add_argument("--prompt", default="")
    pg.add_argument("--config", default=None)
    pg.add_argument("--max_new_tokens", type=int, default=None)
    pg.add_argument("--temperature", type=float, default=None)
    pg.add_argument("--top_k", type=int, default=None)
    pg.add_argument("--top_p", type=float, default=None)
    pg.add_argument("--bench", action="store_true")
    pg.set_defaults(func=cmd_generate)

    pb = sub.add_parser("bench", help="推理吞吐基准")
    pb.add_argument("--model", required=True)
    pb.add_argument("--tokens", type=int, default=64)
    pb.set_defaults(func=cmd_bench)

    pk = sub.add_parser("tokenize", help="分词器训练 / 编解码")
    pk.add_argument("--action", required=True, choices=["train", "encode", "decode", "roundtrip"])
    pk.add_argument("--corpus", default=None)
    pk.add_argument("--text", default=None)
    pk.add_argument("--tokenizer", default=None)
    pk.add_argument("--out", default="tokenizer.json")
    pk.add_argument("--vocab", type=int, default=8192)
    pk.set_defaults(func=cmd_tokenize)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
