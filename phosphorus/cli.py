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
    if getattr(a, "hgf_rank", None) is not None:
        cfg.quant.hgf_rank = a.hgf_rank
    if getattr(a, "block_size", None):
        cfg.train.block_size = a.block_size
    if getattr(a, "batch_size", None):
        cfg.train.batch_size = a.batch_size
    if getattr(a, "lr", None):
        cfg.train.lr = a.lr
    if getattr(a, "seed", None) is not None:
        cfg.train.seed = a.seed
    if getattr(a, "num_threads", None):
        cfg.train.num_threads = a.num_threads
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
    qat = cfg.train.mode.value == "qat-hgf"
    if cfg.quant.scheme.value == "ternary" and qat:
        export_trained(model, tok, cfg, out.with_suffix(".phos"))
        print(f"[ok] checkpoint -> {out.with_suffix('.pt')}")
        print(f"[ok] model     -> {out.with_suffix('.phos')}")
    else:
        print(
            f"[ok] checkpoint -> {out.with_suffix('.pt')}"
            "（fp32 基线不导出 .phos：引擎只接受三值权重）"
        )


def cmd_export(a: argparse.Namespace) -> None:
    """从 .pt 检查点导出 .phos（`quantize` 与 `export` 共用同一实现）。"""
    import torch

    from .export import export_model, inspect_phos
    from .model import LlamaModel
    from .tokenizer import Tokenizer

    if getattr(a, "inspect", False):
        print(inspect_phos(a.checkpoint))
        return

    cfg = build_config_from_args(a)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = LlamaModel(cfg.model, cfg.quant)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.load(a.tokenizer) if getattr(a, "tokenizer", None) else None
    weights = model.export_numpy()
    blob = export_model(weights, cfg.model, cfg.quant, tokenizer=tok, path=a.out)
    print(f"[ok] exported .phos ({len(blob):,} bytes) -> {a.out}")


def cmd_generate(a: argparse.Namespace) -> None:
    from .engine import PhosModel

    cfg = build_config_from_args(a)
    model = PhosModel.from_file(
        a.model,
        backend=getattr(a, "backend", "fp32"),
        kv_mode=getattr(a, "kv", "paged"),
        eviction=getattr(a, "eviction", "none"),
        kv_budget=getattr(a, "kv_budget", 0.5),
        window_size=getattr(a, "window_size", 32),
        sink_size=getattr(a, "sink_size", 4),
        max_seqs=getattr(a, "max_seqs", 1),
    )
    t0 = time.time()
    text = model.generate(
        a.prompt,
        max_new_tokens=cfg.decode.max_new_tokens,
        temperature=cfg.decode.temperature,
        top_k=cfg.decode.top_k,
        top_p=cfg.decode.top_p,
        seed=cfg.decode.seed,
        stream=getattr(a, "stream", False),
    )
    if not getattr(a, "stream", False):
        print(text)
    if a.bench:
        print(f"\n[bench] 生成 {cfg.decode.max_new_tokens} token 用时 {time.time() - t0:.2f}s")


def cmd_sched(a: argparse.Namespace) -> None:
    """连续批处理调度：一次融合前向服务多条 prompt（T4-D）。"""
    from .engine import PhosModel
    from .scheduler import Scheduler

    cfg = build_config_from_args(a)
    model = PhosModel.from_file(
        a.model,
        backend=getattr(a, "backend", "fp32"),
        kv_mode="paged",  # 连续批处理依赖 paged 多 seq_id 隔离
        eviction="none",  # T4-D 范围：连续批处理 + 驱逐不混用
        max_seqs=getattr(a, "max_seqs", 8),
    )
    prompts = list(a.prompts)
    if not prompts:
        raise SystemExit("--prompts 至少提供一条")
    sched = Scheduler(model, max_seqs=getattr(a, "max_seqs", 8))
    t0 = time.time()
    texts = sched.generate(
        prompts,
        max_new_tokens=cfg.decode.max_new_tokens,
        temperature=cfg.decode.temperature,
        top_k=cfg.decode.top_k,
        top_p=cfg.decode.top_p,
        seed=cfg.decode.seed,
    )
    dt = time.time() - t0
    total_tok = sum(len(t) for t in texts)
    print(f"[sched] {len(prompts)} 条 prompt，融合前向并发 max_seqs={sched.max_seqs}")
    for i, t in enumerate(texts):
        head = t[:80].replace("\n", " ")
        print(f"[{i}] {head!r}")
    if a.bench:
        tps = total_tok / max(dt, 1e-9)
        print(f"\n[bench] {total_tok} 新生成 token / {dt:.3f}s = {tps:.1f} tok/s")


def cmd_bench(a: argparse.Namespace) -> None:
    import numpy as np

    from .engine import PhosModel

    model = PhosModel.from_file(
        a.model,
        backend=getattr(a, "backend", "fp32"),
        kv_mode=getattr(a, "kv", "paged"),
        eviction=getattr(a, "eviction", "none"),
        kv_budget=getattr(a, "kv_budget", 0.5),
        window_size=getattr(a, "window_size", 32),
        sink_size=getattr(a, "sink_size", 4),
    )
    n = a.tokens
    # 特殊 token（如 <|bos|>）id 可能 >= vocab，嵌入表无对应行；回退到 id 0 做吞吐基准。
    start = model.bos_id if (model.bos_id is not None and 0 <= model.bos_id < model.vocab) else 0
    prompt_ids = [start] * min(a.prompt_len, model.max_seq_len)
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
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--block-size", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--hgf-rank", type=int, default=None)
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--num-threads", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("export", help="从 .pt 检查点导出 .phos")
    pe.add_argument("--checkpoint", "--in", dest="checkpoint", required=True)
    pe.add_argument("--config", default=None)
    pe.add_argument("--tokenizer", default=None)
    pe.add_argument("--out", required=True)
    pe.add_argument("--scale", default=None, choices=["tiny", "small", "base"])
    pe.add_argument("--vocab", type=int, default=None)
    pe.add_argument("--inspect", action="store_true", help="只打印 .phos 结构与张量索引")
    pe.set_defaults(func=cmd_export)

    # quantize 与 export 同实现：quantize 是面向「训练完直接量化」的语义别名，
    # 额外支持 --inspect（调试 .phos 的主要入口）。
    pq = sub.add_parser("quantize", help="量化导出 .phos（export 的别名，支持 --inspect）")
    pq.add_argument("--checkpoint", "--in", dest="checkpoint", required=True)
    pq.add_argument("--config", default=None)
    pq.add_argument("--tokenizer", default=None)
    pq.add_argument("--out", default=None)
    pq.add_argument("--scale", default=None, choices=["tiny", "small", "base"])
    pq.add_argument("--vocab", type=int, default=None)
    pq.add_argument("--hgf-rank", type=int, default=None)
    pq.add_argument("--inspect", action="store_true")
    pq.set_defaults(func=cmd_export)

    pg = sub.add_parser("generate", help="用 .phos 生成文本")
    pg.add_argument("--model", required=True)
    pg.add_argument("--prompt", default="")
    pg.add_argument("--config", default=None)
    pg.add_argument("--backend", default="fp32", choices=["fp32", "int8", "tl", "i2s"])
    pg.add_argument(
        "--kv", default="paged", choices=["dense", "paged"], help="KV cache 模式（T4-B）"
    )
    pg.add_argument(
        "--eviction",
        default="none",
        choices=["none", "snapkv", "pyramidkv", "h2o", "streamingllm"],
        help="KV 驱逐策略（T4-C，仅 paged 生效）",
    )
    pg.add_argument("--kv-budget", type=float, default=0.5, help="驱逐保留比例（默认 0.5）")
    pg.add_argument("--window-size", type=int, default=32, help="强制保留的最近窗口大小")
    pg.add_argument("--sink-size", type=int, default=4, help="强制保留的注意力锚点数")
    pg.add_argument(
        "--max-seq", type=int, default=1, dest="max_seqs", help="连续批处理并发序列数（T4-D）"
    )
    pg.add_argument("--max_new_tokens", type=int, default=None)
    pg.add_argument("--temperature", type=float, default=None)
    pg.add_argument("--top_k", type=int, default=None)
    pg.add_argument("--top_p", type=float, default=None)
    pg.add_argument("--stream", action="store_true")
    pg.add_argument("--bench", action="store_true")
    pg.set_defaults(func=cmd_generate)

    pb = sub.add_parser("bench", help="推理吞吐基准")
    pb.add_argument("--model", required=True)
    pb.add_argument("--tokens", type=int, default=64)
    pb.add_argument("--prompt-len", type=int, default=8)
    pb.add_argument("--backend", default="fp32", choices=["fp32", "int8", "tl", "i2s"])
    pb.add_argument(
        "--kv", default="paged", choices=["dense", "paged"], help="KV cache 模式（T4-B）"
    )
    pb.add_argument(
        "--eviction",
        default="none",
        choices=["none", "snapkv", "pyramidkv", "h2o", "streamingllm"],
        help="KV 驱逐策略（T4-C，仅 paged 生效）",
    )
    pb.add_argument("--kv-budget", type=float, default=0.5, help="驱逐保留比例（默认 0.5）")
    pb.add_argument("--window-size", type=int, default=32, help="强制保留的最近窗口大小")
    pb.add_argument("--sink-size", type=int, default=4, help="强制保留的注意力锚点数")
    pb.add_argument(
        "--max-seq", type=int, default=1, dest="max_seqs", help="连续批处理并发序列数（T4-D）"
    )
    pb.set_defaults(func=cmd_bench)

    ps = sub.add_parser("sched", help="连续批处理生成（T4-D）：多条 prompt 一次融合前向")
    ps.add_argument("--model", required=True)
    ps.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="一条或多条 prompt（空格分隔；含空格的 prompt 用引号包裹）",
    )
    ps.add_argument("--config", default=None)
    ps.add_argument("--backend", default="fp32", choices=["fp32", "int8", "tl", "i2s"])
    ps.add_argument(
        "--max-seqs", type=int, default=8, dest="max_seqs", help="并发序列上限（物理池按此分配）"
    )
    ps.add_argument("--max_new_tokens", type=int, default=None)
    ps.add_argument("--temperature", type=float, default=None)
    ps.add_argument("--top_k", type=int, default=None)
    ps.add_argument("--top_p", type=float, default=None)
    ps.add_argument("--bench", action="store_true")
    ps.set_defaults(func=cmd_sched)

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
