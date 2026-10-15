"""训练入口（PyTorch，惰性导入）。

支持两条路径（ARCHITECTURE.md §8）：
    * `fp32`：标准 fp32 基线（仅作困惑度对照，不导出可部署 .phos）
    * `qat-hgf`：BitNet 三值 QAT + HGF 低秩校正（默认，可直接导出 1.58-bit .phos）

借鉴来源：
    * nanoGPT 训练循环（AdamW + cosine decay + warmup + grad clip，Karpathy）— A-16
    * BitNet QAT（absmean 三值 + STE，Wang et al. 2024）— A-1
    * HGF 低秩校正（Pizzo, 2026）— A-11

不做花哨技巧，先把端到端跑通、可复现、可验证。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PhosphorusConfig, QuantConfig, TrainConfig
from .data import TextDataset
from .model import LlamaModel
from .tokenizer import Tokenizer, train_from_corpus


def _get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def _estimate_loss(model: LlamaModel, ds: TextDataset, cfg: TrainConfig) -> float:
    model.eval()
    tot = 0.0
    for _ in range(cfg.eval_iters):
        x, y = ds.val_batch(1)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += float(loss.item())
    model.train()
    return tot / cfg.eval_iters


def run_train(
    cfg: PhosphorusConfig,
    corpus_path: str | Path,
    tokenizer: Tokenizer | None = None,
    verbose: bool = True,
) -> tuple[LlamaModel, Tokenizer, dict]:
    """执行一次训练。返回 (model, tokenizer, stats)。"""
    mcfg, qcfg, tcfg = cfg.model, cfg.quant, cfg.train
    # 训练模式映射到量化开关
    qat = tcfg.mode.value == "qat-hgf"
    qcfg = QuantConfig(
        scheme=qcfg.scheme,
        w_granularity=qcfg.w_granularity,
        a_granularity=qcfg.a_granularity,
        block_size=qcfg.block_size,
        a_bits=qcfg.a_bits,
        hgf_rank=qcfg.hgf_rank if qat else 0,
        qat=qat,
    )

    rng = np.random.default_rng(tcfg.seed)
    if tokenizer is None:
        tokenizer = train_from_corpus(corpus_path, vocab_size=mcfg.vocab_size)
    ds = TextDataset.from_corpus(corpus_path, tokenizer, block_size=tcfg.block_size)

    model = LlamaModel(mcfg, qcfg)
    n_params = model.n_params if hasattr(model, "n_params") else None
    if n_params is None:
        n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"[train] params={n_params:,} | mode={tcfg.mode.value} | qat={qat} | hgf_rank={qcfg.hgf_rank}"
        )
        print(f"[train] dataset tokens={len(ds.tokens):,} blocks={len(ds):,}")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, betas=tcfg.betas, weight_decay=tcfg.weight_decay
    )
    if tcfg.num_threads:
        torch.set_num_threads(tcfg.num_threads)

    t0 = time.time()
    best_val = float("inf")
    for step in range(tcfg.steps):
        lr = _get_lr(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = ds.batch(tcfg.batch_size, rng)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if verbose and (step % tcfg.log_interval == 0 or step == tcfg.steps - 1):
            dt = time.time() - t0
            print(
                f"[train] step {step:4d}/{tcfg.steps} | loss {float(loss.item()):.4f} | lr {lr:.5f} | {dt:.1f}s"
            )

        if tcfg.eval_interval and step % tcfg.eval_interval == 0 and step > 0:
            val = _estimate_loss(model, ds, tcfg)
            best_val = min(best_val, val)
            if verbose:
                print(f"[train]   val_loss {val:.4f} | ppl {math.exp(val):.2f}")

    final_val = _estimate_loss(model, ds, tcfg)
    best_val = min(best_val, final_val)
    if verbose:
        print(
            f"[train] done | final_val_loss {final_val:.4f} | best {best_val:.4f} | total {time.time() - t0:.1f}s"
        )

    stats = {
        "n_params": n_params,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "mode": tcfg.mode.value,
    }
    return model, tokenizer, stats


def export_trained(
    model: LlamaModel, tokenizer: Tokenizer, cfg: PhosphorusConfig, out_path: str | Path
) -> bytes:
    """把训练好的模型导出为 .phos（需要 qat 路径）。"""
    from .export import export_model

    weights = model.export_numpy()
    blob = export_model(weights, cfg.model, cfg.quant, tokenizer=tokenizer, path=out_path)
    return blob


def main() -> None:
    import argparse

    from .cli import build_config_from_args

    ap = argparse.ArgumentParser(prog="phosphorus train", description="训练 Phosphorus 模型")
    add_common_args(ap)
    a = ap.parse_args()
    cfg = build_config_from_args(a)
    model, tok, stats = run_train(cfg, a.corpus, verbose=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats}, out.with_suffix(".pt"))
    if cfg.quant.scheme.value == "ternary":
        export_trained(model, tok, cfg, out.with_suffix(".phos"))
        print(f"[train] exported .phos -> {out.with_suffix('.phos')}")
    else:
        print("[train] 非 ternary 方案不导出 .phos（引擎仅支持三值）")


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--corpus", required=True, help="语料文本路径")
    ap.add_argument("--config", default=None, help="PhosphorusConfig JSON 路径（可选）")
    ap.add_argument("--out", default="ckpt/tiny", help="输出前缀（.pt / .phos）")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=["fp32", "qat-hgf"])
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--scale", default=None, choices=["tiny", "small", "base"])


if __name__ == "__main__":
    main()
