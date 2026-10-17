"""T5 · C6 HGF Pareto：`--ranks 0,1,2,4,8,16` 训练扫描 → r / 额外字节% / val_ppl / Δppl%。

用法：
    python scripts/bench_pareto.py                       # 全档位（fp32 天花板 + 6 档 rank）
    python scripts/bench_pareto.py --ranks 0,4           # 子集
    python scripts/bench_pareto.py --steps 150           # 与 ckpt/tiny.phos 同协议

产物（`--out-dir`，默认 `benchmarks/`）：
    pareto.json    原始数据（每档训练统计 + 文件字节 + 引擎 ppl）
    pareto 表同时打印到 stdout 并追加进 REPORT 风格的 Markdown（pareto.md）

口径（诚实性优先）：
    * 基线 = 同配置同种子的 **fp32 训练**（`TrainMode.FP32`）——量化质量损失的参照上界。
    * 两档协议：`tiny`（d=192, 3 层，与 ckpt/tiny.phos 同协议，150 步，~11s/run）；
      `d512`（d=512, 8 层——ARCHITECTURE.md §3.7 的「15% 额外内存」口径：d=512 时
      r=4 ≈ 15% 三值字节；d=192 时 r=4 ≈ 36%，公式 48·r·(in+out)/(5·in·out)）。
    * `val_ppl` 双口径：**torch 口径**（全部档位 torch 前向 teacher-forcing，跨档可比，
      恢复率用它）；**引擎口径**（.phos 导出后 PhosModel fp32 前向，部署路径数字；
      fp32 天花板无法导出 .phos——引擎只支持三值——故只有 torch 口径）。
    * `额外字节%` 双口径：.phos 文件增量（相对 r=0）；HGF/三值权重%（纯权重口径）。
    * `质量损失恢复%` = (ppl(r=0) − ppl(r)) / (ppl(r=0) − ppl(fp32)) × 100%。
    * 评估语料与训练语料同源（tiny shakespeare 未切 held-out）——in-domain 指标，
      已在报告中注明，不作为泛化性能 claim。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phosphorus.config import (  # noqa: E402
    ModelConfig,
    PhosphorusConfig,
    QuantConfig,
    TrainConfig,
    TrainMode,
)
from phosphorus.engine import PhosModel  # noqa: E402
from phosphorus.loader import load_phos  # noqa: E402
from phosphorus.tokenizer import train_from_corpus  # noqa: E402
from phosphorus.train import export_trained, run_train  # noqa: E402

PPL_TOKENS = 2048
PPL_BLOCK = 128

# 与 ckpt/tiny.phos（tiny_train.log）完全同协议，保证跨档可比
# 两档训练协议：
PROTOCOLS = {
    # tiny：与已发布 ckpt/tiny.phos 同协议（d=192, 3 层, 150 步, ~11s/run）
    "tiny": {
        "model": dict(
            d_model=192,
            n_layer=3,
            n_head=3,
            n_kv_head=1,
            d_ff=384,
            vocab_size=4096,
            max_seq_len=256,
        ),
        "train": dict(steps=150, batch_size=4, block_size=64, warmup_steps=10, seed=1337),
    },
    # d512：ARCHITECTURE.md §3.7 的「15% 额外内存」口径（d=512 时 r=4 ≈ 15% 三值字节）
    "d512": {
        "model": dict(
            d_model=512,
            n_layer=8,
            n_head=8,
            n_kv_head=2,
            d_ff=1280,
            vocab_size=4096,
            max_seq_len=256,
        ),
        "train": dict(steps=150, batch_size=4, block_size=64, warmup_steps=10, seed=1337),
    },
}


def eval_ppl(model: PhosModel, tokens: list[int]) -> float:
    """引擎侧 teacher-forcing 困惑度（与 bench_matrix 同口径）。"""
    nll, cnt = 0.0, 0
    for i in range(0, len(tokens) - PPL_BLOCK - 1, PPL_BLOCK):
        ctx = np.asarray(tokens[i : i + PPL_BLOCK], dtype=np.int64)
        tgt = np.asarray(tokens[i + 1 : i + PPL_BLOCK], dtype=np.int64)
        model.reset()
        logits = model.forward(ctx)[:-1]
        m = logits.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        nll += float((lse[..., 0] - logits[np.arange(len(tgt)), tgt]).sum())
        cnt += len(tgt)
    return float(np.exp(nll / max(cnt, 1)))


def _torch_ppl(model, tokens: list[int]) -> float:
    """fp32 天花板专用：.phos 仅支持三值权重无法导出 fp32 基线，
    故用 torch 前向按同一协议（128 块、 teacher-forcing）实测。"""
    import math

    import torch

    model.eval()
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tokens) - PPL_BLOCK - 1, PPL_BLOCK):
            ctx = tokens[i : i + PPL_BLOCK]
            tgt = tokens[i + 1 : i + PPL_BLOCK]
            logits = model(torch.tensor([ctx], dtype=torch.long)).reshape(-1, model.m.vocab_size)
            tgt_t = torch.tensor(tgt, dtype=torch.long)
            nll += float(torch.nn.functional.cross_entropy(logits[:-1], tgt_t, reduction="sum"))
            cnt += len(tgt)
    return math.exp(nll / max(cnt, 1))


def train_one(
    protocol: str, rank: int | None, steps: int, corpus: Path, tok, out_dir: Path, ppl_slice
) -> dict:
    """训练一档并导出。`rank=None` → fp32 天花板基线（torch 前向测 ppl）。

    ppl 双口径：`ppl_torch`（全部档位，torch 前向同协议——跨档可比，恢复率用它）；
    `ppl_engine`（仅 qat 档，.phos 导出后引擎前向——部署路径数字）。
    """
    proto = PROTOCOLS[protocol]
    qat = rank is not None
    cfg = PhosphorusConfig(
        model=ModelConfig(**proto["model"]),
        quant=QuantConfig(hgf_rank=(rank or 0), qat=qat),
        train=TrainConfig(
            **{
                **proto["train"],
                "steps": steps,
                "mode": TrainMode.QAT_HGF if qat else TrainMode.FP32,
            }
        ),
    )
    tag = f"{protocol}_hgf_r{rank}" if qat else f"{protocol}_fp32_ceiling"
    t0 = time.time()
    model, _tok, stats = run_train(cfg, corpus, tokenizer=tok, verbose=True)
    ppl_torch = _torch_ppl(model, ppl_slice)
    row = {
        "protocol": protocol,
        "tag": tag,
        "rank": rank,
        "qat": qat,
        "train_seconds": 0.0,
        "ppl_torch": ppl_torch,
        "ppl_engine": None,
        "file_bytes": 0,
        "ternary_bytes": 0,
        "hgf_bytes_file": 0,
    }
    if qat:
        out_path = out_dir / f"{tag}.phos"
        export_trained(model, tok, cfg, out_path)
        phos = load_phos(out_path)
        row["file_bytes"] = out_path.stat().st_size
        row["ternary_bytes"] = sum(len(v) for v in phos.raw.values())
        # 文件口径：HGF 因子在 .phos 中为 f16（2B/元素）+ 每张量 1 个 f16 gate
        row["hgf_bytes_file"] = sum(
            2 * (int(np.asarray(a).size) + int(np.asarray(b).size) + 1)
            for a, b, _g in phos.hgf.values()
        )
        row["ppl_engine"] = eval_ppl(PhosModel.from_file(out_path, backend="fp32"), ppl_slice)
    row["train_seconds"] = round(time.time() - t0, 1)
    import math

    row["train_val_ppl"] = math.exp(stats["final_val_loss"])
    return row


def render_table(rows: list[dict]) -> str:
    """按 protocol 分组渲染；ppl 用 torch 同口径（跨档可比）；字节双口径。"""
    lines: list[str] = []
    for proto in sorted({r["protocol"] for r in rows}):
        sub = [r for r in rows if r["protocol"] == proto]
        r0 = next((x for x in sub if x["rank"] == 0), None)
        fp = next((x for x in sub if not x["qat"]), None)
        base_file = r0["file_bytes"] if r0 else 0
        ppl0 = r0["ppl_torch"] if r0 else float("nan")
        ppl_fp = fp["ppl_torch"] if fp else float("nan")
        cfg = PROTOCOLS.get(proto, {})
        lines += [
            f"#### 协议 `{proto}`（d={cfg.get('model', {}).get('d_model', '?')}, "
            f"{cfg.get('model', {}).get('n_layer', '?')} 层）",
            "",
            "| 档位 | r | 文件增量% | HGF/三值权重% | train val_ppl | ppl（torch 口径） | ppl（引擎口径） | Δppl%（vs r=0） | 质量损失恢复% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sub:
            if r["qat"] and base_file:
                f_extra = f"{100.0 * (r['file_bytes'] - base_file) / base_file:+.2f}%"
                w_extra = (
                    f"{100.0 * r['hgf_bytes_file'] / max(r['ternary_bytes'], 1):+.2f}%"
                    if r["ternary_bytes"]
                    else "—"
                )
            else:
                f_extra = w_extra = "—"
            dppl = 100.0 * (r["ppl_torch"] - ppl0) / ppl0
            if ppl0 > ppl_fp:
                rec = 100.0 * (ppl0 - r["ppl_torch"]) / (ppl0 - ppl_fp)
                rec_s = f"{rec:+.1f}%"
            else:
                rec_s = "n/a（天花板未高于 r=0）"
        eng = f"{r['ppl_engine']:.2f}" if r["ppl_engine"] is not None else "—"
        lines.append(
            f"| {r['tag']} | {r['rank'] if r['rank'] is not None else '—（天花板）'} | "
            f"{f_extra} | {w_extra} | {r['train_val_ppl']:.2f} | {r['ppl_torch']:.2f} | "
            f"{eng} | {dppl:+.2f}% | {rec_s} |"
        )
        lines.append("")
    return "\n".join(lines)


def render_ascii(rows: list[dict]) -> str:
    """按协议分组：ppl（torch 口径）随 r 的变化，越低越好；条长按跨度归一。"""
    out: list[str] = []
    for proto in sorted({r["protocol"] for r in rows}):
        qat = [r for r in rows if r["protocol"] == proto and r["qat"]]
        if not qat:
            continue
        ppls = [r["ppl_torch"] for r in qat]
        lo, hi = min(ppls), max(ppls)
        span = max(hi - lo, 1e-9)
        out += [
            "<pre>",
            f"[{proto}] val_ppl（torch 口径）随 HGF rank r 的变化（条长 ∝ ppl，越短越好）",
        ]
        for r in sorted(qat, key=lambda x: x["rank"]):
            frac = (r["ppl_torch"] - lo) / span
            bar = "█" * max(1, int(round(28 * frac)))
            out.append(f"r={r['rank']:<3} {bar:<28} ppl={r['ppl_torch']:.2f}")
        out.append("</pre>")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phosphorus T5-C6 HGF Pareto 扫描")
    ap.add_argument("--ranks", default="0,1,2,4,8,16")
    ap.add_argument(
        "--protocol",
        default="both",
        choices=["tiny", "d512", "both"],
        help="训练协议：tiny=d192/3 层（同 tiny.phos）；d512=PRD 的 15%% 口径；both=两者",
    )
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--corpus", default="data/raw/tinyshakespeare.txt")
    ap.add_argument("--out-dir", default="benchmarks")
    ap.add_argument("--ckpt-dir", default="ckpt/pareto")
    ap.add_argument("--no-fp32", action="store_true", help="跳过 fp32 天花板基线")
    args = ap.parse_args()

    protocols = ["tiny", "d512"] if args.protocol == "both" else [args.protocol]
    ranks = [int(x) for x in args.ranks.split(",")]
    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global PPL_SLICE
    tok = train_from_corpus(corpus, 4096)
    PPL_SLICE = tok.encode(corpus.read_text(encoding="utf-8", errors="ignore")[:200_000])[
        :PPL_TOKENS
    ]

    rows: list[dict] = []
    for proto in protocols:
        if not args.no_fp32:
            print(f"\n===== [{proto}] fp32 天花板基线 =====")
            rows.append(train_one(proto, None, args.steps, corpus, tok, ckpt_dir, PPL_SLICE))
        for r in ranks:
            print(f"\n===== [{proto}] hgf_rank={r} =====")
            rows.append(train_one(proto, r, args.steps, corpus, tok, ckpt_dir, PPL_SLICE))

    table = render_table(rows)
    print("\n" + table)
    print()
    print(render_ascii(rows))

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": str(corpus),
            "steps": args.steps,
            "protocols": {k: v for k, v in PROTOCOLS.items() if k in protocols},
            "ppl_eval": (
                f"torch 口径：全部档位 torch 前向 teacher-forcing（{PPL_TOKENS} token 切片，"
                f"block {PPL_BLOCK}，in-domain 未切 held-out）；"
                "引擎口径：.phos 导出后 PhosModel fp32 前向（部署路径）"
            ),
            "byte_ratio_formula": "HGF/三值权重% = 48·r·(in+out) / (5·in·out)（f16 因子 vs 5-bit 打包）；d=512 时 r=4 ≈ 15%",
        },
        "rows": rows,
    }
    (out_dir / "pareto.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pareto.md").write_text(
        "# HGF Pareto（T5-C6）\n\n"
        f"- 生成时间：{payload['meta']['generated_at']}；steps={args.steps}；"
        "协议：tiny（d=192/3 层，同 ckpt/tiny.phos）+ d512（d=512/8 层，PRD 的 15% 口径）\n"
        "- val_ppl 双口径：**torch 口径**（全部档位同协议，跨档可比，恢复率与 Δppl 用它）；"
        "**引擎口径**（.phos 导出后实测，部署路径；fp32 天花板无法导出 .phos 故只有 torch 口径）\n"
        "- 字节双口径：**文件增量%**（.phos 文件相对 r=0）；**HGF/三值权重%**"
        "（HGF f16 因子字节 ÷ 三值 5-bit 打包字节 = 48·r·(in+out)/(5·in·out)；"
        "d=512 时 r=4 ≈ 15%，d=192 时 r=4 ≈ 36%）\n"
        "- 质量损失恢复% = (ppl(r=0) − ppl(r)) / (ppl(r=0) − ppl(fp32)) × 100%（torch 口径）\n\n"
        + table
        + "\n\n"
        + render_ascii(rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n产物：{out_dir / 'pareto.json'}  {out_dir / 'pareto.md'}")


if __name__ == "__main__":
    main()
