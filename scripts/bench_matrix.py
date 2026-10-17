"""T5 · C5 全基准矩阵：{fp32, int8, tl, tl+spec} × batch ∈ {1,4,8} × seq ∈ {128,512}。

用法：
    python scripts/bench_matrix.py                       # 全矩阵（tiny.phos + synthetic_12l）
    python scripts/bench_matrix.py --model ckpt/tiny.phos --quick

产物（`--out-dir`，默认 `benchmarks/`）：
    results.json   全量原始数据（含环境、配置、跳过原因）
    REPORT.md      Markdown 表 + ASCII 柱状图 + 方法论说明

口径（诚实性优先，全部写入 REPORT.md）：
    * **seq = 序列总长**（prompt + 生成）。若按「prompt 长度」语义，则
      prompt+生成 > 全部可用模型的 max_seq_len（tiny.phos 256 /
      synthetic_12l 512），`forward` 会静默滑窗截断致指标失真，故不采用。
    * max_seq_len 不足的组记 `skipped` 并给出原因，**不跑假数据**。
    * 投机解码（tl+spec）当前仅支持单序列；batch>1 的 spec 组记 `skipped`
      （组合为后续工作，T5 范围外）。
    * 全部组 greedy（temperature=0）解码，确定性可复现。
    * 每组重复 `--repeats`（默认 3）次取**中位数**（共享机器时序噪声实测抖动可达
      ±35%，单次采样不可信）；原始计时全量写入 results.json。
    * rss_mb_peak = 进程峰值 RSS（`getrusage` 高水位，含解释器 + numpy +
      权重常驻），同模型不同 backend 差异主要来自反量化缓存。
    * ppl = teacher-forcing 困惑度（128 token 块，块内 reset cache）；
      无 tokenizer 的合成模型记 null（随机 token 上 ppl 无意义）。
      spec 行复用同 backend 的 ppl（同一份权重，不变量 ② 分布等价）。
    * speedup = 该组 tokens/s ÷ 同 (batch, seq) 的 fp32 基线 tokens/s。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phosphorus.engine import PhosModel  # noqa: E402
from phosphorus.scheduler import Scheduler  # noqa: E402
from phosphorus.speculate import LayerSkipDrafter, generate_tokens  # noqa: E402

# (backend, spec) —— "ternary" 部署态取 tl（i2s 与 tl int32 域逐位相等，tl 为代表）
CONFIGS = [("fp32", False), ("int8", False), ("tl", False), ("tl", True)]
BATCHES = (1, 4, 8)
SEQS = (128, 512)
SPEC_K = 8  # 与 BENCHMARKS.md §5.6 矩阵一致
_REPEATS = 3  # 每组重复次数（取中位数，抗共享机器时序噪声；原始计时全量入 results.json）
PPL_BLOCK = 128
PPL_TOKENS = 2048


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _peak_rss_mb() -> float:
    """进程峰值 RSS（MB）。Windows 用 psapi GetProcessMemoryInfo，POSIX 用 getrusage。"""
    if platform.system() == "Windows":
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        # 句柄必须按指针宽返回（默认 c_int 会截断 64 位句柄 → 结构体全 0）
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo 失败")
        return counters.PeakWorkingSetSize / 1e6
    import resource  # POSIX only

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    return ru / 1e3


class _StreamRecorder:
    """假 tokenizer：记录首次 decode 调用时刻 = 首个 token 产出时刻（ttft 探针）。"""

    def __init__(self) -> None:
        self.t_first: float | None = None

    def decode(self, ids) -> str:  # noqa: D102
        if self.t_first is None:
            self.t_first = time.perf_counter()
        return ""


def _env_info() -> dict:
    import phosphorus

    try:
        from phosphorus.backend import native_status

        native_ok, native_reason = native_status()
    except Exception as e:  # noqa: BLE001
        native_ok, native_reason = False, f"探测失败: {e}"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "phosphorus": phosphorus.__version__,
        "platform": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "cpu_count": __import__("os").cpu_count(),
        "native_kernels": native_ok,
        "native_reason": native_reason,
    }


# --------------------------------------------------------------------------- #
# prompt 构造
# --------------------------------------------------------------------------- #
def build_prompts(model: PhosModel, n: int, prompt_len: int, seed: int = 1234) -> list[list[int]]:
    """n 条互不相同的 prompt（各 prompt_len 个 token）。

    有 tokenizer 的模型取语料不重叠切片；合成模型用固定种子伪随机 id
    （ppl 无意义但吞吐口径一致）。
    """
    if model.tok is not None:
        corpus = Path(__file__).resolve().parent.parent / "data" / "raw" / "tinyshakespeare.txt"
        text = corpus.read_text(encoding="utf-8")[:200_000]
        toks = model.tok.encode(text)
        prompts = []
        for i in range(n):
            s = (i * 7919) % max(1, len(toks) - prompt_len - 1)
            prompts.append(list(toks[s : s + prompt_len]))
        return prompts
    rng = np.random.default_rng(seed)
    return [list(rng.integers(0, model.vocab, size=prompt_len)) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 单组运行
# --------------------------------------------------------------------------- #
def run_single(model: PhosModel, prompt_ids: list[int], n_new: int, spec: bool, dl: int):
    """batch=1：走 generate_tokens（串行 = drafter=None；投机 = LayerSkip 自草稿）。

    返回 (n_tok, alpha, [(seconds, ttft_ms), ...])——每个 repeat 一条原始计时。
    """
    drafter = draft_model = None
    if spec:
        drafter = LayerSkipDrafter(model, draft_layers=dl)
        draft_model = model.shadow()
    timings: list[tuple[float, float]] = []
    alpha = None
    n_tok = 0
    for _ in range(_REPEATS):
        rec = _StreamRecorder()
        t0 = time.perf_counter()
        gen, alpha, _info = generate_tokens(
            model,
            prompt_ids,
            max_new_tokens=n_new,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            drafter=drafter,
            K=SPEC_K,
            draft_model=draft_model,
            stream=True,
            tok=rec,
        )
        dt = time.perf_counter() - t0
        ttft = (rec.t_first - t0) * 1000.0 if rec.t_first is not None else None
        timings.append((dt, ttft if ttft is not None else 0.0))
        n_tok = len(gen)
    return n_tok, (alpha if spec else None), timings


def run_batch(model: PhosModel, prompts: list[list[int]], n_new: int):
    """batch>1：走 Scheduler 连续批处理；ttft = 首次 forward_step（prefill）完成时刻。

    返回 (n_tok, None, [(seconds, ttft_ms), ...])——每个 repeat 一条原始计时。
    """
    sched = Scheduler(model, max_seqs=len(prompts))
    orig_fs = model.forward_step
    timings: list[tuple[float, float]] = []
    n_tok = 0
    for _ in range(_REPEATS):
        state: dict[str, float] = {}

        def wrapped(seq_tokens, _state=state):  # 默认参数绑定，规避 B023 循环变量捕获
            out = orig_fs(seq_tokens)
            if "t_first" not in _state:
                _state["t_first"] = time.perf_counter()
            return out

        model.forward_step = wrapped
        try:
            t0 = time.perf_counter()
            gen = sched.generate_ids(prompts, max_new_tokens=n_new, temperature=0.0)
            dt = time.perf_counter() - t0
        finally:
            model.forward_step = orig_fs
        ttft = (state["t_first"] - t0) * 1000.0 if "t_first" in state else 0.0
        timings.append((dt, ttft))
        n_tok = sum(len(g) for g in gen)
    return n_tok, None, timings


def eval_ppl(model: PhosModel, tokens: list[int]) -> float:
    """teacher-forcing 困惑度：每块 reset cache，块内 [B] token 预测后 [B] 个。"""
    nll, cnt = 0.0, 0
    for i in range(0, len(tokens) - PPL_BLOCK - 1, PPL_BLOCK):
        ctx = np.asarray(tokens[i : i + PPL_BLOCK], dtype=np.int64)
        tgt = np.asarray(tokens[i + 1 : i + PPL_BLOCK], dtype=np.int64)  # 127 个，对齐 logits[:-1]
        model.reset()
        logits = model.forward(ctx)[:-1]
        m = logits.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        nll += float((lse[..., 0] - logits[np.arange(len(tgt)), tgt]).sum())
        cnt += len(tgt)
    return float(np.exp(nll / max(cnt, 1)))


# --------------------------------------------------------------------------- #
# 矩阵
# --------------------------------------------------------------------------- #
def run_matrix(model_path: str, quick: bool, n_new: int) -> list[dict]:
    # 每个 (path, backend) 只建一个模型，max_seqs=8（池按 8 分配，batch 子集复用）
    models: dict[str, PhosModel] = {}
    rows: list[dict] = []
    baseline: dict[tuple[int, int], float] = {}  # (batch, seq) -> fp32 tokens/s
    ppl_cache: dict[str, float | None] = {}  # backend -> ppl（spec 复用 tl）

    probe = PhosModel.from_file(model_path, backend="fp32", max_seqs=8)
    has_tok = probe.tok is not None
    ppl_tokens = None
    if has_tok:
        corpus = Path(__file__).resolve().parent.parent / "data" / "raw" / "tinyshakespeare.txt"
        ppl_tokens = probe.tok.encode(corpus.read_text(encoding="utf-8")[:60_000])[:PPL_TOKENS]

    seqs = (128,) if quick else SEQS
    batches = (1,) if quick else BATCHES

    for backend, spec in CONFIGS:
        key = backend
        if key not in models:
            models[key] = PhosModel.from_file(model_path, backend=backend, max_seqs=8)
        model = models[key]
        # ppl 每 backend 只测一次（spec 与同 backend 同权重，分布等价）
        if key not in ppl_cache:
            ppl_cache[key] = eval_ppl(model, ppl_tokens) if ppl_tokens is not None else None
        dl = max(1, model.n_layer - 1)

        for batch in batches:
            for seq in seqs:
                tag = f"{Path(model_path).stem} | {backend}{'+spec' if spec else ''} | b={batch} | seq={seq}"
                row = {
                    "model": Path(model_path).name,
                    "config": f"{backend}+spec" if spec else backend,
                    "backend": backend,
                    "spec": spec,
                    "batch": batch,
                    "seq": seq,
                    "n_new": n_new,
                    "draft_layers": dl if spec else None,
                    "K": SPEC_K if spec else None,
                    "status": "ok",
                    "reason": None,
                }
                # ---- 诚实跳过：容量不足 / 组合不支持 ----
                if model.max_seq_len < seq:
                    row["status"] = "skipped"
                    row["reason"] = (
                        f"max_seq_len={model.max_seq_len} < seq={seq}"
                        f"（prompt={seq - n_new}+生成 {n_new} 超出窗口，forward 将静默滑窗截断）"
                    )
                    rows.append(row)
                    print(f"[skip] {tag}: {row['reason']}")
                    continue
                if spec and batch > 1:
                    row["status"] = "skipped"
                    row["reason"] = "投机解码当前仅支持单序列（T5 范围）；batch+spec 组合为后续工作"
                    rows.append(row)
                    print(f"[skip] {tag}: {row['reason']}")
                    continue

                prompts = build_prompts(model, batch, seq - n_new)
                if batch == 1:
                    n_tok, alpha, timings = run_single(model, prompts[0], n_new, spec, dl)
                else:
                    n_tok, alpha, timings = run_batch(model, prompts, n_new)
                # 中位数聚合（抗共享机器噪声抖动）；原始计时全部进 results.json
                rate = float(np.median([n_tok / max(dt, 1e-9) for dt, _ in timings]))
                decode_rate = float(
                    np.median([n_tok / max(dt - tf / 1000.0, 1e-9) for dt, tf in timings])
                )
                ttft = float(np.median([tf for _, tf in timings]))
                sec = float(np.median([dt for dt, _ in timings]))
                row.update(
                    {
                        "tokens_generated": n_tok,
                        "seconds": round(sec, 4),
                        "repeats": _REPEATS,
                        "raw_timings": [
                            {"seconds": round(d, 4), "ttft_ms": round(t, 2)} for d, t in timings
                        ],
                        "tokens_per_s": round(rate, 2),
                        "decode_tokens_per_s": round(decode_rate, 2),
                        "ms_per_token": round(1000.0 / max(rate, 1e-9), 3),
                        "ttft_ms": round(ttft, 2),
                        "rss_mb_peak": round(_peak_rss_mb(), 1),
                        "ppl": ppl_cache[key],
                        "alpha": round(alpha, 4) if alpha is not None else None,
                    }
                )
                if (batch, seq) in baseline:
                    row["speedup_vs_fp32"] = round(rate / baseline[(batch, seq)], 3)
                else:
                    row["speedup_vs_fp32"] = 1.0
                    baseline[(batch, seq)] = rate
                rows.append(row)
                ppl_s = f"{row['ppl']:.2f}" if row["ppl"] is not None else "—"
                a = f" α={row['alpha']:.3f}" if row["alpha"] is not None else ""
                print(
                    f"[ok] {tag}: {rate:8.1f} tok/s  ms/tok={row['ms_per_token']:7.2f}  "
                    f"ttft={row['ttft_ms']:7.1f}ms  ppl={ppl_s}{a}"
                )
    return rows


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _bar(value: float, vmax: float, width: int = 28) -> str:
    n = 0 if vmax <= 0 else max(1, int(round(width * value / vmax)))
    return "█" * n


def render_report(rows: list[dict], env: dict, args) -> str:
    lines: list[str] = []
    ap = lines.append
    ap("# 基准矩阵报告（T5-C5）")
    ap("")
    ap(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    ap(f"- 环境：{env['phosphorus']} | python {env['python']} | numpy {env['numpy']}")
    ap(f"- 平台：{env['platform']} | cpu_count={env['cpu_count']}")
    native_line = (
        "✅ 可用"
        if env.get("native_kernels")
        else f"❌ 不可用（{env.get('native_reason', '未知（旧版 results.json 未记录）')}）"
        "→ **tl/int8 走 numpy 回退路径**；安装 ziglang 后 tl 预期快 2.8–6.8×（BENCHMARKS.md §4）"
    )
    ap(f"- native C99 内核：{native_line}")
    ap(
        f"- 口径：**seq = 序列总长**（prompt {SEQS[0]}-n_new / {SEQS[1]}-n_new + 生成 {args.n_new}）；"
        f"greedy（temperature=0）确定性解码；speedup = 相对同 (batch, seq) fp32 基线"
    )
    ap(f"- 投机配置：K={SPEC_K}，draft_layers = n_layer-1（O1 自草稿，extra_bytes=0）")
    ap(
        "- rss_mb_peak = 进程峰值 RSS（Windows: psapi `PeakWorkingSetSize`；POSIX: `getrusage`"
        " 高水位；含解释器 + numpy + 权重常驻）"
    )
    ap(
        "- ppl = teacher-forcing 困惑度；无 tokenizer 的合成模型记 null；spec 行复用同 backend"
        " ppl（同一份权重，不变量 ② 分布等价）"
    )
    ap("")

    for model_name in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model_name]
        ap(f"## {model_name}")
        ap("")
        for seq in sorted({r["seq"] for r in sub}):
            srows = [r for r in sub if r["seq"] == seq]
            ap(f"### seq={seq}（总长）")
            ap("")
            ap(
                "| 配置 | batch | 状态 | tokens/s（端到端） | tok/s（解码相） | ms/token | ttft_ms | rss_mb_peak | ppl | α | speedup |"
            )
            ap("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(srows, key=lambda x: (x["config"], x["batch"])):
                if r["status"] != "ok":
                    ap(
                        f"| {r['config']} | {r['batch']} | skipped | — | — | — | — | — | — | — | — |"
                    )
                else:
                    ppl_s = f"{r['ppl']:.2f}" if r["ppl"] is not None else "—"
                    ap(
                        f"| {r['config']} | {r['batch']} | ok | {r['tokens_per_s']:.1f} | "
                        f"{r['decode_tokens_per_s']:.1f} | {r['ms_per_token']:.2f} | "
                        f"{r['ttft_ms']:.1f} | {r['rss_mb_peak']:.0f} | "
                        f"{ppl_s} | {r['alpha'] if r['alpha'] is not None else '—'} | "
                        f"{r['speedup_vs_fp32']:.2f}× |"
                    )
            ap("")
            # ASCII 柱状图（仅 ok 行）
            ok = [r for r in srows if r["status"] == "ok"]
            if ok:
                vmax = max(r["tokens_per_s"] for r in ok)
                ap("<pre>")
                ap(f"端到端 tokens/s（seq={seq}，柱长按峰值归一）")
                for r in sorted(ok, key=lambda x: -x["tokens_per_s"]):
                    label = f"{r['config']:>9} b={r['batch']}"
                    ap(f"{label}  {_bar(r['tokens_per_s'], vmax)} {r['tokens_per_s']:8.1f}")
                ap("")
                ap(f"解码相 tokens/s（剔除 ttft，seq={seq}）")
                for r in sorted(ok, key=lambda x: -x["decode_tokens_per_s"]):
                    label = f"{r['config']:>9} b={r['batch']}"
                    ap(
                        f"{label}  {_bar(r['decode_tokens_per_s'], vmax)} {r['decode_tokens_per_s']:8.1f}"
                    )
                ap("</pre>")
                ap("")
        # 跳过原因
        skipped = [r for r in sub if r["status"] != "ok"]
        if skipped:
            ap("### 跳过组（含原因）")
            ap("")
            ap("| 配置 | batch | seq | 原因 |")
            ap("|---|---|---|---|")
            for r in skipped:
                ap(f"| {r['config']} | {r['batch']} | {r['seq']} | {r['reason']} |")
            ap("")
    ap("## 方法论与诚实性说明")
    ap("")
    ap("1. **seq 口径**：`seq` 为序列总长（prompt + 生成）。若按「prompt 长度」语义，")
    ap(f"   prompt+生成 = {SEQS[1]}+{args.n_new} 超出全部可用模型 max_seq_len")
    ap("   （tiny.phos=256 / synthetic_12l=512），`forward` 的滑窗守卫会静默截断，")
    ap("   指标即失真——故矩阵不采用该口径，容量不足组如实 skipped。")
    ap("2. **batch+spec 未组合**：`generate_tokens` 为单序列实现；batch>1 的 spec 组")
    ap("   记 skipped（多草稿树 + 连续批处理的组合在 T5 范围外，属后续工作）。")
    ap("3. **ppl 口径**：teacher-forcing 分块困惑度（块内 reset cache），非生成式 ppl；")
    ap("   合成模型（无 tokenizer）记 null——随机 token 上的 ppl 不具解释意义。")
    ap("4. **rss_mb_peak 为进程级高水位**：同进程多次运行取 `ru_maxrss`（Windows 为")
    ap("   psapi `PeakWorkingSetSize`）峰值，含 Python 解释器 + numpy + 模型权重常驻；")
    ap("   跨 backend 差异主要来自反量化缓存（fp32/int8 解包 int8，tl 保留 5-bit 打包字节）。")
    ap("5. **解码相 tok/s**：`n_new / (总时长 − ttft)`，由实测时刻差推导（非估算）。")
    ap(f"   n_new={args.n_new} 时端到端吞吐对微模型属**预填主导区**（spec 还需双 cache")
    ap("   预填），解码相口径才能反映投机机制的真实收益区间——长生成场景见")
    ap("   BENCHMARKS.md §5.6（n_new=256，解码主导）。")
    ap("6. **中位数聚合**：每组重复 N 次取中位数（本机实测单次采样抖动可达 ±35%，")
    ap("   共享/虚拟化 Windows 上时序噪声不可忽视）；raw_timings 全量在 results.json。")
    ap("7. **无任何估算/外推数字**：未运行的组一律 skipped + 原因。")
    ap("")
    return "\n".join(lines)


def main() -> None:
    global _REPEATS
    ap = argparse.ArgumentParser(description="Phosphorus T5-C5 全基准矩阵")
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help="模型路径（可多次）；默认 tiny.phos + synthetic_12l.phos",
    )
    ap.add_argument("--out-dir", default="benchmarks")
    ap.add_argument("--n-new", type=int, default=32, help="每组生成 token 数（默认 32）")
    ap.add_argument(
        "--repeats",
        type=int,
        default=_REPEATS,
        help=f"每组重复次数，取中位数（默认 {_REPEATS}；原始计时全量入 results.json）",
    )
    ap.add_argument("--quick", action="store_true", help="只跑 seq=128, batch=1")
    ap.add_argument(
        "--report-only",
        default=None,
        metavar="RESULTS_JSON",
        help="不重跑，从既有 results.json 重新生成 REPORT.md（报告模板升级后用）",
    )
    args = ap.parse_args()

    _REPEATS = max(1, int(args.repeats))

    models = args.model or ["ckpt/tiny.phos", "ckpt/synthetic_12l.phos"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        payload = json.loads(Path(args.report_only).read_text(encoding="utf-8"))
        report = render_report(payload["rows"], payload["meta"]["env"], args)
        (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(f"已重新生成报告：{out_dir / 'REPORT.md'}（数据不动，仅模板刷新）")
        return

    env = _env_info()
    all_rows: list[dict] = []
    for mp in models:
        if not Path(mp).exists():
            print(f"[skip] 模型不存在: {mp}")
            all_rows.append(
                {
                    "model": Path(mp).name,
                    "config": "-",
                    "backend": "-",
                    "spec": False,
                    "batch": "-",
                    "seq": "-",
                    "n_new": args.n_new,
                    "status": "skipped",
                    "reason": f"模型文件不存在: {mp}",
                    "draft_layers": None,
                    "K": None,
                }
            )
            continue
        print(f"\n===== {mp} =====")
        all_rows.extend(run_matrix(mp, args.quick, args.n_new))

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env": env,
            "args": {
                "models": models,
                "n_new": args.n_new,
                "quick": args.quick,
                "repeats": _REPEATS,
                "spec_K": SPEC_K,
                "seq_semantics": "total_length(prompt+n_new)",
            },
        },
        "rows": all_rows,
    }
    rj = out_dir / "results.json"
    rj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(all_rows, env, args)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"\n完成：{n_ok}/{len(all_rows)} 组 ok，{len(all_rows) - n_ok} 组 skipped")
    print(f"产物：{rj}  {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
