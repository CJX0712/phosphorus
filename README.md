# Phosphorus（晨星）

> 三值原生（BitNet b1.58 / **1.58-bit**）LLM 的**训练 + 投机解码推理引擎**
> 单进程 CPU 可跑 · 零 GPU 依赖 · **零系统编译器依赖** · 可量化评测

[![CI](https://github.com/CJX0712/phosphorus/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/phosphorus/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

Phosphorus 把「1.58-bit 三值 LLM」从论文搬到一台普通笔记本上：从 BPE 分词、CPU 训练、
三值 5-bit 打包导出 `.phos`、到纯 NumPy 推理与投机解码，全链路可离线、可单测、可复现。

---

## 5 分钟快速上手

```bash
# 0) 环境（pip 必须带 NO_PROXY='*'）
export NO_PROXY='*' no_proxy='*'
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[train,dev]"

# 1) 训练 BPE 分词器 + 编码语料（离线内置 tinyshakespeare 语料）
python -m phosphorus tokenize --action train --corpus data/raw/tinyshakespeare.txt --vocab 4096 --out artifacts/tokenizer.json

# 2) CPU 训练（~10 分钟），qat-hgf 模式直接导出三值 .phos
python -m phosphorus train --mode qat-hgf --corpus data/raw/tinyshakespeare.txt --vocab 4096 --steps 300 --out ckpt/tiny

# 3) 纯 NumPy 推理（不加载 torch）
python -m phosphorus generate --model ckpt/tiny.phos --prompt "To be, or not to be"

# （如需从 .pt 检查点重新导出 .phos：python -m phosphorus export --checkpoint ckpt/tiny.pt --tokenizer artifacts/tokenizer.json --out artifacts/tiny.phos）
```

只想推理？只装 core 分组即可（**不装 torch**）：

```bash
python -m pip install -e "."
python -c "import phosphorus.engine; print('no torch needed')"
```

想用原生 C99 内核（位精确，tl 后端 prefill 实测 2.7×）？加装 `native` 分组
（pip 安装 zig，**不需要系统 C 编译器**）：

```bash
python -m pip install -e ".[native]"
python -m phosphorus bench --model ckpt/tiny.phos --tokens 64
# 内核级 / 端到端基准（自动先校验 native 与 numpy 逐位相等）：
python -m scripts.bench_kernels --reps 5
python -m scripts.bench_native --model ckpt/tiny.phos --reps 7
```

> 未装 zig 时自动静默回退 NumPy 后端（结果不变，仅速度不同）。
> 实测数字与调度规则见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §4。

想跑连续批处理（多序列融合一次前向，吞吐 2.35× 串行）？用 `sched` 子命令：

```bash
# 一次融合 8 条 prompt，自动按 max-seqs 切分波次，--bench 打印吞吐
python -m phosphorus sched --model ckpt/tiny.phos \
    --prompts "To be, or not to be" "Once upon a time" "The king said" "In the beginning" \
              "It was the best" "She looked at" "All the world" "When the sun" \
    --max-seqs 8 --max_new_tokens 40 --temperature 0.0 --top_k 1 --bench
# 也支持 dense 后端逐序列与 paged 融合等价对照（T4-D 测试已守护）
```

> 连续批处理设计、等价性与吞吐实测见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §5.5 与
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.7。

想开投机解码（O1 自草稿 + Leviathan 验证，输出分布与目标**严格相等**）？用 `--speculate`：

```bash
# on：固定 K 强制开启；auto：O2 自适应控制器按接受率/成本自动开关与选 K*
python -m phosphorus generate --model ckpt/tiny.phos \
    --prompt "To be, or not to be" --max_new_tokens 128 \
    --speculate auto --K 8
# 实测（tiny.phos, K=8, draft_layers=2, α=0.819）：1.40× 串行吞吐；
# temperature=0 时投机与串行贪心 token 序列逐位相同（不变量 ②）
```

> 热 KV 增量解码（双 cache + 拒绝回滚）、加速比矩阵与 regime 分析见
> [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §5.6，算法与超越点见
> [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) A-12/A-15。

---

## 仓库结构（节选）

```
phosphorus/
├── phosphorus/         扁平包（T1–T3 已落地，T4-A/B 原生内核 + 分页 KV 已落地）
│   ├── tokenizer.py    BPE（Sennrich 2016）+ 字节级映射，CJK/emoji 无损往返
│   ├── data.py         语料加载（离线兜底）+ memmap token 数据集
│   ├── model.py        LLaMA 变体：Pre-RMSNorm / RoPE / GQA / ReLU² / SubLN + 三值 QAT + HGF
│   ├── train.py        AdamW + cosine + grad clip + 导出 .phos
│   ├── packing.py      BitNet absmean 三值量化 + 5-bit 打包
│   ├── gemm.py         GEMM 后端：TL(LUT) / I2_S / int32 参考
│   ├── export.py       .phos 字节级导出
│   ├── loader.py       .phos 加载（运行时零 torch）
│   ├── engine.py       推理引擎：NumPy + native 自适应调度 + 分页 KV（--kv dense|paged）+ 融合连续批处理 forward_step
│   ├── scheduler.py     连续批处理调度器 Scheduler：波次切分 + greedy/采样生成（C6/C7）
│   ├── speculate/       投机解码：Leviathan 验证 + O1 自草稿（零额外权重）+ O2 自适应控制器 + 树草稿
│   ├── eviction.py      KV 驱逐：SnapKV / PyramidKV / H2O / StreamingLLM（T4-C）
│   ├── backend.py      实现后端注册：native / numpy（auto 降级）
│   ├── block_manager.py 分页块管理：block_tables / 引用计数 / 滑动窗口（vLLM 语义）
│   ├── kv_cache.py     分页 KV 物理池 + begin/write/gather 三段式协议
│   ├── attention.py    GQA 注意力：paged / dense 同一数学核心
│   ├── csrc/           C99 原生内核源码（随 wheel 分发）
│   │   ├── phos.h      ABI（version 1，scalar/AVX2 flags）
│   │   ├── phos_gemm.c TL / I2_S GEMM（maddubs+madd，移植自 BitNet.cpp MIT）
│   │   ├── phos_pack.c 5-bit pack/unpack
│   │   └── phos_ops.c  rmsnorm / softmax / rope
│   └── native/         zig cc 构建 + ctypes 绑定 + 行块并行
│       ├── build.py    内容寻址缓存 + 文件锁 + 原子替换
│       ├── bindings.py NativeKernels（ctypes 原型）
│       └── parallel.py 行分块 + 线程池
├── scripts/            setup.sh / setup.ps1 / smoke_zig.c / bench_kernels.py / bench_native.py
├── NOTICE              第三方代码归属（BitNet.cpp I2_S SIMD 移植，MIT）
└── docs/               ARCHITECTURE / TASKS / ALGORITHMS / BENCHMARKS
```

## 设计文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 总体架构、`.phos` 字节级格式、内核算法、10 条核心不变量 |
| [`docs/TASKS.md`](docs/TASKS.md) | T1–T5 任务分解与验收标准 |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | A-1…A-25 算法溯源（来源 / 选型理由 / 我们的超越点 / 许可证） |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 全部实测数字（禁止写入未测量值） |

## 引用

```bibtex
@software{phosphorus2026,
  author = {晨星（CJX0712）},
  title  = {Phosphorus: 三值原生 LLM 的训练与投机解码推理引擎},
  year   = {2026},
  url    = {https://github.com/CJX0712/phosphorus},
  license= {Apache-2.0}
}
```

## License

Apache-2.0 © 晨星（CJX0712）
