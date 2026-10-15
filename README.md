# Phosphorus（晨星）

> 三值原生（BitNet b1.58 / **1.58-bit**）LLM 的**训练 + 投机解码推理引擎**
> 单进程 CPU 可跑 · 零 GPU 依赖 · **零系统编译器依赖** · 可量化评测

[![CI](https://github.com/CJX0712/phosphorus/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/phosphorus/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org)
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

---

## 仓库结构（节选）

```
phosphorus/
├── phosphorus/         扁平包（T1–T3 已落地）
│   ├── tokenizer.py    BPE（Sennrich 2016）+ 字节级映射，CJK/emoji 无损往返
│   ├── data.py         语料加载（离线兜底）+ memmap token 数据集
│   ├── model.py        LLaMA 变体：Pre-RMSNorm / RoPE / GQA / ReLU² / SubLN + 三值 QAT + HGF
│   ├── train.py        AdamW + cosine + grad clip + 导出 .phos
│   ├── packing.py      BitNet absmean 三值量化 + 5-bit 打包
│   ├── gemm.py         GEMM 后端：TL(LUT) / I2_S / int32 参考
│   ├── export.py       .phos 字节级导出
│   ├── loader.py       .phos 加载（运行时零 torch）
│   └── engine.py       纯 NumPy 推理（KV cache / 采样）
├── csrc/phos.h         C99 原生内核 ABI 草案（T4）
├── scripts/            setup.sh / setup.ps1 / smoke_zig.c
└── docs/               ARCHITECTURE / TASKS / ALGORITHMS / BENCHMARKS
```

## 设计文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 总体架构、`.phos` 字节级格式、内核算法、7 条核心不变量 |
| [`docs/TASKS.md`](docs/TASKS.md) | T1–T5 任务分解与验收标准 |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | A-1…A-24 算法溯源（来源 / 选型理由 / 我们的超越点） |
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
