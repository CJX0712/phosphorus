# Phosphorus 系统架构设计

> **Phosphorus（晨星）** — 三值原生（1.58-bit / BitNet b1.58）LLM 的**训练 + 投机解码推理引擎**
> 单进程 CPU 可跑 · 零 GPU 依赖 · 零系统编译器依赖 · 可量化评测
> 作者：晨星（GitHub: [CJX0712](https://github.com/CJX0712)） · License: Apache-2.0
> 文档版本：v1.0（对应 `.phos` 格式 version 1.0）

---

## 0. 设计总纲

### 0.1 三条不可违背的红线

| 红线 | 具体含义 | 落地手段 |
|---|---|---|
| **禁重复造轮子** | 凡有顶级现成方案（论文/开源项目）必须显式复用并注明来源 | `docs/ALGORITHMS.md` 每张表都要有「来源 + 链接 + 选型理由 + 我们的超越点」四列 |
| **最小可用优先** | 模型 / 语料 / 训练步数必须在纯 CPU 上几分钟跑完并给出真实数字 | 默认 `d_model=384, n_layer=6, vocab=8192`（≈10M 参数），`--scale {tiny,small,base}` 可放大 |
| **不变量优先于功能** | 每个模块先定义可验证的**不变量**，再写功能代码 | `tests/` 中 7 条核心不变量（见 §7），CI 必跑 |

### 0.2 四个原创超越点（O1–O4）

| ID | 名称 | 一句话 | 可量化交付 |
|---|---|---|---|
| **O1** | 三值原生自草稿投机解码 | 不训练第二个草稿模型，复用目标模型前 E 层 + 早退 head，且**草稿路径跑在同一份三值权重上** | 草稿模型额外参数 = 0（`--early-head-rank 0`），额外显存 = 0 |
| **O2** | 自适应投机控制器 | 在线测 α 与 batch，动态调 K / 树宽 / **是否开启**；低于阈值自动关闭 | `speedup=(1+αK)/(1+ε)` 闭式模型 + 在线拟合；给出「开启/关闭」决策日志 |
| **O3** | 质量-内存 Pareto 旋钮 | HGF 低秩校正秩 r ∈ {0,1,2,4,8,16}，输出 PPL vs 内存曲线 | `bench pareto` 输出 ASCII 曲线 + JSON |
| **O4** | 零编译器依赖的原生内核 | C99 内核用 `python -m ziglang cc` 编译（pip 可装，Win/Mac/Linux 通吃），失败自动回退 NumPy | `scripts/setup.sh` 无需 apt/msvc；`PHOS_BACKEND=auto` 三级回退 |

### 0.3 环境硬约束（已实测，设计必须遵守）

- 无 GPU：全部路径 CPU-only；模型规模 5M–30M 参数。
- **无系统 C 编译器**：MSVC / gcc 路径不可用。所有 C 代码经 `python -m ziglang cc -O2` 编译。
- Windows 开发，Linux/macOS 必须同样跑通（CI: `ubuntu-latest`）。
- `pip` 必须带 `NO_PROXY='*' no_proxy='*'`。
- Python venv：`C:/Users/Administrator/.workbuddy/binaries/python/envs/phos/Scripts/python.exe`（numpy 2.5.4 / pytest / ziglang 0.16.0；torch CPU 版）。
  - C 内核**不使用 OpenMP / pthread**（可移植性风险），并行由 Python `ThreadPoolExecutor` 在 row-block 粒度切分 + `ctypes` 释放 GIL 实现。

### 0.4 当前实现状态（v0.1.0 · 2026-10）

> 下表给出**规划架构（本文 mermaid + §2.2）**与**已落地代码**的对照。
> T1–T3 已交付为**扁平包** `phosphorus/`（单文件模块，无 `utils/`、`model/`、`quantize/`、`kernels/` 子包）；
> T4（原生 C99 内核）已交付 **T4-A**（GEMM + pack + 归一化/softmax/RoPE 内核）、
> **T4-B**（分页 KV + 块管理 + 引擎接入）、**T4-C**（KV 驱逐 SnapKV/PyramidKV/H2O/StreamingLLM）；
> **T4-D**（连续批处理调度器 + 融合前向 + 采样器）已交付：8 条序列融合前向 token id 与逐条串行逐位一致，吞吐 **2.35×**；
> **T4-E**（端到端生成 C8 + 文档收尾）已交付：`phosphorus generate` 连贯文本 + `--stream` 逐 token + `--bench` 首 token 延迟实测 4/18 ms（≪500ms），全量文档同步；
> **T5**（投机解码）核心已交付：O1 自草稿（零额外权重）+ Leviathan 验证（随机 + 贪心双模式）+
> **热 KV 增量解码**（目标/草稿双 cache + `truncate_tail` 拒绝回滚）+ O2 自适应控制器 + 树草稿掩码。
> C1 分布等价（χ² p>0.05 / KL<1e-3，20k 采样）与 C2 零额外字节已由测试硬门证明；C3/C5/C6 见 `BENCHMARKS.md` §T5。

| 规划模块 | 实际落地文件 | 状态 |
|---|---|---|
| `utils/*` | 内联进各模块 | ✅ 已吸收（T1） |
| `data/{corpus,dataset}` | `phosphorus/data.py` | ✅ T1 |
| `tokenizer/*` | `phosphorus/tokenizer.py` | ✅ T1（不变量⑤） |
| `model/{llama,quantization,hgf}` | `phosphorus/model.py` | ✅ T2 |
| `train/*` | `phosphorus/train.py` | ✅ T2/T3 |
| `quantize/{packing,export}` | `phosphorus/packing.py` / `phosphorus/export.py` | ✅ T3（不变量①③） |
| `kernels/numpy_backend/gemm` | `phosphorus/gemm.py` | ✅ T3（不变量①） |
| `engine/model_loader` | `phosphorus/loader.py` | ✅ T3 |
| `engine`（推理主循环） | `phosphorus/engine.py` | ✅ T3（不变量⑦）+ T4-A 自适应 native 调度 + T4-B 分页 KV + **T4-D 融合前向 `forward_step`** |
| 分页 KV / 块管理 / GQA 注意力 | `phosphorus/kv_cache.py`, `block_manager.py`, `attention.py` | ✅ T4-B（不变量⑨⑩） |
| `kernels/native/*` | `phosphorus/csrc/*.{c,h}` + `phosphorus/native/*` | ✅ T4-A（zig cc 编译，位精确） |
| 实现后端注册 | `phosphorus/backend.py` | ✅ T4-A |
| 连续批处理调度器 | `phosphorus/scheduler.py`（`Scheduler` + 波次切分 + greedy/采样生成） | ✅ T4-D（C6 等价 + C7 采样；融合 8 条吞吐 2.35×） |
| KV 驱逐策略 | `phosphorus/eviction.py`（SnapKV/PyramidKV/H2O/StreamingLLM + 统一保留掩码） | ✅ T4-C（不变量⑦） |
| `speculate/*`（投机解码） | `phosphorus/speculate/{verify,layerskip,ngram,controller,tree,eagle,medusa,base}.py` + `engine.truncate_tail/shadow` + `speculate.generate_tokens` 热 KV 主循环 | ✅ T5（C1★ 分布等价 + C2 零额外字节 + C4 控制器；C3/C5/C6 见 BENCHMARKS §T5） |
| `bench/*` | `scripts/bench_speculate.py`（吞吐矩阵）+ `scripts/build_synthetic_model.py`（regime 演示模型） | ✅ T5-C5/C6 |
| `convert/*` | HF 导入/导出 | 🔜 T5-C8（P1） |

---

## 1. 总体分层架构

```mermaid
graph TB
    subgraph L0["L0 · 工程与工具层"]
        CLI["cli.py<br/>train/quantize/export<br/>generate/bench/tokenize"]
        CFG["config.py<br/>ModelConfig / TrainConfig<br/>QuantConfig / DecodeConfig"]
        UTL["utils/<br/>seed · logging · memory · timer"]
    end

    subgraph L1["L1 · 数据层"]
        CRP["data/corpus.py<br/>下载 + 离线兜底 + 校验"]
        DS["data/dataset.py<br/>memmap token id · batch"]
    end

    subgraph L2["L2 · 分词层"]
        PTK["tokenizer/pretokenize.py<br/>GPT-2 正则"]
        BPE["tokenizer/bpe.py<br/>BPE 训练 Sennrich 2016"]
        TOK["tokenizer/tokenizer.py<br/>encode/decode · save/load"]
    end

    subgraph L3["L3 · 模型与训练层 (PyTorch, CPU)"]
        MDL["model/llama.py<br/>Pre-RMSNorm · RoPE · GQA<br/>SwiGLU/ReLU² · SubLN"]
        QAT["model/quantization.py<br/>absmean 三值 · STE · BitLinear"]
        HGF["model/hgf.py<br/>Hybrid Gated Flow 低秩校正"]
        TRN["train/trainer.py<br/>AdamW · cosine · clip · ckpt"]
    end

    subgraph L4["L4 · 量化导出层"]
        EXP["quantize/export.py<br/>torch ckpt → .phos"]
        PCK["quantize/packing.py<br/>24 权重/5 字节 pack"]
    end

    subgraph L5["L5 · 计算内核层"]
        REG["kernels/registry.py<br/>backend 选择 + 三级回退"]
        NPG["kernels/numpy_backend/<br/>fp32 · int8 · TL · I2_S"]
        NAT["kernels/native/<br/>C99 + ctypes (zig cc)"]
    end

    subgraph L6["L6 · 推理引擎层"]
        LD["engine/model_loader.py<br/>.phos → RuntimeWeights"]
        KVC["engine/kv_cache.py<br/>Paged KV (vLLM)"]
        BLK["engine/block_manager.py<br/>块分配 / 抢占"]
        EVC["engine/eviction.py<br/>SnapKV/PyramidKV/H2O"]
        SCH["engine/scheduler.py<br/>Continuous Batching"]
        ATT["engine/attention.py<br/>GQA + 页表 + 树掩码"]
        SMP["engine/sampler.py<br/>top-k / top-p / temp"]
        GEN["engine/generate.py<br/>greedy / sample / stream"]
    end

    subgraph L7["L7 · 投机解码层"]
        DRF["speculate/<br/>layerskip(O1) · ngram · eagle · medusa"]
        TRE["speculate/tree.py<br/>树状草稿 + 掩码"]
        VRF["speculate/verify.py<br/>Leviathan 投机采样"]
        CTL["speculate/controller.py<br/>O2 自适应控制器"]
    end

    subgraph L8["L8 · 评测与转换"]
        BCH["bench/<br/>runner · metrics · report · pareto"]
        CNV["convert/<br/>HF 导入/导出"]
    end

    CLI --> CFG
    CLI --> L1 & L2 & L3 & L4 & L6 & L7 & L8
    CRP --> DS --> TOK
    PTK --> BPE --> TOK
    DS --> TRN
    TOK --> MDL
    MDL --> QAT --> HGF --> TRN
    TRN --> EXP --> PCK
    PCK --> LD
    LD --> REG
    REG --> NPG
    REG --> NAT
    NPG --> ATT
    NAT --> ATT
    KVC --> BLK
    BLK --> EVC
    EVC --> ATT
    SCH --> BLK
    SCH --> ATT
    ATT --> GEN
    SMP --> GEN
    GEN --> DRF
    DRF --> TRE --> VRF --> CTL
    CTL --> DRF
    GEN --> BCH
    CNV --> LD
```

### 1.1 分层职责边界（铁律）

| 层 | 允许依赖 | 禁止依赖 |
|---|---|---|
| L0 工具 | stdlib | 任何上层 |
| L1/L2 数据分词 | numpy, stdlib | torch, 内核层 |
| L3 训练 | torch, numpy | 内核层、引擎层（训练期只用 torch 算子） |
| L4 导出 | torch(仅读 ckpt), numpy | 引擎层 |
| L5 内核 | numpy / ctypes | torch、引擎层 |
| L6 引擎 | L5, numpy | torch（**运行时零 torch 依赖**，保证 CPU 推理可独立安装） |
| L7 投机 | L6 | torch |
| L8 评测 | L6, L7, numpy | torch（除 HF 转换外） |

> **关键设计决策**：L6 运行时**不依赖 PyTorch**。装上 `core` 分组即可推理，装 `train` 分组才能训练。这让「零 GPU 依赖 + 几分钟跑完」成为可安装的现实。

---

## 2. 模块清单

### 2.1 完整文件树

```
phosphorus/
├── pyproject.toml                     # 可选依赖分组 core/train/native/bench/convert/dev
├── Makefile                           # install / test / lint / build-native / bench
├── README.md                          # 中文为主 + 徽章
├── LICENSE                            # Apache-2.0
├── CITATION.cff
├── .gitignore
├── ruff.toml
├── scripts/
│   ├── setup.sh                       # Linux/macOS: venv + pip(NO_PROXY='*') + 内核编译
│   ├── setup.ps1                      # Windows 等价
│   ├── smoke_zig.c                    # O4 zig cc 冒烟（T4 落地前提前验证）
│   ├── bench_kernels.py               # T4-A 内核级基准（含位一致性校验）
│   ├── bench_native.py                # T4-A 端到端基准（native vs numpy）
│   ├── measure_eviction.py            # T4-C 驱逐策略质量基准（下游 logits 余弦）
│   ├── bench_speculate.py             # T5 投机解码吞吐矩阵（C3/C5/C6）
│   └── build_synthetic_model.py       # T5 合成深层模型构造（加速比 regime 演示）
├── .github/workflows/ci.yml           # ubuntu-latest, py3.10/3.11/3.12
├── phosphorus/                        # 扁平包（T1–T5 已落地；无 model/ quantize/ kernels/ 子包）
│   ├── __init__.py                    # __version__, backend 探测
│   ├── __main__.py                    # python -m phosphorus
│   ├── cli.py                         # argparse 子命令：train/export/generate/bench/tokenize
│   ├── config.py                      # 全部 dataclass 配置 + JSON 序列化
│   ├── data.py                        # 语料 + 数据集（合并自 data/）
│   ├── tokenizer.py                   # BPE（合并自 tokenizer/）
│   ├── model.py                       # LLaMA 变体 + 三值 QAT + HGF（合并自 model/）
│   ├── train.py                       # trainer + 导出（合并自 train/）
│   ├── export.py                      # ckpt -> .phos（合并自 quantize/export）
│   ├── packing.py                     # 三值 5-bit pack/unpack（合并自 quantize/packing）
│   ├── gemm.py                        # TL / I2_S / int32 参考（合并自 kernels/numpy_backend/gemm）
│   ├── loader.py                      # .phos 加载（合并自 engine/model_loader）
│   ├── engine.py                      # 推理引擎（numpy + native 自适应调度 + 分页 KV）
│   ├── backend.py                     # 实现后端注册：native / numpy（T4-A）
│   ├── block_manager.py               # 分页块管理：block_tables / 引用计数 / 滑动窗口（T4-B）
│   ├── kv_cache.py                    # 分页 KV 物理池 [n_blocks, n_layer, 2, bs, n_kv, dh]（T4-B）
│   ├── attention.py                   # GQA 注意力：paged / dense 同一数学核心（T4-B）
│   ├── eviction.py                    # KV 驱逐策略：SnapKV / PyramidKV / H2O / StreamingLLM（T4-C）
│   ├── scheduler.py                   # 连续批处理调度器：add/abort + forward_step 融合（T4-D）
│   ├── sampling.py                    # top-k / top-p / temp
│   ├── speculate/                     # 投机解码子包（T5，扁平增量）
│   │   ├── __init__.py                # generate_tokens：热 KV 增量解码主循环（草稿+验证+回滚）
│   │   ├── base.py                    # Drafter / DrafterSpec 抽象
│   │   ├── verify.py                  # Leviathan 残差拒绝采样（随机 + 贪心双模式，不变量②）
│   │   ├── layerskip.py               # O1 自草稿（early-exit，extra_bytes==0）
│   │   ├── ngram.py                   # n-gram 草稿对照（零模型成本命中）
│   │   ├── controller.py              # O2 自适应控制器（EWMA α/c + 状态机 + K* 决策）
│   │   ├── tree.py                    # 树状草稿：祖先因果掩码 + 剪枝
│   │   ├── eagle.py                   # EAGLE 对照位（当前别名 LayerSkip，P1 扩展点）
│   │   └── medusa.py                  # Medusa 对照位（当前别名 LayerSkip，P1 扩展点）
│   ├── csrc/                          # 原生 C99 源码（随 wheel 分发）
│   │   ├── phos.h                     # 原生内核 ABI 头（ABI version 1）
│   │   ├── phos_gemm.c                # TL / I2_S GEMM（scalar + AVX2）
│   │   ├── phos_pack.c                # 5-bit pack/unpack
│   │   └── phos_ops.c                 # rmsnorm / softmax / rope
│   └── native/                        # 原生库构建 + ctypes 绑定（T4-A）
│       ├── __init__.py                # is_available / kernels / flags 导出
│       ├── build.py                   # zig cc 编译 + 内容寻址缓存 + 文件锁
│       ├── bindings.py                # NativeKernels（ctypes 原型 + 参数校验）
│       └── parallel.py                # 行分块 + 线程池（N 维并行）
├── tests/
│   ├── test_data.py                   # 数据集不变量
│   ├── test_gemm_bitwise.py           # 不变量 ①
│   ├── test_packing.py                # 不变量 ③
│   ├── test_tokenizer.py              # 不变量 ⑤
│   ├── test_qat_ste.py                # 不变量 ⑥
│   ├── test_loader.py                 # 三值 / HGF 往返 + 头字段
│   ├── test_engine.py                 # 不变量 ⑦
│   ├── test_hgf.py                    # HGF 低秩校正往返
│   ├── test_rope.py                   # 不变量 ④
│   ├── test_no_torch.py               # 推理路径零 torch 导入
│   ├── test_paged_kv.py               # T4-B：块管理 / 物理池 / 注意力对齐（20 条）
│   ├── test_engine_paged.py           # T4-B：引擎级 paged ↔ dense 等价（7 条）
│   ├── test_native_backend.py         # T4-A：native 内核 50 条断言（L0–L6）
│   ├── test_kv_eviction.py            # T4-C：驱逐策略预算/掩码/质量（不变量⑦）
│   ├── test_scheduler.py              # T4-D：连续批处理调度（add/abort/融合等价）
│   ├── test_speculative_distribution.py # T5-C1/C2：分布等价 χ²+KL / 贪心逐位一致 / O1 零额外字节
│   ├── test_controller.py             # T5-C4：O2 控制器状态机 / 门控 / K* 决策
│   └── test_tree_mask.py              # T5：树草稿因果掩码 / 剪枝不变量
├── data/raw/
│   └── tinyshakespeare.txt            # 离线兜底语料（随仓库提交，1.1 MB）
├── NOTICE                            # 第三方代码归属（BitNet.cpp I2_S SIMD 移植，MIT）
└── docs/
    ├── ARCHITECTURE.md                # 本文
    ├── ALGORITHMS.md                  # 借鉴来源 + 超越点
    ├── TASKS.md                       # 任务分解
    └── BENCHMARKS.md                  # 实测基准（含 T4-A native vs numpy）
```

### 2.2 模块职责 / 输入 / 输出 / 关键数据结构

| 模块（实际文件） | 职责 | 关键对象 / 函数 | 状态 |
|---|---|---|---|
| `phosphorus/data.py` | 语料获取（内置优先 + 下载兜底 + sha256 校验）；token id 滑窗数据集 | `Corpus`, `TokenDataset` | ✅ T1 |
| `phosphorus/tokenizer.py` | BPE 训练（Sennrich 2016）+ 字节级映射；encode/decode；256 字节种子保证 CJK/emoji 无损往返 | `Tokenizer`, `train_from_corpus` | ✅ T1（不变量⑤） |
| `phosphorus/config.py` | 全部 `dataclass` 配置 + JSON 序列化 | `ModelConfig`, `TrainConfig`, `QuantConfig`, `DecodeConfig`, `PhosphorusConfig` | ✅ T1 |
| `phosphorus/model.py` | LLaMA 变体（Pre-RMSNorm / RoPE / GQA / ReLU² / SubLN）+ absmean 三值 QAT-STE + HGF 低秩校正 | `LlamaModel`, `ternary_ste`, `BitLinear`, `HybridGatedFlow` | ✅ T2 |
| `phosphorus/train.py` | AdamW + cosine + grad clip + ckpt + 导出 `.phos` | `run_train`, `export_trained` | ✅ T2/T3 |
| `phosphorus/export.py` | ckpt → `.phos`（三值 5-bit 打包 + HGF 因子 + tokenizer + metadata） | `export_model`, `PhosWriter` | ✅ T3 |
| `phosphorus/packing.py` | 三值 5-bit pack/unpack（24 权重 → 5 字节，14 项 LUT） | `pack_ternary_v1`, `unpack_ternary_v1` | ✅ T3（不变量③） |
| `phosphorus/gemm.py` | TL / I2_S / int32 参考 GEMM（位精确） | `gemm_tl`, `gemm_i2s`, `int32_reference` | ✅ T3（不变量①） |
| `phosphorus/loader.py` | `.phos` 加载 → 运行时权重 + HGF + tokenizer | `load_phos`, `PhosFile` | ✅ T3 |
| `phosphorus/engine.py` | 推理引擎：prefill-decode / 采样 / RoPE / native 自适应调度 / 分页 KV | `PhosModel`, `_KVCache`, `PagedKVCache`, `paged_gqa_attention` | ✅ T3（不变量⑦）+ T4-A + T4-B（不变量⑨） |
| `phosphorus/block_manager.py` | 分页块管理（vLLM 语义）：block_tables / 引用计数 / 滑动窗口 / 使用率 | `BlockManager` | ✅ T4-B（不变量⑩） |
| `phosphorus/kv_cache.py` | 分页 KV 物理池 `[n_blocks, n_layer, 2, bs, n_kv, dh]` + begin/write/gather 协议 | `PagedKVCache` | ✅ T4-B |
| `phosphorus/attention.py` | GQA 注意力：paged 与 dense 共用同一数学核心 | `gqa_attention`, `paged_gqa_attention` | ✅ T4-B（不变量⑨） |
| `phosphorus/sampling.py` | temperature / top-k / top-p 采样 | `sample` | ✅ T3 |
| `phosphorus/cli.py` | 子命令分发：`train / export / generate / bench / tokenize`（`--kv`/`--backend`） | `build_parser`, `main` | ✅ T1 + T4-B |
| `phosphorus/csrc/phos.h` | 原生 C99 内核 ABI 头（TL GEMM / RMSNorm / softmax / RoPE / pack） | `phos_tl_gemm_f32` 等 | ✅ T4-A（随 wheel 分发） |
| `phosphorus/eviction.py` | KV 驱逐策略（SnapKV/PyramidKV/H2O/StreamingLLM，逐层预算 + 统一保留 sink+窗口） | `make_policy`, `EvictionConfig`, `EVICTIONS` | ✅ T4-C（不变量⑦） |
| `phosphorus/scheduler.py` | 连续批处理调度：多 prompt 波次切分 + greedy/采样生成 | `Scheduler` | ✅ T4-D（吞吐 2.35×） |
| `phosphorus/speculate/verify.py` | Leviathan 残差拒绝采样（随机模式分布严格等价；贪心模式逐位一致） | `spec_verify`, `VerifyResult` | ✅ T5-C1★（不变量②） |
| `phosphorus/speculate/layerskip.py` | O1 自草稿：early-exit 复用同一权重，extra_bytes==0 | `LayerSkipDrafter` | ✅ T5-C2 |
| `phosphorus/speculate/ngram.py` | order-3 n-gram 草稿对照（命中零模型成本） | `NGramDrafter` | ✅ T5 |
| `phosphorus/speculate/controller.py` | O2 自适应控制器：EWMA α/c + OFF/WARMUP/ON/DEGRADED 状态机 + K* 决策 | `SpecController`, `SpecPlan` | ✅ T5-C4 |
| `phosphorus/speculate/tree.py` | 树状草稿：平衡树构建 + 祖先因果掩码 + 概率剪枝 | `TreeSpec`, `build_balanced_tree` | ✅ T5 |
| `phosphorus/speculate/__init__.py` | 热 KV 增量投机主循环：双 cache 草稿/验证 + `truncate_tail` 拒绝回滚 | `generate_tokens` | ✅ T5-C3 机制 |
| `scripts/bench_speculate.py` | 投机解码吞吐矩阵（K × draft_layers × seeds，含接受率 α） | — | ✅ T5-C5/C6 |
| `convert/*` | HF 导入/导出 | — | 🔜 T5-C8（P1） |
| `convert.hf_*` | HF `transformers` ↔ `.phos` / ckpt 互转 | safetensors | 目标格式 | `import_hf()`, `export_hf()` |

---

## 3. `.phos` 二进制模型格式（version 1.0）

**字节序：小端（LE）**。**对齐：64 字节**。**所有偏移均为距文件开头的绝对字节偏移**。整数用定宽 `u32/u64`，浮点用 IEEE-754 `f32`。

### 3.1 布局总览

```
┌──────────────────────────────┬────────────────────────────────┐
│ Header            256 B      │ 固定，含 CRC32                  │
├──────────────────────────────┼────────────────────────────────┤
│ Tensor Index  n×128 B        │ 张量目录                        │
├──────────────────────────────┼────────────────────────────────┤
│ (pad to 64)                  │                                │
├──────────────────────────────┼────────────────────────────────┤
│ Data Region                  │ 每个张量: data | scale | aux    │
│   每个子块 64 B 对齐          │  aux = HGF(A,B) 或 row_wsum    │
├──────────────────────────────┼────────────────────────────────┤
│ Tokenizer Blob              │ u32 len + UTF-8 JSON            │
├──────────────────────────────┼────────────────────────────────┤
│ Metadata JSON               │ u32 len + UTF-8 JSON            │
└──────────────────────────────┴────────────────────────────────┘
```

### 3.2 Header（256 B，精确字段偏移）

| Off | Size | Type | 字段 | 说明 |
|---:|---:|---|---|---|
| 0 | 4 | `u8[4]` | `magic` | `50 48 4F 53` = `"PHOS"` |
| 4 | 1 | `u8` | `version_major` | `1` |
| 5 | 1 | `u8` | `version_minor` | `0` |
| 6 | 1 | `u8` | `endian` | `0` = little-endian |
| 7 | 1 | `u8` | `alignment` | `64` |
| 8 | 4 | `u32` | `header_bytes` | `256` |
| 12 | 4 | `u32` | `index_entry_bytes` | `128` |
| 16 | 8 | `u64` | `file_bytes` | 文件总字节数 |
| 24 | 8 | `u64` | `flags` | 见 §3.3 |
| 32 | 4 | `u32` | `n_layer` | 层数 L |
| 36 | 4 | `u32` | `d_model` | d |
| 40 | 4 | `u32` | `n_head` | 查询头数 H |
| 44 | 4 | `u32` | `n_kv_head` | KV 头数（GQA）H_kv |
| 48 | 4 | `u32` | `d_head` | d/H |
| 52 | 4 | `u32` | `d_ff` | FFN 隐层 |
| 56 | 4 | `u32` | `vocab_size` | V |
| 60 | 4 | `u32` | `max_seq_len` | 上下文上限 |
| 64 | 4 | `f32` | `rope_theta` | 默认 10000.0 |
| 68 | 4 | `f32` | `norm_eps` | 默认 1e-5 |
| 72 | 4 | `u32` | `act_enum` | `0`=SwiGLU, `1`=ReLU² |
| 76 | 4 | `u32` | `quant_enum` | `0`=fp32, `1`=int8, `2`=ternary |
| 80 | 4 | `u32` | `w_granularity` | `0`=per_tensor, `1`=per_channel(按输出行), `2`=per_block |
| 84 | 4 | `u32` | `w_block_size` | per_block 块大小，默认 128 |
| 88 | 4 | `u32` | `a_granularity` | `0`=per_tensor, `1`=per_token, `2`=per_block |
| 92 | 4 | `u32` | `pack_scheme` | `0`=none, `1`=ternary_5bit_v1（本文 §3.6） |
| 96 | 4 | `u32` | `hgf_rank` | O3 全局默认秩 r |
| 100 | 4 | `f32` | `hgf_alpha` | 缩放 α（默认 1.0，配合 `/r` 归一化） |
| 104 | 4 | `f32` | `hgf_gate_init` | 门控初值（默认 0.05，可训练） |
| 108 | 4 | `u32` | `kv_block_size` | 分页 KV 块大小（默认 16） |
| 112 | 4 | `u32` | `n_tensor` | 张量条目数 n |
| 116 | 4 | `u32` | `early_exit_layer` | **O1** 早退层号 E（无则 `0xFFFFFFFF`） |
| 120 | 8 | `u64` | `tensor_index_off` | 默认 `256` |
| 128 | 8 | `u64` | `tokenizer_off` | 0 = 无 |
| 136 | 8 | `u64` | `tokenizer_bytes` | |
| 144 | 8 | `u64` | `metadata_off` | |
| 152 | 8 | `u64` | `metadata_bytes` | |
| 160 | 8 | `u64` | `data_region_off` | |
| 168 | 8 | `u64` | `data_region_bytes` | |
| 176 | 4 | `u32` | `crc32_header` | CRC32(字节 `[0,176)`) |
| 180 | 4 | `u32` | `reserved1` | 0 |
| 184 | 72 | `u8[72]` | `reserved_zero` | 全 0，补齐到 256 |

### 3.3 `flags` 位定义

| Bit | 名称 | 含义 |
|---:|---|---|
| 0 | `HAS_HGF` | 至少一个张量带 HGF 低秩因子 |
| 1 | `TIED_EMBED` | `lm_head.weight` 与 `tok_emb` 共享（不重复存储） |
| 2 | `HAS_TOKENIZER` | 含 tokenizer blob |
| 3 | `HAS_ROW_WSUM` | 含每输出行的 `Σ_j W[i,j]`（int32），供 I2_S 零点校正 |
| 4 | `HAS_EARLY_HEAD` | 含 O1 早退头（`{E}.early_head.*`） |
| 5 | `KV_INT8` | KV cache 默认 int8 |
| 6 | `HAS_INT8_MIRROR` | 额外存一份 int8 镜像权重，供 I2_S 快速路径 |
| 7 | `HAS_SCALES_F16` | scale 以 f16 存储（否则 f32） |
| 8 | `QUANT_EMBED` | 嵌入表也被量化 |
| 9–63 | — | 保留 |

### 3.4 Tensor Index Entry（128 B / 条）

| Off | Size | Type | 字段 | 说明 |
|---:|---:|---|---|---|
| 0 | 32 | `char[32]` | `name` | 见 §3.5 命名规范，UTF-8，`\0` 填充 |
| 32 | 1 | `u8` | `dtype` | `0`=f32 `1`=f16 `2`=bf16 `3`=int8 `4`=uint8 `5`=int32 `6`=TERNARY_PACKED |
| 33 | 1 | `u8` | `ndim` | 1–4 |
| 34 | 1 | `u8` | `pack_scheme` | 同 header，逐张量覆盖 |
| 35 | 1 | `u8` | `tflags` | bit0 packed / bit1 has_scale / bit2 has_hgf / bit3 has_row_wsum |
| 36 | 16 | `u32[4]` | `shape` | shape[0..3]，不足补 0（行主序 C-order） |
| 52 | 4 | `u32` | `n_elements` | **打包前**逻辑元素数 |
| 56 | 4 | `u32` | `scale_dtype` | `0`=none `1`=f32 `2`=f16 |
| 60 | 4 | `u32` | `n_scale` | scale 个数（per_channel = shape[0]，per_tensor = 1） |
| 64 | 8 | `u64` | `data_off` | 64 B 对齐 |
| 72 | 8 | `u64` | `data_bytes` | |
| 80 | 8 | `u64` | `scale_off` | 4 B 对齐（块尾补 64） |
| 88 | 8 | `u64` | `scale_bytes` | |
| 96 | 8 | `u64` | `aux_off` | HGF `A∥B` 或 `row_wsum`，64 B 对齐 |
| 104 | 8 | `u64` | `aux_bytes` | |
| 112 | 4 | `u32` | `hgf_rank` | 本张量秩（0 = 无 HGF） |
| 116 | 4 | `u32` | `aux_layout` | `0`=无 `1`=`A[r,d_in]∥B[d_out,r]` (f16) `2`=`row_wsum_i32[d_out]` |
| 120 | 8 | `u64` | `reserved` | 0 |

### 3.5 张量命名规范

```
tok_emb                                   [V, d]        f16（QUANT_EMBED 时 int8/ternary）
norm_final                                [d]           f32
lm_head                                   [V, d]        f16（TIED_EMBED 时不存储）
layers.{i}.attn_norm                      [d]           f32   SubLN before Q/K
layers.{i}.ffn_norm                       [d]           f32
layers.{i}.attn.q_proj.weight             [d, d]        ternary
layers.{i}.attn.k_proj.weight             [d_kv, d]     ternary   d_kv = n_kv_head*d_head
layers.{i}.attn.v_proj.weight             [d_kv, d]     ternary
layers.{i}.attn.o_proj.weight             [d, d]        ternary
layers.{i}.attn.subln_out                 [d]           f32   BitNet SubLN on attn out
layers.{i}.mlp.gate_proj.weight           [d_ff, d]     ternary
layers.{i}.mlp.up_proj.weight             [d_ff, d]     ternary
layers.{i}.mlp.down_proj.weight           [d, d_ff]     ternary
layers.{i}.mlp.subln_out                  [d]           f32
layers.{E}.early_head.lora_a              [d, r]        f16   仅 HAS_EARLY_HEAD 且 r>0
layers.{E}.early_head.lora_b              [r, V]        f16
```

> `q/k/v/o/gate/up/down` 亦可带 `.scale`（per-channel 时 `[d_out]`，per_tensor 时 `[1]`）与 `.row_wsum`（`[d_out]` int32），这些**不占独立 index 条目**，走 entry 的 `scale_off` / `aux_off`。

### 3.6 三值权重 5-bit 打包布局（`ternary_5bit_v1`）★ 本项目核心

**基本单元：24 个三值权重 → 5 字节**（= 1.667 bit/weight，与 BitNet 理论下界 `log2(3)=1.585` 仅差 5.2%）。

**Step 1 · 数字映射**：`w ∈ {-1,0,1} → d ∈ {0,1,2}`（`-1→0, 0→1, +1→2`）

**Step 2 · 3 个权重一组 → base-3 码**
```
c = d0 + 3*d1 + 9*d2          // c ∈ [0, 26]，共 27 态
```

**Step 3 · 符号-索引编码（1 bit 符号 + 4 bit 索引）**
利用对称约简 `27 = 1（全零） + 13×2（互为相反数）` → **14 个规范型**，4 bit 索引足够（16 ≥ 14）：
```
if c == 13:                    # 全零 (d=1,1,1)
    index, sign = 0, 0
elif c <= 12:                  # 规范型恰好是 c ∈ [0,12]
    index, sign = c + 1, 0
else:                          # c ∈ [14,26]，取相反数 26-c ∈ [0,12]
    index, sign = (26 - c) + 1, 1

code5 = (sign << 4) | index    # index ∈ [0,13]，5 bit
```

**Step 4 · 组打包（8 组 → 40 bit = 5 字节）**
```
u64 v = 0;
for g in 0..7:  v |= (u64)code5[g] << (5*g);   // 40 bit 有效
memcpy(out, &v, 5);                            // 小端写出低 5 字节
```

**Step 5 · 行对齐**
矩阵行宽 `K` 补齐到 **24 的倍数**：`K_pad = ceil(K/24)*24`，补 0 权重。
单行打包字节数 `= K_pad * 5 / 24`（整数，5 字节的倍数，LUT chunk 天然对齐）。
全矩阵 `packed_bytes = n_rows * K_pad * 5 / 24`。

**解包（供参考实现 / 测试）**
```
byte_off = (row * K_pad/24) * 5 + (group/8)*5
bit_off  = (group % 8) * 5
code5    = (u16)(load48bits(byte_off) >> bit_off) & 0x1F   // 读 6 字节防跨界
```

> **为什么用「符号+索引」而非直接 base-3？** 直接 base-3（0..26）需要 27 项 LUT；符号+索引只需 **14 项**（补齐 16），LUT 恰好 16 项 —— 在 SIMD 上 16×int8 正好一个 XMM 寄存器、16×int16 正好可用「双 pshufb 高低字节表」技巧查表，符号用一条 `_mm256_sign_epi8/epi16` 施加。这是本项目对 bitnet.cpp `three_lut` 的**微架构级超越点**（详见 `docs/ALGORITHMS.md` A-3 条目）。

### 3.7 HGF 低秩因子布局

`aux_layout = 1` 时，aux 区为 `A[r, d_in] ∥ B[d_out, r]`，均 f16，行主序，`A` 先 `B` 后，整体 64 B 对齐。
前向：`y = (ternary_matmul(W, x) * scale) + gate * (alpha/r) * (B @ (A @ x))`

**内存开销公式**（相对三值打包字节）：
```
overhead% = r * (d_in + d_out) * bytes_per_param / (d_in * d_out * 1.667 / 8) * 100
```
以 `d_in=d_out=512`、f16(2B) 为例：`r=1→3.75%`，`r=2→7.5%`，`r=4→15%`，`r=8→30%`，`r=16→60%`。
→ **默认 r=4 落在 PRD 要求的「12–15% 额外内存」区间**。Pareto 扫描 `r ∈ {0,1,2,4,8,16}`。

### 3.8 Tokenizer Blob

```
[u32 json_bytes][UTF-8 JSON]
```
```jsonc
{
  "type": "bpe", "version": 1,
  "pretokenizer": "gpt2",
  "byte_level": true,          // GPT-2 bytes_to_unicode，保证任意 UTF-8 无损
  "vocab": ["<|bos|>", "..."],  // 下标即 id
  "merges": ["a b", "b c"],     // 有序，rank = 下标
  "specials": {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2},
  "unk_id": 3,
  "vocab_size": 8192
}
```

### 3.9 Metadata JSON

```jsonc
{
  "created_at": "2026-10-16T...Z",
  "generator": "phosphorus 0.1.0",
  "git_sha": "...",
  "source_ckpt": "ckpt/tiny-qat-step20000.pt",
  "torch_version": "2.x",
  "train": {"steps": 20000, "lr": 3e-3, "final_loss": 3.41, "val_ppl": 18.7},
  "quant": {"scheme": "absmean", "granularity": "per_channel", "hgf_rank": 4, "qat": true},
  "pareto": [{"r": 0, "ppl": 21.3, "bytes": 5461300}, ...],
  "license": "Apache-2.0",
  "author": "晨星 CJX0712"
}
```

---

## 4. 关键时序图

### 4.1 训练 → 量化导出 → 推理 全链路

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant CLI as cli.py
    participant CRP as data.corpus
    participant DS as data.dataset
    participant BPE as tokenizer.bpe
    participant TR as train.trainer
    participant M as model.llama
    participant Q as model.quantization
    participant EX as quantize.export
    participant PK as quantize.packing
    participant LD as engine.model_loader
    participant KR as kernels.registry
    participant EG as engine.generate

    U->>CLI: phosphorus train --config tiny.yaml
    CLI->>CRP: load(corpus=tinystories)
    CRP-->>CLI: Corpus(text)  [离线兜底: data/raw/]
    CLI->>BPE: train(text, vocab_size=8192)
    BPE-->>CLI: vocab + merges
    CLI->>DS: encode(corpus) -> memmap ids
    DS-->>CLI: TokenDataset

    loop 每个 step (fp32 基线 或 QAT/STE)
        CLI->>TR: fit()
        TR->>M: forward(x, exit_at=E)  % LayerSkip 多出口
        M->>Q: weight_quant_absmean(W)  % STE: 前向量化/反向直通
        Q-->>M: W_hat {-1,0,1} * scale
        M-->>TR: logits_main, logits_early
        TR->>TR: loss = sum_e w_e * CE(logits_e, y)
        TR->>TR: AdamW + cosine + clip(1.0)
        TR->>TR: checkpoint.save(step)
    end

    U->>CLI: phosphorus quantize --in ckpt.pt --out tiny.phos --rank 4
    CLI->>EX: export(ckpt, quant_cfg)
    EX->>Q: 逐张量 absmean 量化 + SubLN/norm 保留 f32
    EX->>PK: pack_ternary_v1(W_ternary)  % 24→5 字节
    PK-->>EX: packed u8 + per-channel scale + row_wsum
    EX->>EX: 写 Header(256B) + Index(n*128B) + Data + Tokenizer + Metadata
    EX-->>CLI: tiny.phos (含 CRC32)

    U->>CLI: phosphorus generate --model tiny.phos --prompt "..."
    CLI->>LD: load(tiny.phos)  % mmap
    LD->>KR: 探测 native(zig cc) → numpy → 回退
    KR-->>LD: Backend
    LD-->>CLI: RuntimeWeights
    CLI->>EG: generate(weights, prompt, backend, speculate=auto)
    EG-->>U: 流式 token
```

### 4.2 投机解码一轮（含树草稿与验证）★ O1 + O2

```mermaid
sequenceDiagram
    autonumber
    participant G as engine.generate
    participant C as speculate.controller
    participant D as speculate.layerskip (O1)
    participant T as speculate.tree
    participant M as 目标模型 (三值)
    participant V as speculate.verify (Leviathan)

    G->>C: plan(batch_size, hist_alpha)
    C->>C: 读取 EWMA: alpha_hat, cost_ratio c
    C->>C: 枚举 K∈[0..Kmax], 取 argmax S(K)
    alt S(K*) < 1+eps 或 batch > Bmax
        C-->>G: SpecPlan(enabled=false, K=0)
        Note over G,M: 退化为普通自回归一步
    else 开启投机
        C-->>G: SpecPlan(enabled=true, K=K*, width=w)
    end

    G->>D: draft(ctx, K*, w)
    D->>M: 只跑前 E 层 (同一份三值权重!)
    D->>M: early_head(hidden_E) -> probs
    D-->>T: 候选 token 序列 (K* 步, 每步 top-w)
    T->>T: 按联合概率剪枝 (joint_prob < tau 丢弃)
    T-->>G: DraftTree(paths, mask, positions)

    G->>M: verify_forward(tree)  % 一次 batched prefill
    M-->>V: target_probs[每个树节点]
    V->>V: Leviathan 逐个节点: r~U(0,1);<br/>accept if r < min(1, p_t/q_d);<br/>else 用 max(0,p_t-q_d) 重采样修正
    V-->>G: accepted_paths (γ 个 token) + bonus token

    G->>C: observe(K_actual=K*, gamma=γ, t_draft, t_verify)
    C->>C: alpha_hat ← β*alpha_hat + (1-β)*γ/K*<br/>c ← EWMA(t_draft/t_verify)
    G->>G: 提交 γ+1 个 token 到 KV (分页写入)
    Note over G: 若 γ==0，bonus token 保证每轮至少前进 1 步
```

### 4.3 连续批处理 + 分页 KV + 驱逐

```mermaid
sequenceDiagram
    autonumber
    participant S as scheduler
    participant B as block_manager
    participant K as kv_cache
    participant E as eviction
    participant A as attention

    S->>S: 收集 runnable seqs (waiting → running)
    S->>B: can_alloc(seq, n_blocks)?
    alt 块不足
        B-->>S: 抢占最年轻 seq (swap-out, 释放块, 标记 recompute)
    end
    S->>B: alloc(seq) -> block_table
    loop 每个 decode step
        S->>K: gather(seq, layer, head) via block_table
        K-->>A: K/V 页视图
        A->>A: RoPE + GQA + 树掩码
        A-->>S: out
        S->>S: sampler -> next token
        S->>K: append(next)  % 写入当前块
        alt 序列长度 % evict_interval == 0 且 超出预算 M
            S->>E: select(attn_scores, layer_budget)
            E->>E: SnapKV / PyramidKV / H2O / Streaming 策略
            E-->>K: keep_mask
            K->>B: compact()  % 重写 block_table，回收空洞块
        end
        alt 遇到 EOS / max_len
            S->>B: free(seq)
        end
    end
```

---

## 5. 三值 GEMM 内核

### 5.0 公共前提

- 权重 `W ∈ {-1,0,1}`，按 §3.6 打包为 5-bit 流（或镜像为 int8 `{-1,0,1}`）。
- 激活按 token/channel 量化为 **8-bit**：`xq = round(clip(x / sx, -127, 127))`，`sx` 为 `per_token`（BitNet 默认）或 `per_tensor` 缩放。
- **两者都在整数域运算 → 结果与 fp32 参考在量化域内逐位无损**（不变量 ①）。
- 输出 `y_i = sx * sy_i * (Σ_j W[i,j] * xq_j)`；`sy_i` 为权重 per-channel scale。

### 5.1 TL（Ternary LUT）内核 — 查表法，内存受限场景首选

```c
// 伪代码 (C99, 可移植标量路径为权威实现)
// W: packed 5-bit, [N, K_pad] ; xq: int8[K] ; lut 每 3 个激活重建一次
void phos_tl_gemm(const uint8_t* W, const int8_t* xq, const f32* scale,
                  f32* y, int N, int K, int32_t* /*scratch*/) {
    const int NG = K_pad / 3;                 // 每组 3 个权重
    for (int n = 0; n < N; ++n) {             // 行并行：交给 Python 线程池
        const uint8_t* row = W + (size_t)n * (K_pad * 5 / 24);
        int32_t acc = 0;  int16_t acc16 = 0;
        for (int g = 0; g < NG; ++g) {
            uint8_t code5 = unpack5(row, g);  // 见 §3.6 Step 5
            // --- 建表（每 24 组一次，8 组共享 48 个激活）---
            if ((g & 7) == 0) build_lut16(xq + 3*g, lut);   // 见下
            int16_t p = lut[code5 & 0x0F];                  // 0..13 有效
            acc16 += (code5 & 0x10) ? (int16_t)(-p) : p;    // 符号位
            if ((g & 63) == 63) { acc += (int32_t)acc16; acc16 = 0; }  // 防 int16 溢出
        }
        acc += (int32_t)acc16;
        y[n] = (f32)acc * scale[n] * sx;
    }
}

// 规范 LUT 构建：14 项（补齐 16），int16 存储
// 令 a = (xq[3g], xq[3g+1], xq[3g+2])，base-3 码 c = d0+3d1+9d2 对应部分和
//   part(c) = Σ_k (digit(c,k) - 1) * a[k]        // digit∈{0,1,2} → w∈{-1,0,1}
// lut[0] = 0                                     // 全零
// for c in 0..12:  lut[c+1] = part(c)
// 对称性保证 part(26-c) = -part(c)，故只需 13 项 + 符号位
static void build_lut16(const int8_t* a, int16_t lut[16]) {
    lut[0] = 0;
    for (int c = 0; c < 13; ++c) {
        int d0 = c % 3, d1 = (c/3) % 3, d2 = c/9;
        lut[c+1] = (int16_t)((d0-1)*a[0] + (d1-1)*a[1] + (d2-1)*a[2]);
    }
    lut[14] = lut[15] = 0;   // padding
}
```

**为什么 LUT 只需 14 项且可一次性服务 8 个组？**
`build_lut16` 依赖的激活三元组每 8 个 group 换一批（8 组 = 24 权重 ↔ 24 激活 = 恰好一个 5 字节 chunk），所以建表摊销成本 = 1/8。

**SIMD 分块策略（可选路径，`#ifdef`）**
| 平台 | LUT 形态 | 查表指令 | 符号施加 | 累加 |
|---|---|---|---|---|
| 标量 C99 | `int16 lut[16]` | 直接下标 | 三元表达式 | int32（每 64 组提升） |
| x86 AVX2 | 双表 `TLO[16]`/`THI[16]`（低/高字节，各 16 B） | `_mm256_shuffle_epi8` ×2 + 合并 | `_mm256_sign_epi16` | `_mm256_madd_epi16(v, ones)` → int32，每 64 组 flush |
| ARM NEON | 同上 | `vqtbl1q_u8` ×2 | `vnegq_s16` + `vbslq` | `vaddl_s16` → int32 |
| 无 SIMD | 标量 | — | — | — |

- **行块并行**：`N` 维切成 `n_threads` 个 row-block（默认 = `os.cpu_count()`），Python `ThreadPoolExecutor` 分发，`ctypes` 自动释放 GIL；C 侧**无全局状态、可重入**。
- **K 维**：一次处理 8 个 group（= 1 个 5 字节 chunk），寄存器压力最小。
- **适用**：decode（M=1，GEMV，内存带宽受限）、大 KV、低并发；内存占用 = 1.667 bpw。

### 5.2 I2_S（int2 + scale）内核 — VNNI 点积法，算力受限场景首选

```
思路（bitnet.cpp I2_S 等价形式）：
  把三值权重镜像成 int8 ∈ {-1,0,1}（"int2" 值域），激活量化成 uint8 = xq_int8 + 128。
  用 8-bit 点积指令 (u8 × i8 → i32) 一次算 4 个元素。
  零点校正：y = Σ (x_u8[j] - 128) * W[i,j] = Σ x_u8[j]*W[i,j] - 128 * Σ_j W[i,j]
                                                              └── row_wsum[i]，导出时预计算并存 .phos
```

```c
void phos_i2s_gemm(const int8_t* Wi8,       // [N, K], 值 ∈ {-1,0,1}
                   const uint8_t* xu8,      // [K], xq + 128
                   const int32_t* row_wsum, // [N]
                   const f32* scale, f32* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        int k = 0;
#if defined(__AVX2__) && defined(__AVX512VNNI__)   // 或 AVX-VNNI
        __m512i s = _mm512_setzero_si512();
        for (; k + 64 <= K; k += 64)
            s = _mm512_dpbusd_epi32(s, _mm512_loadu_si512(xu8+k), _mm512_loadu_si512(Wi8+(size_t)n*K+k));
        acc += _mm512_reduce_add_epi32(s);
#elif defined(__ARM_NEON)
        int32x4_t s = vdupq_n_s32(0);
        for (; k + 16 <= K; k += 16)
            s = vdotq_s32(s, vld1q_u8(xu8+k), vld1q_s8(Wi8+(size_t)n*K+k));
        acc += vaddvq_s32(s);
#endif
        for (; k < K; ++k) acc += (int32_t)xu8[k] * (int32_t)Wi8[(size_t)n*K+k];
        acc -= 128 * row_wsum[n];                 // 零点校正，位精确
        y[n] = (f32)acc * scale[n] * sx;
    }
}
```

- **内存代价**：权重以 int8 镜像存储 = 8 bpw（是 TL 的 4.8 倍），故仅在 `flags.HAS_INT8_MIRROR` 时可用；否则**运行时从 5-bit 包解包到 int8 scratch**（一次性，prefill 期间摊销）。
- **位精确性**：TL 与 I2_S 都精确计算 `Σ W·xq`。TL 用 int8 激活 + 符号，I2_S 用 u8 激活 + 零点校正 → 数学恒等，**两者输出 int32 必须逐位相同**（这是 `test_gemm_bitwise.py` 的核心断言）。
- **适用**：prefill（M≫1，算力受限）、大 batch；VNNI / NEON dot 指令吞吐 = 4× 标量。

### 5.3 后端选择决策表（`kernels/registry.py`）

| 场景 | 首选 | 理由 |
|---|---|---|
| decode, batch ≤ 4 | **TL (native)** | 内存带宽受限，1.667 bpw 最省 |
| prefill, M ≥ 64 | **I2_S (native)** | 算力受限，VNNI/NEON dot 吞吐最高 |
| 无 native（zig 缺失/编译失败） | NumPy TL/I2_S | `np.packbits`-free，用查表 + `np.bincount` 向量化 |
| `--backend fp32` | fp32 GEMM | 基线对照（基准测试必跑） |
| `--backend int8` | int8 GEMM | 基线对照（基准测试必跑） |

**三级回退**：`native(zig cc)` → `numpy` → `pure-python(仅正确性兜底)`。回退必须在日志里显式 WARN，且 `phosphorus bench` 报告里标注 `backend_used`。

> T4-A 落地情况：`native(zig cc)` 一级已实现（`phosphorus/native/` + `phosphorus/csrc/`），
> `numpy` 一级已实现；`pure-python` 兜底未实现（numpy 已是硬依赖，该级仅作形式规划）。

### 5.4 native 实现层（T4-A 落地）

两个正交轴，代码里分开表达（`PhosModel.__init__`）：

| 轴 | 取值 | 位置 |
|---|---|---|
| **数学后端** | `fp32` / `int8` / `tl` / `i2s` | `PhosModel.backend`（`--backend` CLI 参数） |
| **实现后端** | `native` / `numpy` | `PhosModel.impl`（`phosphorus/backend.py` 注册，默认 `auto`） |

**编译链**（`phosphorus/native/build.py`）：随 wheel 分发 C99 源码（`phosphorus/csrc/`），
首次使用时调 `zig cc` 现场编译为共享库，按「ABI 版本 + zig 版本 + 目标三元组 +
编译参数 + 全部源码字节的 sha256」做**内容寻址缓存**（`phos_native-<digest>.dll`），
源码或工具链任一变化自动重编；文件锁（`O_CREAT|O_EXCL`，120s 超时）防多进程竞争；
Windows 下加载中的 DLL 不可覆盖，故原子 rename（`.tmp<pid>` → 正式名）。

**ABI 与数值契约**（`phosphorus/csrc/phos.h`，`PHOS_ABI_VERSION 1`）：

- 全部内核提供 `unsigned flags` 参数（`PHOS_FLAG_SCALAR` / `PHOS_FLAG_AVX2`），
  运行时 CPUID 探测（不用 `__builtin_cpu_supports`——zig 的 windows-gnu 目标不链
  libgcc 的 `__cpu_model`），无 AVX2 的机器自动走 scalar，**数值结果不变**。
- fp32 输出的缩放必须走 double 域左结合：`y = (float)((double)acc · (double)wscale · (double)sx)`，
  与 numpy 参考（float64）逐位相等；`sx` 在 ABI 里是 `double` 而非 `float`
  （早期版本用 `float` 导致 1 ULP 偏差，已修正）。
- **位精确性是不变量 ① 的延伸**：native 与 numpy 在 int32 域、fp32 域都逐位相等，
  由 `tests/test_native_backend.py` L0/L2 与 `scripts/bench_kernels.py` 每次运行断言。

**自适应调度**（`PhosModel._use_native(T, N)`，实测 crossover 见 `BENCHMARKS.md` §4）：

| 数学后端 | 规则 | 实测依据 |
|---|---|---|
| `tl` | 恒走 native | 全部形状 2.8x~6.8x（numpy 侧是逐 token Python 循环） |
| `i2s` | `T == 1` 或 `N ≥ 1024` | 小 N + T>1 时 numpy 批量矩阵乘赢（0.74x）；N≥1024 时原生赢 4.6x~9.4x |

**并行**（`phosphorus/native/parallel.py`）：仅按 N 维切行块（`PHOS_THREADS` 环境变量，
默认 `min(8, cpu)`），行块结果与整块调用逐位相等（L5 断言）——因为累加顺序不变。

### 5.5 分页 KV（T4-B）

**两种模式**（`PhosModel(kv_mode=...)`，CLI `--kv`）：

| 模式 | 存储 | 适用 |
|---|---|---|
| `dense` | 每层一份 `[1, n_head, T, d_head]`（GQA 已展开） | 参考实现 / 对照测试 |
| `paged`（默认） | 物理池 `kv[n_blocks, n_layer, 2, block_size, n_kv_head, d_head]` + `block_tables[s, j]` | 生产路径；为驱逐 / 连续批处理奠基 |

**块层语义移植自 vLLM v1 `BlockPool`**（Apache-2.0，见 `NOTICE`）：
引用计数（仅 `ref_cnt` 归零才回空闲队列）、非缓存块 LIFO 复用、`null_block`
占位（block_id=0，不参与分配与计数）、空闲队列摘/还。**未移植**：GPU 张量、
前缀缓存哈希（`enable_caching=False`）、跨设备 swap。

**滑动窗口**（T4-C 之前的简易驱逐）：每条序列记 `head_offset` = 首块内已废弃
slot 数，逻辑位置 p 的存储位置是 `head_offset + p`；丢弃任意数量 token 都只在
`head_offset` 累加满一块时回收整块——**逻辑→物理映射永不错位**，被丢块的 KV
数据无需搬运。

**写入协议**（三段式，避免预留与写入的位置歧义）：
`begin_append(seq, T)` 预留块 → 逐层 `write(seq, layer, k, v)` → `end_append(seq)`；
单层便利封装 `append()`。容量预检 `can_append()`，超限时 `begin_append` 返回
False（引擎随即报错，因为池按 `max_seq_len` 精确 sizing）。

**数值等价**：paged 只存 `n_kv_head` 份、读取时 `np.repeat` 展开，与 dense 的
「先展开后计算」逐元素相同（repeat 是精确拷贝）；注意力数学核心只有一份
（`attention.gqa_attention`），故差异只可能来自浮点累加顺序——实测引擎级
logits 最大差 < 1e-4、argmax 100% 一致（不变量 ⑨）。

**块大小**：默认 16（vLLM 默认值，适配小模型）；实测 4/16/32/64 结果一致，
块越大 gather 的间接寻址开销越低、但滑动窗口的粒度越粗。

---

### 5.6 KV 驱逐（T4-C）

四种可插拔策略（`none` 对照组 + `snapkv` / `pyramidkv` / `h2o` / `streamingllm`，CLI `--eviction`），统一接口 `EvictionPolicy.select(scores, budget, n_total) -> keep_mask`：

- **打分**：SnapKV / PyramidKV 用观测窗口（末尾 `window_size` 个 query）对 past key 的注意力分数（`attention.kv_importance_scores`，含窗口内因果 mask + avg/max pool，kernel 5，跨 head 聚合为 token 粒度）；H2O 走全矩阵累积（`self._h2o[i]` 跨 decode 步位置对齐累加）；StreamingLLM 不打分。
- **预算**：`BudgetAllocator` 支持 `uniform`（均分）与 `pyramid`（金字塔，默认）。PyramidKV 公式对齐官方 `pyramidkv_utils.py` 双分支 + `max_num` 钳制，最大余数法把逐层之和精确补到 `L·C`（C5 验收 `|Σ − L·C| ≤ 1`，实测 = 0）。
- **强制保留**：所有策略永久保留前 `sink_size`（默认 4）个 attention sink + 末尾 `window_size`（默认 32）个最近 token。StreamingLLM 的 sink 思路上升为全局硬约束。
- **物理压实**：`PagedKVCache.compact(seq, keep_mask)` 先拷出保留 K/V、再 `free_seq` 回收、按 `max(layer_len)` 重建块表（`head_offset` 归零、逻辑坐标重排到 `[0, layer_len)`）；逐层 `layer_len[l]` 跟踪每层有效长度，gather 按层读取。LIFO 回收保证释放后拿回同一批物理块、避免不同层搬运互相覆盖。
- **触发**：`PhosModel.forward` 在 `end_append` 后、缓存超过 `_evict_threshold`（= `max(sink+window+1, round(max_seq_len·kv_budget))`）时调用 `_run_eviction()`；`none` 完全不触发。
- **验收（C5，不变量 ⑦）**：下游 final-logits 余弦 ≥ 0.99（50% 预算）；原始注意力 L2 即便 oracle 也 ≈ 0.5，属指标固有性质，故以文献同款 logits / PPL 衡量。实测见 `BENCHMARKS.md` §5.4。

---

### 5.7 连续批处理调度器（T4-D）

把多条序列的新 token **融合进一次前向**，是吞吐从「N 条串行」到「N 条并行」的关键（C6）。设计要点：

- **`PhosModel.forward_step(seq_tokens: dict[int, np.ndarray]) -> dict[int, np.ndarray]`**：单次融合前向。
  - 输入 `{seq_id: 新token ids}`（每条序列**仅本轮新产生的 token**，不含历史），输出每条序列**末位** token 的 logits。
  - 内部把所有序列 token 拼成 `[N_total, d]` 矩阵，逐层计算 `q/k/v` 投影后**按 `seq_id` 切片隔离**：RoPE 绝对位置 `abs_start = _abs_dropped + cache_len(seq)` 逐序列独立；注意力对每条序列单独调用 `paged_gqa_attention(qs, cache, s, i, n_rep, starts[s])`，物理 KV 由 `block_tables[s]` 自然隔离。
  - **残差守恒**：注意力输出只 `attn_x = concat(各序列 out)` 送 `o_proj`，再 `x = x + subln_attn * o`；`x`（残差）全程不覆盖，保证与逐条 `forward` 逐位一致（不变量①）。
  - 三段式写入：`begin_append(s, T)`（每序列预留块）→ 逐层 `write` → 全部层结束后统一 `end_append(s)`。
  - `dense` 模式无融合收益，自动降级为逐序列串行 `forward`；`eviction != none` 时禁用融合（两类动态 KV 结构不可同时混用），抛 `NotImplementedError`。

- **`Scheduler`（`phosphorus/scheduler.py`）**：
  - `generate_ids(prompts_ids, max_new_tokens, temperature, top_k, top_p, seed)`：把 `len(prompts) > max_seqs` 的请求**按波次（wave）切分**为每波 `≤ max_seqs` 条；每波 `model.reset()` 后用 `forward_step` 一次性 prefill + decode；每条序列用独立 `np.random.default_rng(seed + 全局下标)` 保证**融合批与逐条串行在随机采样下也逐位一致**。
  - `dense` 后端同样支持（自动逐序列）；`eviction` 与连续批处理互斥，调度器始终走 `paged + none`。
  - CLI：`phosphorus sched --model tiny.phos --prompts "..." "..." --max-seqs 8 [--bench]`，`--bench` 打印吞吐。

- **关键数据结构变更**：`PagedKVCache.layer_len` 由 `[n_layer]`（全局共享写游标）重构为 `[n_seqs, n_layer]`，使每条序列在每一层都有独立有效长度——这是多序列融合写入不互相覆盖的前提。

- **验收（C6/C7）**：8 条序列融合生成的 token id 与逐条串行**逐位一致**（`test_scheduler_greedy_matches_serial`）；融合吞吐 **2.35×**（1158.8 vs 492.8 tok/s，tiny 模型 8×40 tok，`test_scheduler_throughput_speedup` 守护 ≥ 2×）；采样端到端：同 seed 两次运行结果 `==`、随机采样（temp=0.9）与贪心 `!=`、top_k/top_p/temperature 行为由 `sampling.py` 单测兜底。

## 6. 自适应投机控制器（O2）

### 6.1 观测与在线拟合

每轮投机结束后调用 `controller.observe(...)`：

| 观测量 | 定义 | 更新方式 |
|---|---|---|
| `alpha_hat` | 接受率 `γ / K` | EWMA，`β=0.9`：`α ← β·α + (1-β)·γ/K` |
| `c` | 成本比 `t_draft / t_verify`（单 token 草稿耗时 / 单轮目标验证耗时，均按 batch 摊销） | EWMA `β=0.9` |
| `batch_size` | 当前并发序列数 | 直接读 |
| `kv_util` | KV 块占用率 | 直接读 |

### 6.2 决策公式

**精确模型**（用于选 K）：单轮投机期望产出 token 数（几何分布截断）：
```
E[accepted | K, α] = (1 - α^(K+1)) / (1 - α)          // α→1 时取 K+1
cost(K)            = 1 + c·K                          // 1 轮目标验证 + K 步草稿
S(K)               = E[accepted | K, α] / cost(K)
K* = argmax_{K ∈ [0, K_max]} S(K)                    // K_max 默认 8
```

**简化闭式模型**（用于对外汇报 / 预测，PRD 指定）：
```
speedup ≈ (1 + α·K) / (1 + ε)          // ε = 投机框架固定开销占比，在线由 1 - 实测/预测 反拟合
```

**树宽决策**（EAGLE-2 式按累积概率分配预算）：
```
w = clamp( round( sqrt(α_hat · K_max · budget_factor) ), 1, W_max )     // W_max 默认 4
剪枝：对候选路径按 joint_prob = Π p_draft(t_i) 降序，丢弃 joint_prob < τ (=0.02) 的路径
```

### 6.3 状态机

```mermaid
stateDiagram-v2
    [*] --> OFF
    OFF --> WARMUP: batch_size ≤ B_max 且 S(K*) ≥ 1+eps_on
    WARMUP --> ON: 连续 N_warm(=4) 轮完成且 alpha_hat 有效
    WARMUP --> OFF: alpha_hat < alpha_min(=0.15)
    ON --> ON: S(K*) ≥ 1+eps_off
    ON --> DEGRADED: alpha_hat < alpha_min 或 batch_size > B_max
    DEGRADED --> ON: alpha_hat ≥ alpha_min 且 batch_size ≤ B_max 且 dwell ≥ N_dwell
    DEGRADED --> OFF: dwell ≥ N_dwell(=8) 且 S(K*) < 1.0
    OFF --> OFF: batch_size > B_max
    ON --> [*]
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `B_max` | 8 | **batch 大于此值直接关闭投机**（CPU 上草稿成本随 batch 线性增长，而目标验证被 amortize —— 这就是 vLLM 社区公认的「盲目开启反模式」） |
| `alpha_min` | 0.15 | α 低于此值判为无效 |
| `eps_on` | 0.15 | 开启阈值：`S(K*) ≥ 1 + 0.15` |
| `eps_off` | 0.05 | 保持阈值：`S(K*) ≥ 1 + 0.05` |
| `N_dwell` | 8 | 最小驻留轮数，防抖动 |
| `K_max` | 8 | 草稿长度上限 |
| `W_max` | 4 | 树宽上限 |

### 6.4 反模式规避（显式设计约束）

1. **batch 门控**：`batch_size > B_max` → 强制 OFF。CPU 上这条尤其致命：草稿是 GEMV（内存带宽受限）而目标是 GEMM（可摊batch），batch 越大投机越亏。
2. **α 门控**：α < 0.15 → 不开启。
3. **抖动抑制**：`N_dwell` + 双阈值（开 1.15 / 关 1.00）滞回。
4. **`--speculate on|off|auto`**：`auto` 走控制器，`on` 强制开启（用于基准对照，报告里必须标注"非自适应"）。

---

## 7. 核心不变量（测试契约）

| # | 不变量 | 测试文件 | 断言 |
|---|---|---|---|
| ① | TL / I2_S GEMM 与 fp32 参考**在量化域内逐位无损** | `test_gemm_bitwise.py` | L1：`kernel_out == int32_reference` 逐位相等；L2：`\|kernel·s − fp32(W_deq·x_deq)\| / \|fp32\| < 1e-5` |
| ② | 投机解码采样分布与目标分布**统计等价** | `test_speculative_distribution.py` | 100k 采样 token，χ² 检验 `p > 0.05`；经验 KL < 1e-3 |
| ③ | pack / unpack 往返恒等 | `test_packing.py` | `unpack(pack(w)) == w`，对 1e6 随机三值向量 + 边界（长度非 24 倍数） |
| ④ | RoPE 相对性 | `test_rope.py` | `⟨RoPE(x,m), RoPE(y,n)⟩` 只依赖 `m−n` |
| ⑤ | BPE encode→decode 恒等 | `test_bpe.py` | 对 10k 条含 emoji/CJK/控制字符文本，`decode(encode(s)) == s` |
| ⑥ | QAT STE 梯度 == 数值梯度 | `test_qat_ste.py` | 前向量化点的反向梯度与 `finite_diff(·, h=1e-3)` 相对误差 < 5% |
| ⑦ | KV 驱逐后注意力输出可控误差界 | `test_kv_eviction.py` | 50% 预算下，逐层注意力输出相对 L2 误差 < 10%，且 PPL 退化 < 10% |
| ⑧ | native 与 numpy 实现在 int32/fp32 域**逐位相等**（T4-A） | `test_native_backend.py` | L0：`native.tl_i32 == native.i2s_i32 == int32_reference`；L2：fp32 输出与 float64 参考逐位相等；L5：行块切分 == 整块 |
| ⑨ | 分页（paged）KV 与 dense 参考**数值一致**（T4-B） | `test_paged_kv.py` / `test_engine_paged.py` | 内核级最大绝对差 < 1e-5（三方比对含朴素逐元素实现）；引擎级 logits 最大差 < 1e-4 且 argmax 100% 一致；分块 prefill == 整段 prefill |
| ⑩ | 块管理器分配/回收**无泄漏**（T4-B） | `test_paged_kv.py` | 10,000 次随机分配/回收后 `utilization()` 精确回 0、`Σref_cnt == 0`；容量超限返回 False 且不改状态 |

---

## 8. 依赖包清单与可选分组

```toml
# pyproject.toml 依赖分组
[project]
requires-python = ">=3.10"
dependencies = [                      # core：装了就能推理
  "numpy>=1.24,<3",
  "tqdm>=4.66",
]

[project.optional-dependencies]
train   = ["torch>=2.2", "numpy>=1.24"]                    # CPU 版即可（不锁 +cu）
native  = ["ziglang>=0.13"]                                # O4：python -m ziglang cc
bench   = ["numpy>=1.24", "psutil>=5.9"]                   # RSS / CPU 计数
convert = ["transformers>=4.40", "safetensors>=0.4"]       # HF 互转（P1）
dev     = ["pytest>=7.4", "pytest-cov>=4.1", "hypothesis>=6.100",
           "ruff>=0.6", "pre-commit>=3.7"]
viz     = ["matplotlib>=3.8"]                              # 可选：Pareto PNG（ASCII 不依赖）
all     = ["phosphorus[train,native,bench,convert,dev]"]

[project.scripts]
phosphorus = "phosphorus.cli:main"
```

| 包 | 分组 | 必要性 | 备注 |
|---|---|---|---|
| `numpy` | core | 必需 | NumPy 后端 + 所有数组 |
| `tqdm` | core | 必需 | 训练/生成进度 |
| `torch` | train | 仅训练 | CPU 版 ~250 MB；推理路径**不 import torch** |
| `ziglang` | native | 仅编译内核 | pip 可装，**替代系统编译器** |
| `psutil` | bench | 评测 | RSS 峰值 / CPU 核数 |
| `hypothesis` | dev | 属性测试 | 不变量 ①③⑤ 用 `@given` |
| `ruff` | dev | lint | 配置见 `ruff.toml` |
| `transformers`, `safetensors` | convert | P1 | HF 互转 |
| `matplotlib` | viz | 可选 | 仅出图，ASCII 报告不需要 |

**严禁依赖**：`gcc`/`msvc`（无）、`openmp`、`cuda`、`triton`、`deepspeed`、`accelerate`。

---

## 9. 关键类图

```mermaid
classDiagram
    class ModelConfig {
        +int d_model
        +int n_layer
        +int n_head
        +int n_kv_head
        +int d_ff
        +int vocab_size
        +int max_seq_len
        +float rope_theta
        +float norm_eps
        +ActType act
        +bool tied_embed
        +int early_exit_layer
        +int early_head_rank
        +n_params() int
    }
    class QuantConfig {
        +QuantEnum scheme
        +Granularity w_granularity
        +Granularity a_granularity
        +int block_size
        +int hgf_rank
        +float hgf_alpha
        +bool qat
    }
    class DecodeConfig {
        +int max_new_tokens
        +float temperature
        +int top_k
        +float top_p
        +bool stream
        +SpecMode speculate
        +int kv_block_size
        +EvictionStrategy eviction
        +float kv_budget_ratio
    }
    class Tokenizer {
        +list~str~ vocab
        +list~tuple~ merges
        +dict specials
        +encode(str) list~int~
        +decode(list~int~) str
        +save(Path)
        +load(Path) Tokenizer
    }
    class BitLinear {
        +Tensor weight_fp
        +Tensor weight_scale
        +HybridGatedFlow hgf
        +forward(Tensor) Tensor
        +quantize_weight_() Tensor
        +straight_through(Tensor) Tensor
    }
    class HybridGatedFlow {
        +int rank
        +float alpha
        +Tensor gate
        +Tensor A
        +Tensor B
        +forward(Tensor) Tensor
        +param_bytes() int
    }
    class LlamaModel {
        +ModelConfig cfg
        +list~Block~ layers
        +forward(ids, exit_at) Tensor
    }
    class Trainer {
        +TrainConfig cfg
        +fit() dict
        +step(batch) float
        +save_ckpt(path)
        +load_ckpt(path)
    }
    class PhosWriter {
        +write_header(ModelConfig, flags)
        +add_tensor(name, arr, quant)
        +finalize(tokenizer, metadata) Path
    }
    class PhosReader {
        +parse_header() Header
        +tensor_index() dict
        +get(name) np.ndarray
        +tokenizer() Tokenizer
    }
    class Backend {
        +str name
        +gemm_fp32(a, b) ndarray
        +gemm_int8(a, b) ndarray
        +gemm_tl(Wp, xq) ndarray
        +gemm_i2s(Wi8, xu8) ndarray
        +rmsnorm(x, w) ndarray
        +softmax(x) ndarray
        +rope(x, pos) ndarray
    }
    class RuntimeWeights {
        +dict tensors
        +dict scales
        +dict hgf
        +Header header
    }
    class BlockManager {
        +int block_size
        +alloc(n) list~int~
        +free(seq)
        +can_alloc(n) bool
        +utilization() float
    }
    class PagedKVCache {
        +ndarray pool
        +ndarray block_tables
        +write(seq, layer, pos, k, v)
        +gather(seq, layer) ndarray
        +compact(seq, keep_mask)
    }
    class EvictionPolicy {
        <<abstract>>
        +select(scores, budget) ndarray
    }
    class SnapKVPolicy
    class PyramidKVPolicy
    class H2OPolicy
    class BudgetAllocator {
        +float ratio
        +layer_budget(l, L, total) int
    }
    class Scheduler {
        +step() Batch
        +add_request(req)
        +finished() list
    }
    class DraftModel {
        <<abstract>>
        +draft(ctx, K, width) DraftTree
    }
    class LayerSkipDrafter {
        +int exit_layer
        +draft(ctx, K, width) DraftTree
    }
    class NGramDrafter {
        +int n
        +draft(ctx, K, width) DraftTree
    }
    class AdaptiveController {
        +float alpha_hat
        +float cost_ratio
        +SpecState state
        +observe(K, gamma, t_draft, t_verify)
        +plan(batch_size) SpecPlan
    }
    class VerifyResult {
        +int n_accepted
        +list~int~ tokens
        +bool all_accepted
    }
    class Generator {
        +generate(prompt, cfg) Iterator~int~
        +speculative_step(ctx) VerifyResult
    }
    class BenchRunner {
        +run(matrix) list~BenchResult~
        +pareto_sweep(ranks) list~ParetoPoint~
    }

    LlamaModel *-- ModelConfig
    LlamaModel *-- BitLinear
    BitLinear *-- HybridGatedFlow
    Trainer --> LlamaModel
    Trainer --> ModelConfig
    PhosWriter ..> LlamaModel : export
    PhosWriter ..> Tokenizer
    PhosReader ..> Tokenizer
    PhosReader --> RuntimeWeights
    RuntimeWeights --> Backend
    Backend <|-- NumpyBackend
    Backend <|-- NativeBackend
    Scheduler --> BlockManager
    Scheduler --> PagedKVCache
    BlockManager --> PagedKVCache
    EvictionPolicy <|-- SnapKVPolicy
    EvictionPolicy <|-- PyramidKVPolicy
    EvictionPolicy <|-- H2OPolicy
    EvictionPolicy --> BudgetAllocator
    PagedKVCache --> EvictionPolicy
    DraftModel <|-- LayerSkipDrafter
    DraftModel <|-- NGramDrafter
    Generator --> DraftModel
    Generator --> AdaptiveController
    Generator --> PagedKVCache
    Generator --> Backend
    AdaptiveController ..> VerifyResult
    BenchRunner --> Generator
    Tokenizer --> ModelConfig
```

---

## 10. 数据流与关键算法选型摘要

| 环节 | 采用方案 | 来源 | 我们的差异 |
|---|---|---|---|
| 三值量化 | absmean + STE | BitNet b1.58 (arXiv:2402.17764) | 沿用；增加 per-block 粒度选项 |
| 网络结构 | SubLN + ReLU² FFN + 无 bias | BitNet b1.58 / a4.8 (2411.05805) | 沿用；ReLU² 与 SwiGLU 可切 |
| 内核 | TL / I2_S mpGEMM | bitnet.cpp (2502.11880) | **符号+索引 14 项 LUT**（16 项对齐 SIMD），比 three_lut 27 项省一半表 |
| 运行时 | GGUF 风格二进制 + RMSNorm/RoPE/GQA/KV | llama.cpp, llama2.c | 自定 `.phos`，字段为三值/HGF/早回头扩展 |
| 内存 | PagedAttention 分页 KV + 连续批处理 | vLLM (SOSP'23) | 移植到 NumPy/CPU；块大小 16 适配小模型 |
| KV 驱逐 | SnapKV / PyramidKV / H2O | 2404.14469 / 2406.02069 / 2306.14048 | **统一预算分配器**做成可插拔策略 + 逐层金字塔分配 |
| 投机验证 | 投机采样（分布严格等价） | Leviathan et al. 2023 | 沿用，加树状批量验证 |
| 草稿结构 | 树状草稿 + 概率剪枝 | EAGLE-2 (2406.16858) / Medusa (2401.10774) | 换成本项目 O1 自草稿 |
| 自草稿 | LayerSkip 早退 + 层 dropout | 2404.16710 | **草稿跑在三值权重上**（原作跑 FP） |
| 零成本草稿 | n-gram / prompt lookup | llama.cpp PR#18471 | 作为对照组 |
| 质量补偿 | HGF 低秩旁路 | Hybrid Gated Flow 思路 | **r 可调 → Pareto 曲线**（O3） |
| 自适应开关 | 在线 α / 成本比拟合 | 社区反模式共识 | **闭式收益模型 + 状态机 + batch 门控**（O2） |

---

## 11. 风险、未决与假设

| # | 项 | 风险 | 缓解 |
|---|---|---|---|
| R1 | torch CPU 版安装中（250MB），可能失败 | 训练路径不可用 | T2 前置校验 `import torch`；失败时给出明确指引，T3–T5（推理/评测）**不依赖 torch** 可继续 |
| R2 | `ziglang` 编译在 CI (ubuntu) 上未验证 | native 内核不可用 | 三级回退 + CI 增加「强制 fallback」job 验证 NumPy 路径 |
| R3 | 10M 参数 + tiny 语料，PPL 数值可能难看 | 基准说服力不足 | 报告里明确标注「玩具规模，看相对值不看绝对值」；提供 `--scale base` 走 30M |
| R4 | CPU 上投机解码加速比可能 < 1 | O2 关不掉就成负优化 | 这正是 O2 的价值：给出「何时不该开」的量化证据，作为交付物 |
| R5 | HGF 低秩因子用 f16 存，精度损失 | Pareto 曲线噪声 | 提供 `--hgf-dtype f32`；默认 f16 但 scale 始终 f32 |
| R6 | 三值量化后 PPL 退化可能 > 30% | 质量不可接受 | HGF r=4 兜底；同时报告 int8 基线作为「中间档」 |
| R7 | Windows `ctypes` + 大数组 GIL 释放 | 并行无效 | 每个线程独立输出缓冲；行块无重叠，无需锁 |

**已做的假设（需在实现中确认或推翻）**
1. 默认模型 `d_model=384, n_layer=6, n_head=6, n_kv_head=2, d_ff=1024, vocab=8192` ≈ **10.6M 参数**。
2. 默认语料：`TinyStories` 前 20 MB（或内置 5 MB 兜底），BPE `vocab_size=8192`。
3. 默认量化粒度：权重 **per-channel（按输出行）**，激活 **per-token**；`per_block(128)` 作为可选项。
4. 早退层 `E = ceil(L/2) = 3`（6 层模型）。
5. 分页 KV 块大小 16，默认驱逐策略 `pyramid`，预算 `0.5 × max_seq_len`。
6. 草稿默认 K=4、树宽 2（控制器 `auto` 下由 O2 决策）。
7. 嵌入表与 LM head 默认 **不量化**（f16），符合 BitNet 论文惯例。
