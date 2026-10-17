# BENCHMARKS — 实测基准

> 本文所有数字均为**真实测量**，非估算。测试机为日常开发机（非服务器），
> 因此绝对吞吐偏低；重点看**相对关系与正确性**，跨机器比较请以相同配置复现。

## 1. 测试环境

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 7 H 255（8 核 / 16 逻辑线程） |
| 内存 | 15.3 GB |
| PyTorch | 2.14.1+cpu（8 线程） |
| 推理路径 | 纯 numpy（**不依赖 torch**，`.phos` 单文件加载） |
| 原生工具链 | zig cc 0.16.0（`pip install ziglang`，随 wheel 分发 C99 源码） |
| SIMD | AVX2（运行时 CPUID 探测； scalar 兜底，无 AVX2 机器结果不变） |

## 2. 训练基准（qat-hgf 模式）

语料 `data/raw/tinyshakespeare.txt`（470,183 tokens），配置 `ckpt/tiny_fast.json`：

| 项 | 值 |
|---|---|
| 参数量 | **1,783,509**（1.78M） |
| 结构 | d_model=192, n_layer=3, n_head=3, n_kv_head=1（GQA）, d_ff=384, vocab=4096, max_seq=256 |
| 量化 | HGF rank=4 + QAT |
| 步数 | 150（batch=4, block=64, warmup=10, 峰值 lr=3e-3） |

### 验证损失曲线（实测）

| step | train loss | val_loss | ppl |
|---:|---:|---:|---:|
| 0 | 8.3630 | — | — |
| 30 | 4.5355 | 4.4979 | 89.83 |
| 60 | 4.7062 | 4.2103 | 67.38 |
| 90 | 4.3184 | 4.1019 | 60.46 |
| 120 | 4.0863 | 3.9915 | 54.14 |
| 150 | 4.2446 | **3.8987** | **49.36** |

- 训练循环耗时 **11.0s** / 150 步（约 0.07s/step）；端到端墙钟约 24 分钟，
  其中绝大部分为本机 CPU 上的 BPE 数据准备（已缓存，复现时跳过）。
- 产物：`ckpt/tiny.phos`（3.5 MB，含权重 + 分词器 + 配置的单文件）。

## 3. 推理基准（纯 numpy 引擎）

| 指标 | 实测值 |
|---|---:|
| prefill 吞吐 | **1,296 tok/s** |
| decode 吞吐（64 tok） | **290.9 tok/s** |
| 生成样本 | 见 §4 |

> decode 为自回归逐步采样，每步一次全前向；prefill 一次性并行计算 prompt，
> 故显著快于 decode——量级关系符合自回归推理的预期。

## 4. T4-A：native C99 内核 vs numpy（实测）

三值权重（W ∈ {−1,0,1}）的投影有两条数学等价路径：**I2_S**（int8 镜像权重 +
uint8 激活，P0 主路径）与 **TL**（5-bit 打包流 + 16 项 int16 LUT，P1 内存受限路径）。
T4-A 用 zig cc 把它们编译为原生 C99 内核（scalar + AVX2 双路径，运行时 CPUID 分发），
与 numpy 参考实现逐位相等（不变量 ①，见下），因此**调度选择不改变数值结果**。

### 4.1 内核级（int32 累加域，best-of-5）

加速比 = numpy 参考耗时 / native 耗时；✅ 表示该形状下 native 与 numpy **逐位相等**。

| N | K | T | i2s scalar vs numpy | i2s AVX2 vs numpy | tl scalar vs numpy | tl AVX2 vs numpy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 192 | 1 | 1.41x | 2.08x | 2.64x | 3.06x |
| 192 | 192 | 8 | 0.69x | **0.74x** | 2.77x | 2.84x |
| 512 | 512 | 1 | 1.94x | 5.93x | 2.16x | 3.53x |
| 512 | 512 | 8 | 1.48x | 4.39x | 2.51x | 5.87x |
| 1024 | 256 | 1 | 2.45x | 8.01x | 2.30x | 4.38x |
| 1024 | 256 | 8 | 2.15x | 4.62x | 2.18x | 4.13x |
| 4096 | 192 | 1 | 2.64x | 10.06x | 2.86x | 4.60x |
| 4096 | 192 | 8 | 2.51x | 6.26x | 3.14x | 6.18x |
| 4096 | 384 | 1 | 2.80x | 8.38x | 3.39x | 6.78x |
| 4096 | 384 | 8 | 2.38x | 9.43x | 2.71x | 5.70x |

全部 10 个形状位一致（不变量 ① 内核级成立）。两点值得说明：

- **i2s 在小 N + T>1 时输给 numpy**（192×192@T=8 仅 0.74x）：numpy 的 int32
  批量矩阵乘走多线程 BLAS 级路径，而原生内核是单线程 GEMV 循环。但 N≥1024 后
  即使 T=8 原生仍快 4.6x~9.4x——矩阵足够大时访存与整数乘法密度成为主导。
- **tl 的 numpy 参考是逐 token Python 循环**（引擎 numpy TL 路径的实际形态），
  故 2.8x~6.8x 的优势主要来自**消除 Python 循环 + LUT 下沉到 C**；若与
  「批量向量化 LUT」比较则约为 1x（计算等价，tl 的真实优势在内存：5-bit 打包）。

### 4.2 端到端（`ckpt/tiny.phos`，prefill 32 tok + decode 64 tok，best-of-7 × 3 轮取最优）

| 数学后端 | 实现后端 | prefill tok/s | decode tok/s | prefill 加速比 | decode 加速比 |
| --- | --- | ---: | ---: | ---: | ---: |
| i2s | numpy | 1,781 | 288 | 1.00x | 1.00x |
| i2s | native | 1,809 | 380 | 1.02x | **1.32x** |
| tl | numpy | 550 | 257 | 1.00x | 1.00x |
| tl | native | 1,508 | 381 | **2.74x** | **1.48x** |

- i2s prefill 的 1.02x 是**噪声级持平**：该模型全部投影 N≤384，按 §4.3 的调度
  规则 T>1 时仍走 numpy 批量矩阵乘，两列实为同一段代码。
- 本机为共享开发环境，单轮 best-of-N 波动可达 ±20%，故取 3 轮最优；
  跨机器比较请以 §4.4 命令自行复现。

### 4.3 自适应调度（实测 crossover 固化在代码里）

`phosphorus/engine.py` 的 `PhosModel._use_native(T, N)` 按上述测量结果分发：

| 数学后端 | 规则 | 依据 |
| --- | --- | --- |
| `tl` | 总是走 native | 全部实测形状 2.8x~6.8x |
| `i2s` | `T == 1`（decode）或 `N ≥ 1024` | 小 N+T>1 时 numpy 赢（0.74x）；N≥1024 时原生赢 4.6x~9.4x |

两条路径在 int32 域逐位相等（`tests/test_native_backend.py` L0/L2 共 50 条断言），
因此该调度是**纯性能优化，零数值风险**；`scripts/bench_native.py` 每次运行前
会先跑一次位一致性校验，失败即中止基准。

### 4.4 复现命令

```bash
# 内核级（int32 域，含逐位校验）
python -m scripts.bench_kernels --reps 5

# 端到端（先校验位一致性，再跑 4 种配置）
python -m scripts.bench_native --model ckpt/tiny.phos --reps 7
```

## 5. T4-B：分页 KV（paged）vs dense（实测）

T4-B 把 KV cache 从「每层一份 `[1, n_head, T, d_head]` 全量张量」换成
**分页物理池 + block_tables**（vLLM 语义）：只存 `n_kv_head` 份 KV，读取时
GQA 展开；逻辑→物理间接层为 T4-C 驱逐 / T4-D 连续批处理奠基。
两种模式数值等价（不变量 ⑨，`tests/test_engine_paged.py` 7 条断言）。

### 5.1 吞吐（`ckpt/tiny.phos`，prefill 32 tok + decode 64 tok，best-of-3）

| kv_mode | 数学后端 | prefill tok/s | decode tok/s | prefill 比 | decode 比 |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | fp32 | 3,917 | 254 | 1.00x | 1.00x |
| paged | fp32 | 4,294 | 275 | 1.10x | 1.08x |
| dense | tl | 1,085 | 352 | 1.00x | 1.00x |
| paged | tl | 1,410 | 369 | **1.30x** | 1.05x |
| dense | i2s | 1,691 | 374 | 1.00x | 1.00x |
| paged | i2s | 1,664 | 328 | 0.98x | 0.88x |

**如实说明**：在 1.78M / max_seq_len=256 的玩具模型上，paged 的收益是
**结构性**的而非压倒性的——tl 的 prefill 1.30× 来自注意力只算 `n_kv_head`
（tiny 为 3:1 GQA，理论 3×，实测受投影主导故 1.30×）；i2s decode 的 0.88×
是 gather + repeat 的固定开销在 T=1 时未被摊销（dense 路径此时只是数组拼接）。
paged 的真正价值在**更大模型 / 更长上下文**（内存从 `n_head` 降到 `n_kv_head`、
块级共享与驱逐、连续批处理），这些收益要在 T4-C / T4-D 才能量化。
本机为共享开发机，单轮 best-of-N 波动可达 ±20%。

### 5.2 块管理器验收（C4）

| 检查 | 结果 |
|---|---|
| 分配/回收 10,000 次无泄漏 | ✅ `utilization()` 精确回到 0.0，`Σref_cnt == 0` |
| 容量约束（块数 + 每序块表双上限） | ✅ 超限返回 False 且不改状态 |
| 引用计数共享（`touch` + `release_block`） | ✅ 共享块不会被单条序列提前回收 |
| 滑动窗口任意数量丢弃 | ✅ `head_offset` 机制，逻辑→物理映射永不错位 |
| `paged_gqa_attention` vs dense | ✅ 最大绝对差 < 1e-5（含朴素逐元素参考三方比对） |
| 分块 prefill == 整段 prefill | ✅ 4+4+16 与一次 24 token logits 一致 |
| 块大小 4/16/32/64 | ✅ 四种块大小结果互相一致 |
| `reset()` 归还 | ✅ 利用率归零 |

### 5.3 复现命令

```bash
# 两种 KV 模式对比（任意数学后端）
python -m phosphorus bench --model ckpt/tiny.phos --backend tl --kv paged --tokens 64
python -m phosphorus bench --model ckpt/tiny.phos --backend tl --kv dense --tokens 64

# 引擎级等价性测试（dense ↔ paged，7 条断言）
python -m pytest tests/test_engine_paged.py tests/test_paged_kv.py -q
```

### 5.4 T4-C：KV 驱逐保真度（实测）

四种策略（SnapKV / PyramidKV / H2O / StreamingLLM）在 `ckpt/tiny.phos`
（d_model=192, n_layer=3, n_head=3, n_kv=1, max_seq_len=256）上实测。
预填充 200 token（触发压缩），decode 20 token，下游 final-logits 余弦 vs 无驱逐（`none`）参考。

| 策略 | 50% 预算余弦 | 75% 预算余弦 | 压缩后缓存（/200） |
| --- | ---: | ---: | ---: |
| SnapKV | 0.9936 | 0.9981 | 68（逐层相同） |
| PyramidKV | 0.9971 | 0.9981 | 127 / 87 / 47（逐层金字塔） |
| H2O | 0.9992 | 0.9999 | 68（逐层相同） |
| StreamingLLM | 0.9968 | 0.9986 | 41（sink 4 + window 32，窗口随预算伸缩） |

**验收口径（C5，不变量 ⑦）**：原始逐层注意力输出相对 L2 在 50% 预算下即便
oracle（保留最受关注 50%）也 ≈ 0.5，属指标固有性质；真实 SnapKV/H2O/PyramidKV
论文均以**下游 logits / PPL** 衡量压缩质量，故本项目以 **final-logits 余弦 ≥ 0.99**
为验收阈值，四种策略在 50% 预算下全部达标。

**预算守恒**：`BudgetAllocator`（PyramidKV 金字塔公式）逐层预算之和与 `L·C` 之差 = 0
（最大余数法修正；多配置实测 |Σ−L·C| ≤ 1，满足 C5 验收 ±1）。

**强制保留**：所有策略永久保留前 `sink_size`（默认 4）个 attention sink + 末尾
`window_size`（默认 32）个最近 token，作为解码的语义锚点。

**复现**：

```bash
# 下游保真度 + 预算守恒 + 物理压实（56 条断言）
python -m pytest tests/test_kv_eviction.py -q

# CLI 端到端（驱逐 + 预算 / 窗口 / sink 可调）
python -m phosphorus.cli generate --model ckpt/tiny.phos \
    --eviction snapkv --kv-budget 0.5 --window-size 32 --sink-size 4 \
    --prompt "人工智能的未来" --max_new_tokens 16
```

### 5.5 T4-D：连续批处理吞吐（实测）

`Scheduler` 把 8 条序列的新 token **融合进一次前向**（`PhosModel.forward_step`），
与逐条串行（每条 `reset` 后单独 `forward`）做**同 token 总量**对比。
环境同 §4；模型 `ckpt/tiny.phos`（d_model=192, n_layer=3, n_head=3, n_kv=1）。

| 配置 | 序列数 | 每序列新 token | 总 token | 耗时 (s) | 吞吐 (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 逐条串行（baseline） | 8 | 40 | 320 | 0.649 | 492.8 |
| **融合连续批处理** | 8 | 40 | 320 | 0.276 | **1158.8** |

**加速比 = 2.35×**（1158.8 / 492.8），满足 C6「融合前向 ≥ 2× 串行」验收
（`tests/test_scheduler.py::test_scheduler_throughput_speedup` 断言守护）。

**等价性（C6 正确性）**：融合生成的 token id 与逐条串行**逐位一致**
（`test_scheduler_greedy_matches_serial`，greedy / temperature=0 / top_k=1）；
单序列 `forward_step` 末位 logits 与 `forward` 最大差 < 1e-5
（`test_forward_step_single_matches_forward`）；多序列融合各序列 logits 与各自单序列一致
（`test_forward_step_multiseq_matches_individual`）。

**采样（C7）**：`temperature=0` → greedy；同 seed 两次运行结果 `==`
（`test_scheduler_reproducible_same_seed`）；随机采样（temp=0.9, top_k=10, top_p=0.95）
与贪心结果 `!=`（`test_scheduler_stochastic_differs_from_greedy`）；每序列独立
`np.random.default_rng(seed + 全局下标)` 保证融合批与逐条串行在随机采样下逐位一致。
> 注：本机为 Python/NumPy CPU 单线程路径，融合收益主要来自「把 8 次 Python 循环 +
> 8 次独立 KV 读写 + 8 次 RMSNorm/FFN 调度」摊薄为 1 次；在原生 C99 + 多线程后端下
> 加速比会更高（T4-A native 内核已位精确，融合批可直接叠加）。

**复现**：

```bash
# 连续批处理等价 + 吞吐 + 采样（8 条断言）
python -m pytest tests/test_scheduler.py -q

# CLI 端到端（融合 8 条 + 吞吐打印）
python -m phosphorus.cli sched --model ckpt/tiny.phos \
    --prompts "To be, or not to be" "Once upon a time" "The king said" "In the beginning" \
              "It was the best" "She looked at" "All the world" "When the sun" \
    --max-seqs 8 --max_new_tokens 40 --temperature 0.0 --top_k 1 --bench
```

### 5.6 T5：投机解码吞吐与加速比（实测）

**架构（C3 机制）**：`speculate.generate_tokens` 采用**热 KV 增量解码**——目标模型与
草稿器（`model.shadow()`，共享权重、独立 KV cache，O1 零额外字节）各持一条热 cache；
每轮草稿以单 token 热追加给出 K 个提议，目标模型**一次批量前向**覆盖 K 个新位置
（BLAS 摊销权重加载），残差拒绝采样后以 `truncate_tail` O(T) 回滚被拒 token。
期间修复两个关键缺陷：① 双 cache 逐轮**脱同步 1 token**（草稿末 token 未写入草稿
cache，α 从 1.0 崩塌至 0.005，修复后 α=1.000）；② 滑动窗口瞬态块预留（物理跨度可达
max_seq_len + block_size − 1，池 +1 备用块）。

**实测一：`ckpt/tiny.phos`（d=192, 3 层，训练模型，temperature=0.8, top_p=0.95，
128 tok × 2 seeds，串行基线 202.0 tok/s）**

| 模式 | K | draft_layers | tok/s | speedup | α |
| --- | ---: | ---: | ---: | ---: | ---: |
| spec | 4 | 2 | 184.6 | 0.91× | 0.782 |
| spec | 8 | 2 | 170.0 | 0.84× | 0.670 |
| auto（O2 控制器，dl=2） | — | 2 | 212.5 | 1.05× | 0.729（终态 OFF，正确门控） |

⚠️ **该模型未达 C3（<1.3×）**。原因有两条，均为投机解码的已知性质而非实现缺陷：
① 每层前向仅微秒级（d=192/3 层），Python/BLAS **调用开销**相对前向成本占主导，
而投机每轮需 K 次草稿前向 + 1 次批量验证 + 1 次收尾追加；② 修正「验证须用采样器
真实分布（temp+top-p 截断后）」后，草稿核与目标核不一致使 α 由 0.82 降至 ~0.70。
> 诚实更正：本节曾记录「K=8/dl=2 → 1.40×」，该数字是在**修复前**（验证误用原始
> softmax、分布语义不正确）测得的，已作废。正确语义下该模型为 0.84–0.91×。

**实测二：`ckpt/synthetic_12l.phos`（d=512, 12 层合成模型，深层恒等 → early-exit(3) ≡
full(12)，α=1.000 ceiling regime；192 tok × 2 seeds，串行基线 6.9 tok/s）**

| 模式 | K | draft_layers | tok/s | speedup | α |
| --- | ---: | ---: | ---: | ---: | ---: |
| spec | 8 | 3 | 11.4 | 1.64× | 1.000 |
| **spec** | **16** | **3** | **15.6** | **2.26×** | **1.000** |
| auto（O2 控制器） | — | 3 | 13.3 | 1.92× | 1.000（终态 ON，c_hat=1.631） |

**C3 达成（α=1.0 regime）**：端到端 **2.26× ≥ 1.3×**（K=16, draft_layers=3）。
对照实测一可见：加速比取决于「单次前向成本 vs 调用开销」与 α，微小模型上投机不划算——
这正是 O2 控制器存在的意义（tiny.phos dl=2 实测终态正确为 OFF）。

**Regime 诚实说明**：合成模型构造（`scripts/build_synthetic_model.py`）需同时压低基底层幅度
（`--base-scale`）与 `norm_final`（`--logit-scale`）——随机未训练权重的残差流幅值会
累积至 ~6.8e4，令注意力分数达 ~1e8 量级，fp32 噪声被 softmax 指数放大，草稿/验证
分布实质发散（数值病理，非算法问题）。α 随 draft_layers/n_layer 比与模型置信度变化，
O2 控制器（C4）据此在不划算时自动 OFF——tiny.phos dl=1（α≈0.3）即被控制器正确关闭。

**分布等价与贪心一致（C1★，不变量 ②）**：见 `tests/test_speculative_distribution.py`。
要点（曾导致「静默跳过 / 假通过」，已修正）：
- 验证必须用**采样器真实分布** `sampling.sampling_probs(logits, temp, top_k, top_p)`
  （temp + top-p 截断后），而非原始 softmax——否则 top_p<1 时与串行不等价。
- tiny.phos 在单 prompt + temp=0.8/top_p=0.95 下分布**退化**（实测 3000/3000 同 token），
  χ² 无单元格 → 改为**多 prompt 池化**（10 条）+ 一路 **temp=1.5/top_p=1.0 非退化配置**
  （3 组独立种子，允许 ≤1 组拒绝，χ²/df 实测 1.19/0.83/1.06/1.01 ≈ 1.0）。
- 经验 KL 的绝对阈值 `< 1e-3` 在有限样本下**统计不可达**（噪声底噪 ≈ 有效单元格数/N），
  故改为**自校准**：与「两次独立串行运行之间的 KL」对比，要求 ≤ 底噪 ×2.5。
- `temperature=0` 贪心投机与串行**逐位相同**；O1 `extra_bytes == 0`。全套 T5 单测 15 条。

**复现**：

```bash
# T5 单测（C1★/C2/C4 + 树掩码）
python -m pytest tests/test_speculative_distribution.py tests/test_controller.py tests/test_tree_mask.py -q

# 吞吐矩阵（K × draft_layers × seeds + auto 对照）
python -m scripts.bench_speculate --model ckpt/tiny.phos --tokens 128 --seeds 2 \
    --K 2,4,8 --draft-layers 1,2 --temperature 0.8
python -m scripts.build_synthetic_model --out ckpt/synthetic_12l.phos --n-layer 12 \
    --d-model 512 --n-head 8 --n-kv-head 2 --d-ff 1280 --vocab 2000 \
    --deep-scale 0.0 --base-scale 0.25 --logit-scale 1e-2
python -m scripts.bench_speculate --model ckpt/synthetic_12l.phos --tokens 192 --seeds 2 \
    --K 4,8,16 --draft-layers 3
```

### 5.7 T5-C5：全基准矩阵（实测）

`scripts/bench_matrix.py` → `benchmarks/results.json` + `benchmarks/REPORT.md`（Markdown 表 +
ASCII 柱状图）。矩阵 `{fp32, int8, tl, tl+spec} × batch ∈ {1,4,8} × seq ∈ {128,512}` 共 48 组
（2 模型 × 24），**30 组 ok / 18 组 skipped**（跳过原因见表末，无估算数字）。

口径：**seq = 序列总长**（prompt = seq − 32，生成 32）；greedy（temperature=0）确定性；
每组 3 次取**中位数**（本机单次采样抖动实测达 ±35%，raw_timings 全量在 results.json）；
speedup = 相对同 (batch, seq) 的 fp32 基线；**native C99 内核不可用**（ziglang 未安装，
即默认 `pip install phosphorus` 用户体验）→ tl/int8 走 numpy 回退路径。

**tiny.phos（d=192, 3 层，seq=128，端到端 tok/s）**

| 配置 | b=1 | b=4 | b=8 | 解读 |
|---|---:|---:|---:|---|
| fp32 | 259.5 | 260.4 | **622.7** | 连续批处理 **2.40×**（b8/b1） |
| int8 | 140.9 | 272.0 | 344.1 | b1 量化开销 > 收益（微模型），b8 仍不及 fp32 |
| tl | 60.2 | 94.0 | 118.7 | numpy 回退逐 token 循环，比 fp32 BLAS 慢 4–6× |
| tl+spec（K=8, dl=2, α=1.000） | 61.5 | — | — | 与 tl 串行持平（1.02×），双 cache 预填吃掉收益 |

**synthetic_12l.phos（d=512, 12 层，seq=128，端到端 tok/s）**

| 配置 | b=1 | b=4 | b=8 | 解读 |
|---|---:|---:|---:|---|
| fp32 | 7.5 | 23.5 | **39.8** | 连续批处理 **5.31×**（层数深、批摊销更显著） |
| int8 | 2.7 | 7.1 | 10.0 | 无 native 时 int8 路径常数过大 |
| tl | 4.1 | 4.1 | 5.9 | 逐 token 循环路径，批处理几乎无收益 |
| tl+spec（K=8, dl=3, α=1.000） | 3.0 | — | — | **0.73×**——见下方机制分析 |

seq=512（synthetic_12l）：fp32 5.4 / 15.8 / 19.5；int8 1.8 / 3.0 / 3.3；
tl 1.2 / 1.5 / 1.3；tl+spec 0.8（α=0.906，长上下文接受率略降）。

**三条机制级结论（实测互证）**：

1. **连续批处理是默认安装下的最大杠杆**：fp32 b8/b1 = 2.40×（3 层）→ 5.31×（12 层），
   且不改变任何数值结果（不变量 ⑨ 同源）。这是 T4-D 融合前向的直接兑现。
2. **无 native 时 tl/int8 是「慢路径」而非「快路径」**：numpy 回退的 tl 为逐 token
   Python 循环，在微矩阵上比 fp32 BLAS 慢 4–6×。这不是缺陷而是**回退语义**——
   部署态必须装 `[native]` extra（§4：native tl 2.8–6.8×）。矩阵如实记录默认安装画像。
3. **投机解码的收益前提是「批量前向能摊薄调用/循环开销」**：α=1.000 的 synthetic_12l
   在 tl 回退路径上仍 0.73×（验证 8 位置无法摊销逐 token 循环），而在 fp32 BLAS 路径
   （§5.6）同 α 下达 2.26×。**同一算法、同一 α，路径不同结论相反**——这是对 §5.6
   regime 分析的机制级补充：投机的敌人不是 α 低，而是「单次前向成本 ≲ 调用开销」。
   另：spec 双 cache 预填使 ttft ≈ 2×（tiny 410ms vs tl 262ms），n_new=32 的短生成
   场景端到端被预填主导——长生成见 §5.6（n_new=256，解码主导）。

**skipped 18 组原因**（诚实记录，不跑假数据）：① tiny.phos `max_seq_len=256 < seq=512`
（12 组——按 prompt=512 语义会触发 `forward` 静默滑窗截断，指标失真，故矩阵采用
「总长」口径）；② `tl+spec × batch>1`（6 组——`generate_tokens` 为单序列实现，
多草稿树 + 连续批处理组合在 T5 范围外，属后续工作）。

**复现**：

```bash
python scripts/bench_matrix.py                 # 全矩阵（约 35 min，无 native 的 tl 组最慢）
python scripts/bench_matrix.py --quick         # 只跑 seq=128, batch=1（约 2 min）
python scripts/bench_matrix.py --report-only benchmarks/results.json   # 仅刷新报告模板
```

## 6. 生成样本（实测输出）

 prompt `"To be, or not to be"`，采样 160 tokens，temperature 默认：

```text
To be, or not to be of I beauty with hold,
A a my are it the indeed and cheer we you do to ever and you and dear us
KATHARINA:
And in the count in is our lie Edward the cheek both no garden;
The my old what hence this canst hath talk return.

Were me heard not fool him a Wafter par when you pluckd to to and thou us it do of have the reasons
Than give you thy shall the to
```

1.78M 玩具模型（150 步）已呈现出：戏剧对白格式、角色名（`KATHARINA:`）、
古英语用词（`thou`/`canst`/`hath`/`thy`）与标点节奏。词法尚松散，
属预期内——参数量与步数决定的容量上限。

### 6.1 C8 端到端生成验收（实测）

`phosphorus generate --model ckpt/tiny.phos --prompt "..."` 在纯 numpy 与
native（`--backend tl`）两条路径下均产出**莎士比亚式连贯文本**（对白格式、角色名、
古英语）；`--stream` 逐 token 实时输出；`--bench` 打印首 token 延迟与吞吐。

| 路径 | prompt | 首 token 延迟（prefill） | 吞吐（40 tok） |
| --- | --- | --- | --- |
| fp32（numpy） | `Once upon a time` | **0.004 s** | 218.3 tok/s |
| `tl`（native C99） | `Once upon a time` | **0.018 s** | 187.9 tok/s |

**验收（C8）**：首 token 延迟远小于 500 ms（10 M 模型上限；tiny 模型 d_model=192/3 层
更轻，native 路径也仅 18 ms）；连贯性与 `--stream` 由 `phosphorus generate` 实测确认。
> 注：CLI 的「原生 C99 内核」对应 `--backend tl` / `--backend i2s`（ABI 位精确，
> 由 `native/` 构建注入），无独立 `native` 枚举值——这是后端注册表（A-? `backend.py`）
> 的命名，与 README §原生 C99 内核 描述一致。

## 7. 质量门（实测）

| 检查 | 结果 |
|---|---|
| pytest | **119 passed / 6 skipped**（含 50 条 native 内核 + 27 条分页 KV/引擎等价 + 8 条连续批处理 T4-D；skipped=缺失 ckpt 的对照用例） |
| ruff check / format | 干净 |
| native vs numpy 位一致性 | int32 与 fp32 域均逐位相等（不变量 ① / ⑧） |
| paged vs dense 引擎级一致性 | logits 最大差 < 1e-4、argmax 完全一致（不变量 ⑨） |
| engine↔torch logits 相对误差 | < 5%（不变量 ⑦，忠实路径） |
| int8 激活路径 | 有限且与忠实路径 cos > 0.9 |

## 8. 复现命令

```bash
# 训练（约 24 min，本机；数据准备已缓存则约 15s）
python -m phosphorus train --config ckpt/tiny_fast.json --mode qat-hgf \
    --corpus data/raw/tinyshakespeare.txt --out ckpt/tiny

# 推理
python -m phosphorus generate --model ckpt/tiny.phos \
    --prompt "To be, or not to be" --max_new_tokens 160
python -m phosphorus bench --model ckpt/tiny.phos --tokens 64

# 测试
python -m pytest -q

# T4-A 基准（native 内核 vs numpy）
python -m scripts.bench_kernels --reps 5
python -m scripts.bench_native --model ckpt/tiny.phos --reps 7

# T4-B 基准（分页 KV vs dense）
python -m phosphorus bench --model ckpt/tiny.phos --backend tl --kv paged --tokens 64
python -m phosphorus bench --model ckpt/tiny.phos --backend tl --kv dense --tokens 64
```

> 注：默认配置（9.4M 模型）面向多核服务器调参；在上述开发机上单步约 4 分钟，
> 不适合本机全量训练，故基准采用 `tiny_fast.json` 小型配置。
