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

## 7. 质量门（实测）

| 检查 | 结果 |
|---|---|
| pytest | **127/127 通过**（含 50 条 native 内核 + 27 条分页 KV/引擎等价测试） |
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
