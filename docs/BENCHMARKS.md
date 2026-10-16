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

## 5. 生成样本（实测输出）

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

## 6. 质量门（实测）

| 检查 | 结果 |
|---|---|
| pytest | **98/98 通过**（含 50 条 native 内核测试） |
| ruff check / format | 干净 |
| native vs numpy 位一致性 | int32 与 fp32 域均逐位相等（不变量 ①） |
| engine↔torch logits 相对误差 | < 5%（不变量 ⑦，忠实路径） |
| int8 激活路径 | 有限且与忠实路径 cos > 0.9 |

## 7. 复现命令

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
```

> 注：默认配置（9.4M 模型）面向多核服务器调参；在上述开发机上单步约 4 分钟，
> 不适合本机全量训练，故基准采用 `tiny_fast.json` 小型配置。
