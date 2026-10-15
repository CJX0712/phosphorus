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

## 4. 生成样本（实测输出）

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

## 5. 质量门（实测）

| 检查 | 结果 |
|---|---|
| pytest | **26/26 通过** |
| ruff | 干净 |
| engine↔torch logits 相对误差 | < 5%（不变量 ⑦，忠实路径） |
| int8 激活路径 | 有限且与忠实路径 cos > 0.9 |

## 6. 复现命令

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
```

> 注：默认配置（9.4M 模型）面向多核服务器调参；在上述开发机上单步约 4 分钟，
> 不适合本机全量训练，故基准采用 `tiny_fast.json` 小型配置。
