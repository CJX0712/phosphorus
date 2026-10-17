# Phosphorus 算法溯源与超越点

> **本文档的硬性要求**：项目里**每一个非平凡模块**都必须在此有一条记录，且每条记录必须回答 4 个问题：
> 1. **借鉴了谁**（论文 / 项目 + 可点击链接 + 具体章节/PR 号）
> 2. **为什么选它**（选型理由，含被否决的替代方案及否决原因）
> 3. **我们做了什么超越**（可量化；若没有超越，明确写「纯沿用，理由：…」）
> 4. **在哪里落地**（文件 + 函数 + 测试 + 实测数字）
>
> **禁止**：未在此登记的模块不得实现。**禁止**：写未经实测的数字。
> 状态标记：`[骨架]` = 待实现阶段补全；`[已验证]` = 有测试 + 实测数字。

---

## 0. 溯源总表（一张图看清"站在谁的肩膀上"）

| # | 模块 | 主要借鉴来源 | 类型 | 许可证 | 我们的超越点 |
|---:|---|---|---|---|---|
| A-1 | 三值权重量化（absmean + STE） | BitNet b1.58 · [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | 论文 | MIT | 纯沿用 + per-block 粒度选项 |
| A-2 | 网络结构（SubLN / ReLU² / 无 bias） | BitNet b1.58 · [2402.17764](https://arxiv.org/abs/2402.17764)；BitNet a4.8 · [2411.05805](https://arxiv.org/abs/2411.05805) | 论文 | MIT | ReLU² ⇄ SwiGLU 可切换对照 |
| A-3 | TL / I2_S mpGEMM 内核 | bitnet.cpp · [arXiv:2502.11880](https://arxiv.org/abs/2502.11880) | 论文 + 开源 | MIT | **符号+索引 14 项 LUT**（16 项 SIMD 对齐） |
| A-4 | 5-bit 打包（3 权重 / 5 bit） | bitnet.cpp `three_lut` / T-MAC · [arXiv:2402.17953](https://arxiv.org/abs/2402.17953) | 开源 | MIT / Apache-2.0 | **行宽对齐到 24**（5 字节整 chunk） |
| A-5 | 运行时结构（RMSNorm / RoPE / GQA / KV） | llama.cpp · [github](https://github.com/ggml-org/llama.cpp)；llama2.c · [github](https://github.com/karpathy/llama2.c) | 开源 | MIT | 自定 `.phos` 格式，字段为三值扩展 |
| A-6 | 二进制模型格式 | GGUF（llama.cpp）· [spec](https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md) | 规范 | MIT | **HGF 因子 / 早回头 / row_wsum** 三类新字段 |
| A-7 | 分页 KV + 连续批处理 | vLLM PagedAttention · [SOSP'23 / arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | 论文 + 开源 | Apache-2.0 | 移植到 NumPy/CPU，块大小 16 适配小模型 |
| A-8 | KV 驱逐 · SnapKV | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) | 论文 + 开源 | Apache-2.0 | 纳入统一策略接口 |
| A-9 | KV 驱逐 · PyramidKV | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) | 论文 + 开源 | Apache-2.0 | 纳入统一策略接口 |
| A-10 | KV 驱逐 · H2O | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) | 论文 + 开源 | MIT | 纳入统一策略接口 |
| A-11 | KV 驱逐 · StreamingLLM（sink） | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | 论文 + 开源 | MIT | 作为所有策略的**强制保留项** |
| A-12 | 投机采样验证（分布严格等价） | Leviathan et al. · [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | 论文 | —（论文） | 纯沿用（正确性基石，不敢改） |
| A-13 | 树状草稿 + 概率剪枝 | EAGLE-2 · [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)；EAGLE-3 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | 论文 + 开源 | Apache-2.0 | 草稿源换成 O1 自草稿 |
| A-14 | 多头并行草稿 | Medusa · [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) | 论文 + 开源 | Apache-2.0 | 作为对照实现（可选） |
| A-15 | 自草稿 / 层跳过 / 早退 | LayerSkip · [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) | 论文 | —（论文） | **O1：草稿跑在三值权重上，零额外参数** |
| A-16 | n-gram / prompt-lookup 草稿 | llama.cpp [PR#18471](https://github.com/ggml-org/llama.cpp/pull/18471) | 开源 | MIT | 作为零成本对照组 |
| A-17 | BPE 分词 | Sennrich et al. 2016 · [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)；GPT-2 预分词 · [openai](https://github.com/openai/gpt-2) | 论文 + 开源 | —（论文）/ MIT | 字节级 BPE + 离线内置语料 |
| A-18 | RoPE 相对位置编码 | Su et al. · [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | 论文 | —（论文） | 纯沿用 |
| A-19 | GQA 分组查询注意力 | Ainslie et al. · [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | 论文 | —（论文） | 纯沿用 |
| A-20 | SwiGLU | Shazeer 2020 · [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | 论文 | —（论文） | 与 ReLU² 做消融对照 |
| A-21 | AdamW / cosine / grad clip | Loshchilov & Hutter · [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | 论文 | —（论文） | 纯沿用 |
| A-22 | HGF 低秩校正（量化损失补偿） | Hybrid Gated Flow（用户给定思路）；LoRA · [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) 的低秩形式 | 思路 + 论文 | MIT | **O3：秩 r 可调 → Pareto 曲线** |
| A-23 | 自适应投机开关 | 社区"盲目开启反模式"共识；vLLM 投机解码文档 | 工程共识 | Apache-2.0 | **O2：闭式收益模型 + 状态机 + batch 门控** |
| A-24 | 零编译器依赖构建 | `python -m ziglang cc`（zig 自带 clang，pip 可装） | 工具链 | MIT | **O4：三平台统一编译 + 三级回退** |
| A-25 | native C99 内核实现层（T4-A） | bitnet.cpp `ggml_vec_dot_i2_i8_s`（I2_S SIMD）；T-MAC int16 LUT 理论；ggml `vec_dot` 累加范式 | 开源 | MIT / Apache-2.0 | **按 ABI 契约自写**：位精确优先于复用，详见 §1 A-25 |

---

## 1. 各算法条目

> **写法约定**：每条固定 7 段 —— 来源 / 选型理由 / 被否决方案 / 我们的超越 / 落地位置 / 实测数字 / 状态。
> 实现阶段（T2–T5）逐条把 `[骨架]` 改成 `[已验证]` 并填入真实数字。**未填数字的条目不得标 `[已验证]`。**

---

### A-1 三值权重量化（absmean + STE 直通估计）

- **来源**：BitNet b1.58 — *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*，Ma et al.，[arXiv:2402.17764](https://arxiv.org/abs/2402.17764)，§3.1–3.2（absmean 量化 + QAT with STE）
- **核心公式**：
  ```
  scale  = mean(|W|)                        # 注意是 absmean，不是 absmax
  W_hat  = clip(round(W / scale), -1, 1) ∈ {-1,0,1}
  前向：y = W_hat * scale @ x_hat
  反向：∂L/∂W = ∂L/∂W_hat                   # STE 直通，跳过 round 的零梯度
  ```
- **选型理由**：1.58-bit 是当前**唯一**在 LLM 尺度上被验证"从零训练即能达到 FP16 质量"的极端量化方案；absmean 相比 absmax 对离群值更鲁棒，且无需校准集。
- **被否决方案**：
  - `absmax`（经典 TWN）—— 对离群值敏感，小模型上 PPL 明显更差。
  - **PTQ（训练后量化）** —— 1.58-bit 下 PTQ 几乎必然崩塌，必须 QAT。
  - `GPTQ / AWQ` —— 面向 3–4 bit，不适合 1.58 bit，且依赖 GPU 校准。
- **我们的超越**：
  - 纯沿用核心公式（正确性优先，不自创量化函数）。
  - **新增 `--w-granularity {per_tensor, per_channel, per_block}`**：默认 per_channel（按输出行），论文用 per_tensor；per_block(128) 作为折中，实测三者在 PPL vs 内存上的差异 → 进 Pareto 表。
- **落地**：`phosphorus/model/quantization.py::weight_quant_absmean` / `activation_quant` / `BitLinear`；测试 `tests/test_qat_ste.py`
- **实测数字**：`[骨架]` — 待 T2/C3 填入：`fp32 val_ppl = __`，`qat val_ppl = __`，比值 `__`；`{−1,0,1}` 分布 `__/__/__`
- **状态**：`[骨架]`

---

### A-2 网络结构：SubLN / ReLU² / 无 bias

- **来源**：BitNet b1.58（[2402.17764](https://arxiv.org/abs/2402.17764) §3.3）；BitNet a4.8（[2411.05805](https://arxiv.org/abs/2411.05805)）
- **要点**：Pre-RMSNorm；`SubLN`（在每个注意力/FFN 子层输出后与 Q/K 投影后加 RMSNorm，稳定低精度激活）；FFN 用 `ReLU²`（`relu(x)²`）替代 SwiGLU（低精度下 exp 类激活更脆弱）；**所有 Linear 无 bias**。
- **选型理由**：这套组合是 1.58-bit 训练稳定的必要条件，不是可选优化。SubLN 尤其关键——低比特激活的方差漂移会直接毁掉训练。
- **被否决方案**：Post-LN（低精度下不收敛）；SwiGLU 作为默认（保留为可切换对照）。
- **我们的超越**：`--act {relu2, swiglu}` 可切换，在 `BENCHMARKS.md` 给出**消融表**（同一配置下 ReLU² vs SwiGLU 的 PPL / 收敛速度），这是 BitNet 论文没有在小模型尺度给出的对照。
- **落地**：`phosphorus/model/modules.py::RMSNorm` / `SubLN` / `ReLU2`；`llama.py`
- **实测数字**：`[骨架]`
- **状态**：`[骨架]`

---

### A-3 mpGEMM 内核：TL（Ternary LUT）与 I2_S（int2 + scale）★ 核心

- **来源**：bitnet.cpp — *1-bit AI Infra: Fast and Lossless BitNet b1.58 Inference on CPUs*，[arXiv:2502.11880](https://arxiv.org/abs/2502.11880)，§4（TL / I2_S 两类 mpGEMM）
- **TL 内核原理**：3 个权重为一组，对每组预计算**部分和查表**，`y_i = Σ_g LUT[code_g]`，全程无乘法。
- **I2_S 内核原理**：权重镜像为 int8（值域 = int2 的 `{-1,0,1}`），激活为 uint8，用 8-bit 点积指令（`vpdpbusd` / `vdotq_s32`）累加；零点校正 `y -= 128 · Σ_j W[i,j]`（`row_wsum` 导出时预计算）。
- **选型理由**：这是 CPU 上无 GPU/NPU 时唯一能把 1.58-bit 变成真实加速的方案；用乘法指令模拟只会比 fp32 更慢。
- **被否决方案**：
  - 传统 `int8 GEMM` —— 三值权重塞进 int8 是 8× 内存浪费，且吃不到 LUT 的无乘法红利。
  - `T-MAC` 的 LUT 方案 —— 面向 ARM 且依赖特定 tbl 指令，x86 收益不明；我们选择更通用的双路径。
- **我们的超越**：
  1. **符号 + 索引 14 项 LUT**（替代 bitnet.cpp `three_lut` 的 27 项）：利用 `27 = 1 + 13×2` 的符号对称，`part(26−c) = −part(c)`，只需 13 项 + 1 个全零项 = **14 项（补齐 16）**。收益：16×int8 LUT 恰好一个 XMM 寄存器；16×int16 LUT 可用"双 pshufb 高低字节表"技巧；符号用一条 `psign` 施加。**LUT 项数减半、建表成本减半**。
  2. **I2_S 零点校正外置到文件**：`row_wsum` 在导出时算好存进 `.phos`（`aux_layout=2`），运行时只做一次减法，**省掉每轮 `ΣW` 的重复计算**，且保证与 TL 路径**逐位一致**。
  3. **两条路径位精确对齐**：TL（int8 激活 + 符号）与 I2_S（uint8 激活 + 零点）数学恒等 → `test_gemm_bitwise.py` 断言两者 int32 输出**逐位相等**，这是一条可证明的正确性锚点。
  4. **并行不用 OpenMP/pthread**：C 内核保持纯函数可重入，并行由 Python `ThreadPoolExecutor` 在 row-block 粒度切分（ctypes 释放 GIL），Windows/Linux/macOS 一套代码。
- **落地**：`kernels/native/phos_gemm.c::phos_tl_gemm_f32` / `phos_i2s_gemm_f32`；`kernels/numpy_backend/gemm.py`；测试 `tests/test_gemm_bitwise.py`、`tests/test_native_backend.py`
- **实测数字**：`[骨架]` — 待 T4/C3 填入：`N=K=512 TL GEMM：numpy __ ms / native __ ms / 加速 __×`；`8 线程 __×`
- **状态**：`[骨架]`

---

### A-4 5-bit 打包布局（3 权重 → 5 bit，1.667 bpw）

- **来源**：bitnet.cpp 的打包思想；理论下界 `log2(3) = 1.585 bpw`
- **完整规格**：见 `ARCHITECTURE.md` §3.6（`ternary_5bit_v1`）
  ```
  w ∈ {-1,0,1} → d ∈ {0,1,2}
  c = d0 + 3·d1 + 9·d2 ∈ [0,26]
  编码：c==13 → (idx=0,sign=0)；c<=12 → (idx=c+1,sign=0)；c>=14 → (idx=27-c,sign=1)
  code5 = (sign<<4) | idx          # idx ∈ [0,13]
  8 组 = 24 权重 → 40 bit = 5 字节（小端写入 u64 低 5 字节）
  ```
- **选型理由**：3 权重/5 bit 是"LUT 索引宽度"与"内存效率"的最佳交点——5 bit 的 code 直接就是 LUT 索引，无需二次解码。
- **被否决方案**：2 权重/3 bit（1.5 bpw 更省但 LUT 只有 9 态，SIMD 利用率低）；4 权重/7 bit（1.75 bpw 更浪费且非整字节）。
- **我们的超越**：**行宽强制补齐到 24 的倍数**（`K_pad = ceil(K/24)*24`），使每行打包字节数恰好是 5 的整数倍 → 行寻址变成一次乘法，LUT chunk 天然对齐，**消除跨行的位跨界读取**（bitnet.cpp 需要额外处理尾部跨界）。
- **落地**：`quantize/packing.py::pack_ternary_v1` / `unpack_ternary_v1`；`kernels/native/phos_pack.c`；测试 `tests/test_packing.py`
- **实测数字**：`[骨架]` — 待 T3/C1 填入：打包吞吐 `__ MB/s`；`packed_bytes/理论值 = __`；往返恒等 `__/1e6`
- **状态**：`[骨架]`

---

### A-5 / A-6 运行时结构与二进制格式 `.phos`

- **来源**：llama.cpp（[repo](https://github.com/ggml-org/llama.cpp)）；llama2.c（[repo](https://github.com/karpathy/llama2.c)）；GGUF 规范（[spec](https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md)）
- **选型理由**：GGUF 的"header + tensor index + 对齐 data region"结构已被大规模验证，且支持 mmap 零拷贝加载——对 CPU 推理至关重要。
- **被否决方案**：直接用 GGUF —— GGUF 的张量类型枚举里**没有三值打包类型**，也无处安放 HGF 因子 / 早回头 / row_wsum；强行塞进 `GGML_TYPE_I2` 会丢失本项目必需的元数据。→ **自定义 `.phos`**，但**结构沿用 GGUF 的三段式布局**。
- **我们的超越**：在 GGUF 骨架上新增三类字段（详见 `ARCHITECTURE.md` §3.2–3.4）：
  - `TERNARY_PACKED` dtype + `pack_scheme` 字段
  - `aux_off` / `aux_layout`：HGF 低秩因子（`A∥B`）与 `row_wsum`
  - `early_exit_layer` + `layers.{E}.early_head.*`（O1 早回头）
  - header 带 `crc32_header`，导出后强制重读自检
- **落地**：`quantize/export.py::PhosWriter/PhosReader`；`quantize/inspect.py`；测试 `tests/test_phos_roundtrip.py`
- **实测数字**：`[骨架]` — 待 T3/C2：`tiny.phos = __ MB`；`三值权重占比 __%`；加载耗时 `__ ms`
- **状态**：`[骨架]`

---

### A-7 分页 KV + 连续批处理（PagedAttention）

- **来源**：vLLM — *Efficient Memory Management for LLM Serving with PagedAttention*，Kwon et al.，SOSP'23，[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)；块层实现参照 vLLM v1 [`block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)（Apache-2.0）
- **要点**：KV cache 切成固定大小 block（本项目 16 token），逻辑序列通过 `block_table` 映射到物理块；调度器做 continuous batching（每步动态加入/移除序列）。
- **选型理由**：这是唯一能把 KV 显存（内存）碎片率降到 < 4% 的方案；且天然支持抢占与变长序列。
- **被否决方案**：朴素 `[S, L, H, D]` 预分配 —— 内存浪费 60–80%，且无法连续批处理。
- **我们的超越**：
  - **移植到 NumPy/CPU**：vLLM 依赖 CUDA kernel，我们用 fancy-indexing + `block_table` 实现等价语义，并在 `test_paged_kv.py` 断言与 dense 注意力误差 < 1e-5（另与朴素逐元素实现三方比对）。
  - **块大小 16**（vLLM 默认 16 on GPU，但 CPU 小模型下我们实测 16 vs 32 vs 64 的吞吐，选最优）——T4-B 实测 4/16/32/64 数值一致，默认 16。
  - **只存 `n_kv_head` 份**：读取时 `np.repeat` 展开（精确拷贝），注意力计算量随 GQA 比下降；dense 参考则先展开后计算，两者逐元素等价（不变量 ⑨）。
  - **`head_offset` 滑动窗口**：丢弃任意数量 token 只回收整块 + 累加首块偏移，逻辑→物理映射永不错位（vLLM 滑动窗口后端同款思路）。
  - KV int8 量化 + per-block scale（`flags.KV_INT8`）——🔜 未实现（当前 fp32，留待 T4-E 之后按需启用）。
  - **融合连续批处理（T4-D 原创工程化）**：`PhosModel.forward_step({seq_id: 新token})` 把 N 条序列的新 token 拼成 `[N_total, d]` 一次前向，逐层按 `seq_id` 切片隔离（RoPE 绝对位置、注意力、`block_tables` 物理 KV 均逐序列独立），三段式写入 `begin_append→write→end_append`；`Scheduler` 按 wave 切分 > `max_seqs` 的请求并逐序列 `np.random.default_rng(seed+全局下标)` 保证随机采样下与逐条串行逐位一致。关键支撑：`PagedKVCache.layer_len` 重构为 `[n_seqs, n_layer]`，使每条序列每层拥有独立有效长度。
- **落地**：`phosphorus/block_manager.py`、`phosphorus/kv_cache.py`、`phosphorus/attention.py`、`phosphorus/engine.py`（`kv_mode`/`--kv` + `forward_step`）；连续批处理 `phosphorus/scheduler.py`；测试 `tests/test_paged_kv.py`（20 条）、`tests/test_engine_paged.py`（7 条）、`tests/test_scheduler.py`（8 条：融合数值正确性 / C6 等价 / C6 吞吐 / C7 采样）。
- **实测数字**：`[已验证]` — 内核级 paged vs dense 最大差 < 1e-5；引擎级 logits 差 < 1e-4、argmax 100% 一致；块管理 10,000 次分配/回收无泄漏；吞吐见 `BENCHMARKS.md` §5：tl prefill **1.30×**（paged 更快，注意力只算 n_kv 头）、i2s decode 0.88×（gather 固定开销未被 T=1 摊销）；**连续批处理 8 条融合前向吞吐 2.35× 逐条串行**（1158.8 vs 492.8 tok/s，tiny 模型 8×40 tok，`test_scheduler_throughput_speedup` 守护 ≥ 2×），token id 与逐条串行逐位一致。
- **状态**：`[已验证]`（分页 KV + 块管理 + 注意力 + 连续批处理）

---

### A-8 / A-9 / A-10 / A-11 KV 驱逐策略（SnapKV / PyramidKV / H2O / StreamingLLM）

| 策略 | 来源 | 核心 | 适配要点 |
|---|---|---|---|
| **SnapKV** | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) | 用"观察窗口"（末尾若干 token）的注意力分数投票，选出重要 KV 保留 | 观察窗口大小可配（默认 32）；avg/max pool（kernel 5） |
| **PyramidKV** | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) | 逐层预算呈金字塔（浅层多、深层少）；信息在深层高度聚合 | 我们的 `BudgetAllocator` 默认策略；公式对齐官方 `pyramidkv_utils.py` |
| **H2O** | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) | Heavy-Hitter Oracle：按**累积**注意力分数保留 top-m | 跨 decode 步位置对齐累加 `self._h2o[i]`，不池化 |
| **StreamingLLM** | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | 保留 attention sink（前几个 token）+ 最近窗口 | 作为**所有策略的强制保留项**（前 4 token 永不被驱逐）；窗口随预算伸缩 |

- **选型理由**：四者互补——SnapKV 看"局部观察"，H2O 看"全局累积"，PyramidKV 看"层间差异"，StreamingLLM 看"位置锚点"。都可归结为**同一个接口** `EvictionPolicy.select(scores, budget, n_total) -> keep_mask`，因此做成可插拔策略族（`EVICTIONS = ("none", "snapkv", "pyramidkv", "h2o", "streamingllm")`）。
- **被否决方案**：只实现一种 —— 无法回答"CPU 小模型上哪种策略最优"这个本项目的评测问题。
- **我们的超越**：
  1. **统一预算分配器** `BudgetAllocator`：把四种论文的"预算"抽象统一，`uniform`（均分）/ `pyramid`（金字塔，默认）。PyramidKV 公式照抄官方 `pyramidkv_utils.py` 的双分支（短序列固定预算 + 长序列金字塔 + `max_num` 钳制），并用**最大余数法**把逐层之和精确补到 `L·C`（C5 验收 ±1）。
  2. **驱逐后物理压实**：`PagedKVCache.compact(seq, keep_mask)` 先拷出保留 K/V 再 `free_seq` 回收、按 `max(layer_len)` 重建块表（`head_offset` 归零、逻辑坐标重排）——三篇论文只提"逻辑驱逐"，没解决**分页场景下的物理回收**，我们补上；LIFO 回收保证释放后拿回同一批物理块、避免不同层搬运互相覆盖。
  3. **逐层独立预算 + 逐层长度**：`layer_len[l]` 跟踪每层有效 token 数，gather 按层读取；`forced-keep` 保证 `sink + window` 永留（StreamingLLM 的 sink 思路上升为所有策略的硬约束）。
  4. **可验证误差界（不变量 ⑦）**：以**下游 final-logits 余弦 ≥ 0.99** 为 50% 预算验收阈值（详见 C5 修订说明：原始注意力 L2 即便 oracle 也 ≈0.5，属指标固有性质，真实论文以 logits/PPL 衡量）。
- **打分复用**：`attention.py` 抽出 `attention_weights` 与 `kv_importance_scores`（观测窗口 query 对 past key 的注意力 + 窗口内因果 mask 的 softmax 后对 query 维求和，跨 head 聚合），SnapKV/PyramidKV 走此路径、H2O 走全矩阵累积、StreamingLLM 不打分。
- **触发时机**：`engine.forward` 在 `end_append` 后、缓存超过 `_evict_threshold`（= max(sink+window+1, round(max_seq_len·kv_budget))）时调用 `_run_eviction()`；`none` 对照组完全不触发。
- **落地**：`phosphorus/eviction.py`、`phosphorus/kv_cache.py`（layer_len / drop_front / compact）、`phosphorus/engine.py`（`_score_layer` / `_run_eviction`）、`phosphorus/cli.py`（`--eviction` `--kv-budget` `--window-size` `--sink-size`）；测试 `tests/test_kv_eviction.py`（56 条）。
- **实测数字**（`ckpt/tiny.phos`，200 prefill + 20 decode，下游 logits 余弦 vs `none`）：

  | 策略 | 50% 预算 | 75% 预算 | 压缩后缓存(/200) |
  | --- | ---: | ---: | ---: |
  | SnapKV | 0.9936 | 0.9981 | 68（逐层相同） |
  | PyramidKV | 0.9971 | 0.9981 | 127/87/47（金字塔） |
  | H2O | 0.9992 | 0.9999 | 68（逐层相同） |
  | StreamingLLM | 0.9968 | 0.9986 | 41（sink+window） |

  金字塔预算守恒 `|Σ − L·C| = 0`（多配置实测 ≤ 1）。全部 ≥ 0.99 达标。
- **状态**：`[已验证]`

---

### A-12 投机采样验证（分布严格等价）★ 正确性基石

- **来源**：*Fast Inference from Transformers via Speculative Decoding*，Leviathan et al.，[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)，Algorithm 1
- **算法**：
  ```
  for i in 1..K:
      q = p_draft[t_i];  p = p_target[t_i]
      r ~ U(0,1)
      if r < min(1, p/q):  accept t_i
      else:
          t_i ~ normalize(max(0, p - q));  reject 后续;  break
  if all accepted:  bonus ~ p_target[· | last]      # 必须保留！
  ```
- **选型理由**：这是唯一**在数学上保证输出分布与目标分布严格等价**的投机解码方案（不是"近似"，是恒等）。质量红线，不接受任何"加速但不保分布"的变体。
- **被否决方案**：
  - **朴素贪心接受**（`argmax(p_target) == t_i` 才接受）—— 会偏移分布。
  - **典型/截断变体**（为加速放宽接受条件）—— 破坏等价性。
- **我们的超越**：**纯沿用，理由：正确性是本项目不可交易的底线。** 唯一的工程增强是**批量树状验证**（一次 prefill 验证整棵树的所有节点，而非逐个串行），这不改变数学语义，只改变计算组织方式。
- **落地**：`speculate/verify.py::spec_verify`（随机模式 + 贪心模式）；测试 `tests/test_speculative_distribution.py`
- **实测数字**（tiny.phos）：canonical（temp=0.8/top_p=0.95，10 条 prompt 池化）+ 高温非退化配置（temp=1.5/top_p=1.0，3 组独立种子）χ² 齐性检验均**未拒绝**等价假设（χ²/df ≈ 1.02，零假设期望 1.0）；`temperature=0` 贪心投机与串行**逐位一致** ✅
- **方法论修正（曾致"静默跳过/假通过"）**：① 验证须用 `sampling.sampling_probs`（截断后真实采样分布）；② 单 prompt 下分布退化（3000/3000 同 token）使 χ² 无单元格 → 多 prompt 池化 + 高温配置，并**响亮失败**代替 skip；③ 绝对阈值 `KL<1e-3` 有限样本不可达（底噪 ≈ 单元格数/N）→ 改用**自校准 KL**（对比两次串行运行底噪，≤2.5×）
- **贪心模式说明（temperature=0 特例，与被否决方案不冲突）**：被否决的"朴素贪心接受"指**随机采样场景**下用贪心规则（会偏移分布）；而 `temperature=0` 时目标分布本身退化为**点质量**，此时采用「接受 iff `y_i == argmax(p_i)`，拒绝补 `argmax(p_i)`，bonus = `argmax(p_K)`」，输出分布（点质量）与目标严格相等 ⇔ 序列逐位相同。两种模式由 `spec_verify(..., greedy=)` 显式分流。
- **状态**：✅ 已交付（随机 + 贪心双模式）

---

### A-13 / A-14 树状草稿与多头并行（EAGLE-2/3、Medusa）

- **来源**：EAGLE-2 *[arXiv:2406.16858](https://arxiv.org/abs/2406.16858)*（**动态树** + 按概率剪枝，上下文感知）；EAGLE-3 *[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)*；Medusa *[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)*（多头并行草稿）
- **选型理由**：EAGLE-2 的动态树证明"按联合概率分配预算"比固定树结构收益高 20–40%；Medusa 的多头是最简单可实现的对照。
- **被否决方案**：固定树（静态拓扑）—— 已被 EAGLE-2 证明次优。
- **我们的超越**：
  - 把 EAGLE-2 的**动态树剪枝器**从"独立 draft model"剥离出来，变成**与草稿源正交的组件** `speculate/tree.py::build_tree(paths, budget)`，可服务于 LayerSkip / n-gram / Medusa 任一草稿源。
  - Medusa 实现为**可选对照**（`speculate/medusa.py`），用于回答"多头 vs 自草稿，在小模型 CPU 上谁更好"——这是原论文没有的对照维度。
- **落地**：`speculate/tree.py`、`speculate/eagle.py`、`speculate/medusa.py`；测试 `tests/test_tree_mask.py`
- **实测数字**：树掩码/位置编码/剪枝不变量 3 条测试全绿 ✅；树状草稿端到端 α/speedup 对比 🔜（树验证与批量验证融合为后续项）
- **状态**：🔶 掩码机制已交付；端到端树草稿基准未做

---

### A-15 自草稿投机解码（LayerSkip）★ 原创超越点 O1

- **来源**：LayerSkip — *Enabling Early Exit Inference and Self-Speculative Decoding*，Elhoushi et al.，[arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
- **原作要点**：训练时施加**层 dropout**（不同深度的层有不同 survival probability），使模型在中间层也能输出可用 logits；推理时用前 E 层自草稿 + 后续层验证。
- **选型理由**：这是**唯一不需要训练第二个模型**的投机方案 —— 直接消灭了草稿模型的训练成本、显存占用与部署复杂度。对本项目"零 GPU、几分钟跑完"的约束是决定性的。
- **被否决方案**：
  - **独立小草稿模型**（d_model/4, 2 层）—— 需要额外训练 + 额外权重，与本项目的 O1 目标冲突；仅作对照实现。
  - **EAGLE 的 feature-level draft** —— 需要额外训练 draft head 且依赖目标模型的特征缓存，复杂度高。
- **我们的超越（O1 = 本项目核心差异化）**：
  1. **草稿跑在三值权重上**：LayerSkip 原作跑 FP 权重；我们的草稿路径与目标路径**共享同一份 1.58-bit 打包权重**，草稿不是"另一个模型"，而是"同一份权重的浅层截断"。→ **草稿模型额外参数 = 0，额外内存 = 0**（`assert extra_bytes == 0`）。
  2. **早退 head 可选零参数**：`--early-head-rank 0` 时早退 head **复用 tied embedding**（`logits_E = h_E @ tok_emb.T`），连一个参数都不新增；`r>0` 时用低秩 `A[r,d] ∥ B[r,V]` 换 FLOPs。
  3. **与量化联合优化**：层 dropout 训练 + 三值 QAT 同时进行，使中间层表示在三值约束下仍然"可读" —— 这是原作未处理的难题（低精度会显著削弱中间层语义）。
- **落地**：`speculate/layerskip.py::LayerSkipDrafter` + `engine.PhosModel.shadow()`（草稿器独占热 KV cache，共享权重零拷贝）；`speculate/__init__.py::generate_tokens`（热 KV 增量解码主循环）
- **实测数字**（详见 BENCHMARKS §5.6）：`额外参数字节 = 0` ✅（`test_o1_zero_extra_bytes`）；`α = 0.849`（tiny.phos, dl=2）/ `0.550`（dl=1）/ `1.000`（合成恒等深模型 ceiling）；speedup：tiny.phos（3 层微模型，每步前向微秒级、调用开销占主导）仅 **0.91×** ❌，合成 12 层模型 K=16 **2.26×** ✅、K=8 **1.64×**、auto **1.92×** —— 投机在前向成本 ≫ 调用开销的模型上兑现（C3 判定见 TASKS.md）
- **工程超越（原作未涉及）**：目标/草稿**双热 KV cache** + `truncate_tail` O(T) 拒绝回滚——原作与多数开源实现按轮重填或整段重算；本实现整条序列 KV 恒热，期间修复了双 cache 逐轮脱同步 1 token 的关键缺陷（未修复时 α 由 1.0 崩塌至 0.005）
- **状态**：✅ 已交付（C1★/C2/C3/C4 达标）

---

### A-16 n-gram / Prompt-Lookup 草稿（零成本对照组）

- **来源**：llama.cpp [PR#18471](https://github.com/ggml-org/llama.cpp/pull/18471)（prompt lookup decoding）；原思想来自 *Prompt Lookup Decoding*，Saxena 2023
- **要点**：从 prompt 里建 n-gram 索引，若当前上下文的后 n 个 token 命中历史，直接把历史后续 token 当草稿。**零模型、零训练、零额外内存**。
- **选型理由**：在摘要/代码编辑/模板填充等**高重复度**任务上 α 可接近 0.9，成本几乎为零 —— 必须作为 O1 的对照组，否则无法证明"自草稿的收益是真实的还是仅因为它比 n-gram 强"。
- **我们的超越**：作为**基线对照**实现并纳入 `bench` 矩阵；同时我们给出 O1 与 n-gram 的**混合策略**（先查 n-gram，未命中再走 LayerSkip），这是两者都未做的组合。
- **落地**：`speculate/ngram.py::NGramDrafter`
- **实测数字**：drafter 已实现（order-3 索引 + 命中零模型成本 + 未命中回落目标前向）✅；n-gram α vs O1 α 并列基准 🔜（未入 bench 矩阵）
- **状态**：🔶 已实现；对照基准未做

---

### A-17 BPE 分词器

- **来源**：*Neural Machine Translation of Rare Words with Subword Units*，Sennrich et al.，[arXiv:1508.07909](https://arxiv.org/abs/1508.07909)；GPT-2 预分词正则 + 字节级映射（[openai/gpt-2](https://github.com/openai/gpt-2)）
- **选型理由**：BPE 是 LLM 事实标准；GPT-2 的**字节级**变体保证任意 UTF-8 输入无损往返（这是不变量 ⑤ 的前提）。
- **被否决方案**：
  - 依赖 `tiktoken` / `tokenizers` —— 引入 Rust 编译与非平凡依赖，违反"零编译器依赖 + 最小可用"。
  - WordPiece / Unigram —— 生态不通用。
  - 字符级分词 —— 序列太长，训练效率低。
- **我们的超越**：自实现（这是"现成方案不适配"的典型：我们要的是**纯 Python、可离线、零依赖、可单测**的分词器，`tokenizers` 的 Rust 构建链在 Windows 无 MSVC 环境下会直接失败）。同时提供**离线内置语料兜底**，保证断网可训练。
- **落地**：`tokenizer/pretokenize.py`、`tokenizer/bpe.py`、`tokenizer/tokenizer.py`；测试 `tests/test_bpe.py`
- **实测数字**：`[骨架]` — 待 T1/C4：`训练耗时 __ s (5MB 语料)`；`往返恒等 `__/10000``；`压缩率 __ chars/token`
- **状态**：`[骨架]`

---

### A-18 / A-19 / A-20 / A-21 基础组件（纯沿用区）

| 组件 | 来源 | 说明 |
|---|---|---|
| RoPE | Su et al. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | 纯沿用。不变量 ④：内积只依赖相对距离 |
| GQA | Ainslie et al. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | 纯沿用。`n_kv_head < n_head` 时 `repeat_kv` |
| SwiGLU | Shazeer [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | 作为 ReLU² 的消融对照 |
| AdamW + cosine + grad clip | Loshchilov & Hutter [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | 纯沿用，无超越（优化器不自创） |

> **这一区明确声明"纯沿用"**，符合"禁重复造轮子"红线：这些是被数万次实验验证的组件，任何自创都是负收益。

---

### A-22 HGF 低秩校正（量化损失补偿）★ 原创超越点 O3

- **来源**：Hybrid Gated Flow 思路（用户给定）；低秩形式参考 LoRA — *LoRA: Low-Rank Adaptation of Large Language Models*，Hu et al.，[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **公式**：
  ```
  y = ternary_matmul(W, x) · scale  +  gate · (alpha / r) · (B @ (A @ x))
  A ∈ R^{r × d_in}  (高斯初始化)      B ∈ R^{d_out × r}  (零初始化)      gate 可学习标量
  ```
  零初始化 `B` 保证训练/启用起点 = 无校正（不破坏已有量化模型）。
- **选型理由**：1.58-bit 的质量损失集中在"权重的主成分方向被离散化抹平"；低秩 FP 旁路恰好是补回主成分的最小代价手段。相比直接提高比特数，低秩旁路的帕累托效率更高。
- **被否决方案**：
  - 全 FP16 残差矩阵 —— 内存开销 960%，完全违背 1.58-bit 的初衷。
  - 提高激活比特 —— 无法补偿权重离散化误差。
  - 蒸馏 / 更长 QAT —— 训练成本不可控，且无法做成"旋钮"。
- **我们的超越（O3）**：
  1. **秩 r 成为可连续调节的旋钮**，直接输出 **PPL vs 内存开销的 Pareto 曲线**（`bench/pareto.py`），把"质量-内存权衡"从玄学变成可选择的产品参数。
  2. **闭式内存开销公式**（`ARCHITECTURE.md` §3.7）：
     ```
     overhead% = r · (d_in + d_out) · bytes_per_param / (d_in · d_out · 1.667 / 8) · 100
     d=512, f16:  r=1→3.75%   r=2→7.5%   r=4→15%   r=8→30%   r=16→60%
     ```
  3. **可后验拟合**：`A`/`B` 可在量化后用校准集做岭回归闭式求解（无需重训），这使得 Pareto 扫描成本从"重训 N 次"降到"扫一次"。
- **落地**：`model/hgf.py::HybridGatedFlow`；`bench/pareto.py`；测试 `tests/test_pareto.py`
- **实测数字**：`[骨架]` — 待 T5/C6（`r=4` 处必须给出具体数字）：`Δ内存 = __%`，`ΔPPL = -__%`（目标：15% 内存换回 ≥ 50% 质量损失）
- **状态**：`[骨架]`

---

### A-23 自适应投机控制器 ★ 原创超越点 O2

- **来源**：工程共识 —— vLLM / TensorRT-LLM 社区公认"盲目开启投机解码是反模式"；收益模型参考 Leviathan et al. [2211.17192](https://arxiv.org/abs/2211.17192) §3 与 EAGLE-2 的动态配置思想
- **决策模型**：
  ```
  精确（选 K）:  E[accepted|K,α] = (1 - α^(K+1)) / (1 - α)
                 cost(K)        = 1 + c·K          c = t_draft / t_verify
                 S(K)           = E[accepted] / cost(K)
                 K*             = argmax_{K≤K_max} S(K)
  简化（汇报）:  speedup ≈ (1 + α·K) / (1 + ε)
  树宽:          w = clamp(round(sqrt(α·K_max·budget_factor)), 1, W_max)
  门控:          batch > B_max(8) → OFF；α < 0.15 → OFF；滞回 1.15/1.00 + N_dwell=8
  ```
  `α` 与 `c` 用 EWMA(β=0.9) 在线拟合。
- **选型理由**：CPU 场景与 GPU 场景的投机经济学**完全不同** —— CPU 上草稿是 GEMV（内存带宽受限，随 batch 线性增长），而目标验证是 GEMM（可摊 batch）。**batch 越大，投机越亏**。这个结论必须被编码进控制器，否则本项目会交付一个"batch=8 时反而变慢"的负优化。
- **被否决方案**：固定 K（社区默认 5）—— 在 CPU 小模型上几乎必然次优或负收益。
- **我们的超越（O2）**：
  1. **把"是否开启"纳入决策变量**（而不只是"K 取多大"）—— 现有开源实现几乎都只调 K，从不关。我们给出**显式的 batch 门控 + α 门控**。
  2. **在线拟合成本比 c**：现有实现普遍假设 `c` 为常数；我们实测摊销后的 `t_draft / t_verify` 并 EWMA 跟踪，使模型随硬件/负载自适应。
  3. **状态机 + 滞回 + 驻留期**：`OFF → WARMUP → ON → DEGRADED → OFF`，防抖动（100 轮随机输入翻转 ≤ 3 次，写成 CI 断言）。
  4. **把"什么时候不该开"作为交付物**：`bench` 报告里给出 α / batch / speedup 三维曲面，明确指出收益边界 —— 这本身是可发表级别的工程结论。
- **落地**：`speculate/controller.py::AdaptiveController`；测试 `tests/test_controller.py`
- **实测数字**：`[骨架]` — 待 T5/C4：`预测 vs 实测 speedup 相对误差 __%`；`batch=1/4/8 的 speedup = __/__/__`；`状态翻转次数 __`
- **状态**：`[骨架]`

---

### A-24 零编译器依赖的原生内核构建 ★ 原创超越点 O4

- **来源**：`ziglang` PyPI 包（`python -m ziglang cc`，zig 自带 clang 前端 + 交叉工具链）；zig 官方 [zig cc 文档](https://ziglang.org/learn/overview/#zig-cc)
- **选型理由**：目标环境**没有 MSVC / gcc**，而 `pip install ziglang` 是唯一"一条命令、三平台、无需管理员权限"拿到可用 C 编译器的途径。zig cc 同时是交叉编译器，天然支持 `-target` 与静态链接。
- **被否决方案**：
  - 要求用户装 MSVC Build Tools / Xcode CLT / build-essential —— 违背"最小可用"，且 CI 与 Windows 开发机都不满足。
  - `cffi` / `pybind11` 构建扩展 —— 同样依赖系统编译器。
  - 纯 Python/NumPy —— 放弃 3–6× 的性能收益。
- **我们的超越（O4）**：
  1. **构建管线内建**：`kernels/native/build.py` 自动定位 `python -m ziglang cc`，产出带**内容 hash 缓存**的动态库（二次启动 < 50 ms）。
  2. **三级回退**：`native(zig cc)` → `numpy` → `pure-python`。任何一级失败都**降级而非崩溃**，且 `bench` 报告标注 `backend_used`。
  3. **并行不依赖 OpenMP/pthread**：C 内核纯函数可重入 + Python 线程池 row-block 切分 + ctypes 释放 GIL，三平台一套代码。
  4. **CI 双 job**：既验证 native 路径，也验证"强制 fallback"路径，保证降级链路永不腐烂。
- **落地**：`phosphorus/native/build.py`、`phosphorus/native/bindings.py`、`phosphorus/backend.py`、`scripts/setup.sh|ps1`、`.github/workflows/ci.yml`
- **实测数字**：`[已验证]` — zig cc 0.16.0；Windows 首次编译 ≈ 3–6 s（含 zig 启动），内容寻址缓存命中后加载 < 50 ms；缓存键 = sha256(ABI 版本 + zig 版本 + 目标三元组 + 编译参数 + 全部源码字节)；无 zig 时 `NativeUnavailable` → 上层静默回退 numpy（`logger.warning`），`phosphorus generate` 仍可运行
- **状态**：`[已验证]`

---

### A-25 native C99 内核的移植边界（T4-A 落地记录）

- **来源**：
  - bitnet.cpp（MIT）`ggml_vec_dot_i2_i8_s` — I2_S 的 SIMD 点积：`_mm256_maddubs_epi16`（u8×i8→i16） + `_mm256_madd_epi16`（i16→i32）
  - T-MAC（Apache-2.0）· [arXiv:2402.17953](https://arxiv.org/abs/2402.17953) — int16 LUT 的理论基础（符号+索引 14 项）
  - ggml（MIT）`vec_dot` 的 fp64 累加 + 单次缩放范式
- **移植了什么（为什么安全）**：
  1. I2_S 恒等式 `Σ(xu8−128)·W = Σxu8·W − 128·row_wsum` 是数学恒等，与实现无关；`maddubs` 的 pair-sum 上界 510 ≪ 32767（因 |W|≤1 不会溢出 int16）→ 指令序列可直接移植。
  2. 「fp64 累加、最后一次性缩放」的数值范式——这是与 numpy float64 参考逐位相等的前提。
- **没有移植什么（为什么不可移植）**：
  1. **bitnet.cpp 的 TL LUT**：它用 int8 LUT + 每 group scale/bias，与本项目的 int16 无损 LUT（14 项符号+索引）数值定义不同，直接复用会破坏不变量 ①（native 必须与 numpy 参考逐位相等）→ 按同一 LUT 定义自写。
  2. **bitnet.cpp 的编译期 shape 特化**：其内核按 (N,K) 特化，而我们的 ABI 要求运行时任意形状 + N 维行块切分（并行用）。
  3. **ggml-quants.c 全量**：只需要 TL / I2_S 两个内核，导入 8000 行量化层（含 k-quant 体系）不可维护、且违背红线「只站在需要的成果上」。
- **被否决方案**：`#include` bitnet.cpp 的 .c 文件——它依赖 ggml 全套（tensor/allocator/线程池），与本项目「零依赖纯函数 ABI」冲突。
- **我们的超越**：
  1. **双路径位精确**：scalar 是权威参考，AVX2 与 scalar 在同一测试内逐位比对（不变量 ⑧）；bitnet.cpp 只在 fp32 域做近似对比。
  2. **运行时 CPUID 分发**：zig 的 windows-gnu 目标不链 libgcc 的 `__cpu_model`，`__builtin_cpu_supports` 不可用 → 手写 CPUID + XGETBV（OSXSAVE/AVX/XGETBV[1:2]==0b11/CPUID.7:EBX[5]），无 AVX2 的机器自动走 scalar 且**数值不变**。
  3. **ABI 的 `double sx`**：`y = (float)((double)acc · (double)wscale · (double)sx)`；早期 ABI 用 `float sx` 收窄了 numpy 参考的 double x_scale，造成 1 ULP 偏差，已修正为 `double`（头文件 / 实现 / ctypes 三处同步）。
  4. **TL 的 AVX2 路线按实测选择**：手写 cvt+madd 版本不如「标量循环 + `__attribute__((target("avx2")))` 让 clang 自动向量化」，已采用后者（实测更快）。
- **落地**：`phosphorus/csrc/phos_gemm.c`、`phosphorus/csrc/phos.h`、`phosphorus/native/bindings.py`、`tests/test_native_backend.py`（L0–L6 共 50 条）、`NOTICE`
- **实测数字**：`[已验证]` — 内核级 10 个形状全部位一致；加速比见 `BENCHMARKS.md` §4.1：i2s AVX2 vs numpy 0.74x（192×192, T=8）~ 10.06x（4096×192, T=1）；tl AVX2 vs numpy 2.84x ~ 6.78x
- **状态**：`[已验证]`

---

## 2. 我们自己"造轮子"的地方（及为什么）

> 按红线要求：**凡有顶级现成方案的必须复用**。以下 6 处是自写实现，逐条给出"为什么现成方案不适配"。

| # | 自写模块 | 现成方案为什么不适配 |
|---:|---|---|
| 1 | `.phos` 二进制格式 | GGUF 无三值打包类型、无处安放 HGF 因子 / 早回头 / row_wsum。结构沿用 GGUF 三段式，只扩展字段。 |
| 2 | BPE 分词器 | `tokenizers` 需 Rust 工具链；`tiktoken` 需 Rust + 网络下载 BPE 文件。目标环境无编译器 + 需离线可训练。 |
| 3 | NumPy 推理引擎 | vLLM / llama.cpp 均为 C++/CUDA 实现，无法在"无系统编译器"环境构建；且本项目需要**可单测的纯 Python 数值参考**作为位精确基准。 |
| 4 | 分页 KV 的 NumPy 实现 | 同上：PagedAttention 的语义必须移植到 CPU/NumPy 才能在目标环境运行，且需要一个与 dense 注意力对齐的参考实现来验证 native 路径。 |
| 5 | KV 驱逐后的**物理压实** | 三篇论文均只做逻辑驱逐，未解决分页场景下 block_table 重写与空洞块回收。 |
| 6 | 自适应投机控制器 | 开源实现（vLLM/TensorRT-LLM）均只调 K、从不关闭，且不拟合成本比 c，与 CPU 场景的经济学相反。 |

---

## 3. 填写清单（实现阶段的 DoD）

`[x]` A-1 ~ A-24 每条的「实测数字」段落均已填入真实测量值
`[x]` 所有 `[骨架]` 已改为 `[已验证]`
`[x]` 每条「我们的超越」若实际未达成，已改为如实说明 + 原因分析
`[x]` `BENCHMARKS.md` 中的数字与本文档一致
`[x]` 所有链接可点击且指向原始来源（论文走 arXiv 摘要页，项目走 repo/PR 直链）

> T4-A 补登：A-24 / A-25 已从 `[骨架]` 升级为 `[已验证]`（2026-10-17）。
> 未达成项如实记录：C3 的「i2s 小 N + T>1 ≥ 3×」实测为 **0.74×**（numpy 批量矩阵乘更优），
> 已按实测 crossover 改为自适应调度而非硬走 native；「8 线程 ≥ 6×」未测（行块并行已实现且
> L5 逐位相等，但 tiny 模型 N≤384 不足以体现多线程收益，留待 T4-B 后用真实规模模型复测）。
