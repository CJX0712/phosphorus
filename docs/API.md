# Phosphorus 公开 API 参考

> **版本**：0.1.0 · **作者**：晨星（CJX0712） · **许可**：Apache-2.0
> **约定**：本文所有签名与 `phosphorus/` 源码逐字核对（抽查 `generate_tokens` /
> `spec_verify` / `sampling_probs` 等 6 个入口，`inspect.signature` 实测一致）。
> 「契约陷阱」章节列出**看签名看不出来、踩过坑**的硬约束——改代码前必读。

## 0. 30 秒上手

```python
from phosphorus.engine import PhosModel

model = PhosModel.from_file("ckpt/tiny.phos", backend="tl")  # 反量化 fp32 / int8 / tl / i2s
print(model.generate("To be, or", max_new_tokens=64, speculate="auto", K=8))
```

```python
# 连续批处理（8 条 prompt 一次前向）
from phosphorus.scheduler import Scheduler

texts = Scheduler(model, max_seqs=8).generate(["Hello"] * 8, max_new_tokens=32)
```

```python
# 手动投机解码（拿接受率 α 与统计信息）
from phosphorus import speculate as sp

ids, alpha, info = sp.generate_tokens(
    model,
    model.tok.encode("To be, or"),
    max_new_tokens=64,
    drafter=sp.LayerSkipDrafter(model),
    K=8,
    draft_model=model.shadow(),  # 草稿器独占热 KV cache（共享权重，零额外字节）
)
```

---

## 1. 包级（`phosphorus/__init__.py`）

运行时**零 torch 依赖**：`import phosphorus` 只依赖 numpy + tqdm。torch 仅
`phosphorus.model` / `phosphorus.train` 惰性导入。

| 符号 | 签名 / 值 | 说明 |
|---|---|---|
| `__version__` | `"0.1.0"` | 语义版本 |
| `__author__` | `"晨星（CJX0712）"` | |
| `__license__` | `"Apache-2.0"` | |
| `PHOS_VERSION_MAJOR` / `_MINOR` | `1` / `0` | `.phos` 格式版本（ARCHITECTURE.md §3） |
| `has_torch()` | `() -> bool` | 探测 torch 是否可用（不导入） |
| `has_ziglang()` | `() -> bool` | 探测 ziglang（O4 零系统编译器构建链） |
| `default_backend()` | `() -> str` | `PHOS_BACKEND` 环境变量优先，否则 `"auto"` |
| `default_threads()` | `() -> int` | `PHOS_THREADS` 环境变量优先，否则 CPU 逻辑核数 |
| `env_summary()` | `() -> str` | 一行环境摘要（报告/issue 复现用） |
| `Scheduler` | 类 | 从包根直接导出（`from phosphorus import Scheduler`） |

---

## 2. 推理引擎（`phosphorus/engine.py`）

### 2.1 `PhosModel`

```python
PhosModel(phos: PhosFile, backend: str = "fp32", native: str | None = None,
          kv_mode: str = "paged", block_size: int = DEFAULT_BLOCK_SIZE,
          eviction: str = "none", kv_budget: float = 0.5, window_size: int = 32,
          sink_size: int = 4, max_seqs: int = 1) -> PhosModel
PhosModel.from_file(path, backend="fp32", native=None, kv_mode="paged",
                    block_size=DEFAULT_BLOCK_SIZE, eviction="none",
                    kv_budget=0.5, window_size=32, sink_size=4, max_seqs=1) -> PhosModel
```

构造参数取值：

| 参数 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `backend` | `"fp32"` / `"int8"` / `"tl"` / `"i2s"` | `"fp32"` | fp32 = 反量化后 fp32 GEMM（torch QAT 对照面，不变量 ⑦）；tl/i2s = 部署态整数路径，二者 int32 域逐位相等 |
| `native` | `None` / `"native"` / `"numpy"` | `None` | 内核实现选择；`None` = 按 backend 解析（`backend.resolve`） |
| `kv_mode` | `"dense"` / `"paged"` | `"paged"` | paged = 分页 KV 池 + block_tables（T4-B）；dense = 全量张量参考实现；数值等价（不变量 ⑨） |
| `eviction` | `"none"` / `"snapkv"` / `"pyramidkv"` / `"h2o"` / `"streamingllm"` | `"none"` | 仅 paged 生效；逐层预算 + 保留掩码压长序列 KV |
| `kv_budget` | `(0, 1]` | `0.5` | 保留比例 |
| `window_size` / `sink_size` | int | `32` / `4` | 强制保留的最近窗口 / attention sink |
| `max_seqs` | int ≥ 1 | `1` | 连续批处理并发上限；分页物理池按 `max_seqs × ⌈max_seq_len/block_size⌉ + 1` 分配 |

主要属性：`backend`、`kv_mode`、`d`、`n_layer`、`n_head`、`n_kv_head`、`d_head`、
`d_ff`、`vocab`、`max_seq_len`、`rope_theta`、`norm_eps`、`act_relu2`、
`hgf_alpha`、`early_exit_layer`、`n_rep`、`tok`、`bos_id`、`eos_id`。

#### 方法

| 方法 | 签名 | 说明 |
|---|---|---|
| `forward` | `(tokens: np.ndarray, return_hidden: bool = False, stop_at_layer: int \| None = None)` | 对 `[T]` token 前向，更新 KV，返回 `[T, V]` logits。`stop_at_layer` 仅前向到该层（O1 早退草稿），**不触发驱逐**。`T > max_seq_len` 时自动滑窗（RoPE 表守卫） |
| `forward_step` | `(seq_tokens: dict[int, np.ndarray]) -> dict[int, np.ndarray]` | 连续批处理一步解码：多序列新 token 融合一次前向，返回各序列末位 logits。⚠️ **不支持驱逐路径**（各层长度必须一致）；`seq_id` 必须 `< max_seqs` |
| `cache_len` | `(seq_id: int = 0) -> int` | 某序列 KV 中的 token 数（dense/paged 统一接口） |
| `generate` | `(prompt: str, max_new_tokens=128, temperature=0.8, top_k=0, top_p=0.95, seed=0, stream=False, speculate="off", draft_layers=None, K=4, controller=None) -> str` | 文本生成。`speculate ∈ {"off","none","on","auto"}`；`"on"/"auto"` 且 `n_layer ≥ 2` 时走 O1 自草稿。⚠️ 要求 `.phos` 内含 tokenizer |
| `shadow` | `() -> PhosModel` | 克隆**共享权重、独立 KV cache** 的镜像（草稿器专用）。驱逐强制 `"none"`（投机要求逐层长度一致） |
| `truncate_tail` | `(seq_id: int = 0, n: int = 1) -> int` | 逐序列丢弃 KV 末尾 n 个 token（拒绝回滚），返回实际丢弃数。⚠️ 要求 `eviction == "none"` |
| `reset` | `() -> None` | 清空 KV cache |

---

## 3. 投机解码（`phosphorus/speculate/`）

对外主入口 `generate_tokens`；草稿器协议 `Drafter`；验证 `spec_verify`；
控制器 `SpecController`。包 `__all__` 导出：`Drafter, DrafterSpec,
SpecController, SpecPlan, LayerSkipDrafter, NGramDrafter, TreeSpec, TreeNode,
build_balanced_tree, prune_by_prob, VerifyResult, spec_verify, generate_tokens`。

### 3.1 `generate_tokens`（热 KV 增量主循环）

```python
generate_tokens(model, ctx_ids: list[int], max_new_tokens: int = 128,
                temperature: float = 0.8, top_k: int = 0, top_p: float = 0.95,
                seed: int = 0, drafter: Drafter | None = None, K: int = 4,
                controller: SpecController | None = None, stream: bool = False,
                tok=None, draft_model=None) -> tuple[list[int], float, dict]
```

返回 `(生成 token 列表, 平均接受率 α, 统计 info)`，`info = {"alpha", "steps",
"n_new", "drafts_accepted", "drafts_proposed"}`。

| 参数 | 说明 |
|---|---|
| `drafter` | 草稿器；`None` 时强制逐 token 串行（cache 仍热） |
| `K` | 每轮草稿长度；`controller` 给出 `enabled=False` 或 `K < 2` 时退化串行 |
| `controller` | O2 自适应控制器；其 `plan()` 每轮决策 |
| `draft_model` | 草稿器的热 KV 载体。**不传则内部 `model.shadow()`**；显式传入可复用同一镜像 |

行为契约：目标/草稿各持一条独立热 KV cache；每轮草稿 K 个单 token 热追加
（每个草稿 token 含最后一个**都必须前向写入草稿 cache**——否则双 cache 每轮
脱同步 1 token，α 崩塌，实测 round3 α→0），目标一次前向覆盖 K 个新位置，
残差拒绝采样保证输出分布与目标**严格相等**（不变量 ②），`truncate_tail`
O(T) 回滚被拒草稿。

### 3.2 `Drafter` 协议（`base.py`）

```python
class Drafter(ABC):
    spec: DrafterSpec                      # name / extra_bytes / draft_layers
    def draft_logits(self, seq: list[int]) -> np.ndarray   # 末位 logits（q 的来源）
    @property
    def extra_bytes(self) -> int            # 相对目标模型新增权重字节数（O1 必须 0）
    def reset(self) -> None                 # 默认无状态；有内部缓存者可重写
```

`DrafterSpec(name: str, extra_bytes: int, draft_layers: int)`。

| 实现 | 构造签名 | 状态 |
|---|---|---|
| `LayerSkipDrafter` | `(model, draft_layers: int \| None = None, name="layerskip")` | ✅ O1：用目标前 `draft_layers` 层 logits 作 q；默认 `max(1, n_layer-1)`；要求 `1 ≤ draft_layers < n_layer`；`extra_bytes == 0` |
| `NGramDrafter` | `(model=None, order: int = 3, name="ngram")` | ✅ order-3 n-gram 对照；命中路径零模型成本，未命中回退目标模型 |
| `EagleDrafter` | `(model, draft_layers=None)` | ⚠️ 当前 = `LayerSkipDrafter` 别名（`name="eagle"`），P1 扩展点 |
| `MedusaDrafter` | `(model, draft_layers=None, heads=1)` | ⚠️ 当前 = 浅层自草稿占位，`heads` 暂未使用，P1 扩展点 |

⚠️ `draft_logits` 约定「无状态推测」：调用方负责在草稿/验证前管理 KV。
`generate_tokens` 的做法是草稿器持 `draft_model`（shadow）并**热追加**，
不走 `draft_logits` 的全序列重算路径；自实现草稿循环时务必保持双 cache 同步。

### 3.3 `spec_verify`（Leviathan 残差拒绝采样，`verify.py`）

```python
spec_verify(draft_tokens: list[int], q_dists: list[np.ndarray],
            p_dists: list[np.ndarray], rng: np.random.Generator,
            greedy: bool = False) -> VerifyResult
```

| 参数 | 契约 |
|---|---|
| `draft_tokens` | 草稿 token `y_0..y_{K-1}` |
| `q_dists` | 长度 K，**概率向量（Σ=1）**——`sampling_probs(q_logits, ...)` 的输出 |
| `p_dists` | 长度 K+1，**概率向量（Σ=1）**；`p_dists[K]` = bonus 位置目标分布 |
| `greedy` | `True` 走确定性贪心验证（temperature=0 专用）：接受 iff `y_i == argmax(p_i)`；拒绝补 `argmax(p_i)`；bonus = `argmax(p_K)` |

`VerifyResult(accepted: list[int], n_proposed: int)`；属性 `n_new`、
`n_accepted_drafts`（= 总长 − 1）、`alpha_step`。

⚠️ **本函数不再内部 softmax**。q/p 必须是采样器真正采样所依据的分布
（temperature + top-k + top-p 截断后）——传原始 softmax 会在 `top_p < 1`
时破坏分布等价（不变量 ②），这是本项目修过的最严重缺陷之一。

### 3.4 `SpecController`（O2 自适应控制器，`controller.py`）

```python
SpecController(K_max: int = 8, W_max: int = 4, B_max: int = 8,
               alpha_min: float = 0.15, eps_on: float = 0.15, eps_off: float = 0.05,
               N_warm: int = 4, N_dwell: int = 8, budget_factor: float = 1.0,
               smooth: float = 0.8)
observe(alpha: float, c: float | None, batch_size: int, kv_util: float | None = None) -> None
plan() -> SpecPlan
predicted_speedup() -> float
```

- `SpecPlan(enabled: bool, K: int, state: str, alpha_hat: float, speedup: float)`
- 决策公式：`S(K) = ((1-α^(K+1))/(1-α)) / (1+c·K)`；汇报式 `(1+αK)/(1+cK)`
- 状态机 `OFF→WARMUP→ON→DEGRADED`；硬门控：`batch_size > B_max` 或
  `α < alpha_min` → OFF（CPU 上 batch 大时投机必亏）
- `smooth` 是 EWMA 系数（默认 0.8，实测 0.9 收敛过慢：α 骤降需 12 步以上才 OFF）

### 3.5 树草稿（`tree.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `TreeNode` | dataclass `(token, position_id, parent=-1, joint_prob=1.0)` | |
| `TreeSpec` | dataclass `(nodes: list[TreeNode])`；`__len__` / `attention_mask()` / `position_ids()` | 掩码 `(n,n)` 布尔，`mask[i][j]=True` ⟺ i 是 j 的祖先或自身 |
| `build_balanced_tree` | `(root_token: int, depth: int, width: int, base_position: int = 0) -> TreeSpec` | 演示树（单测用）；真实场景由草稿候选 + 剪枝生成 |
| `prune_by_prob` | `(tree: TreeSpec, tau: float = 0.02) -> TreeSpec` | 丢弃 `joint_prob < tau` 节点（保留其祖先连通） |

---

## 4. 采样（`phosphorus/sampling.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `softmax` | `(logits, axis=-1) -> np.ndarray` | 数值稳定 softmax |
| `sampling_probs` | `(logits, temperature=1.0, top_k=0, top_p=1.0) -> np.ndarray` | **采样器实际使用的分布**（截断后，Σ=1）。投机验证构造 p/q 必须用它 |
| `sample` | `(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None) -> int` | 与 `sampling_probs` 同分布取一个 token；`rng=None` 时用 `default_rng(0)` |

`temperature ≤ 0` = 贪心（argmax）。`top_k=0` = 不过滤；`top_p=1.0` = 不截断。

---

## 5. 连续批处理（`phosphorus/scheduler.py`）

```python
Scheduler(model, max_seqs: int | None = None)
generate_ids(prompt_ids: list[list[int]], max_new_tokens=64, temperature=0.0,
             top_k=0, top_p=0.95, seed=0) -> list[list[int]]      # 新生成 id
generate(prompts: list[str], max_new_tokens=64, temperature=0.0,
         top_k=0, top_p=0.95, seed=0) -> list[str]                # 文本
```

- `max_seqs` 缺省取 `model.max_seqs`；prompt 按 `max_seqs` 切波次
- 每序列独立 rng：`seed + 全局 prompt 下标` → 与逐条串行同 seed **逐位一致**
- 返回顺序与输入严格对齐；已完成序列不再进 batch，不影响他人
- ⚠️ `dense` 模式下 `forward_step` 退化为逐序列串行（无融合收益但仍正确）

---

## 6. KV 管理（`kv_cache.py` / `block_manager.py`）

### 6.1 `PagedKVCache`（物理池）

```python
PagedKVCache(
    n_blocks,
    n_layer,
    n_head,
    d_head,
    block_size=DEFAULT_BLOCK_SIZE,
    max_blocks_per_seq=None,
    eviction=None,
)
```

池形 `[n_blocks, n_layer, 2, block_size, n_kv_head, d_head]`。方法：
`reset()`、`length(seq_id, layer)`、`can_append(seq_id, n)`、
`begin_append(seq_id, n)` / `end_append(seq_id)`、`write(seq_id, layer, k, v)`、
`append(seq_id, layer, k, v)`、`gather(seq_id, layer)`、
`gather_heads(seq_id, layer, n_rep)`、`drop_front(seq_id, n)`、
`truncate_tail(seq_id, n)`、`compact(seq_id, keep_masks)`、`nbytes()`、
`block_size` 属性。

⚠️ 写协议是 `begin_append → 逐层 write → end_append` 三阶段；`truncate_tail`
按**尾部整块**回收并同步所有层 `layer_len`——要求各层长度一致（驱逐模式
逐层预算不同，禁止 truncate）。

### 6.2 `BlockManager`（vLLM 语义）

`__init__(pool, ...)` 等；主要方法：`num_free_blocks()`、`num_used_blocks()`、
`utilization()`、`seq_len(seq_id)`、`n_blocks_of(seq_id)`、
`block_table(seq_id)`、`can_append(seq_id, n)`、`append_tokens(seq_id, n)`、
`release_block(block_id)`、`free_seq(seq_id)`、`drop_tokens(seq_id, n)`、
`truncate_tail(seq_id, n)`、`touch(seq_id, n_blocks)`、`reset()`。

⚠️ 滑动窗口瞬态：物理跨度 = `head_offset + seq_len` 可达
`max_seq_len + block_size − 1`，故引擎块池按
`max_seqs × (⌈max_seq_len/block_size⌉ + 1) + 1` 分配（否则窗口回绕时
「分页 KV 容量不足」）。

---

## 7. KV 驱逐（`phosphorus/eviction.py`）

```python
EvictionConfig(budget_ratio=0.5, window_size=32, sink_size=4, kernel_size=5,
               pooling="avgpool", head_agg="mean", beta=20, m_min=8)
make_policy(name: str, cfg: EvictionConfig | None = None) -> EvictionPolicy
```

策略：`EvictionPolicy`（基类，`select(scores, budget, n_total) -> 掩码`）、
`SnapKVPolicy`、`H2OPolicy(cfg=None)`、`StreamingLLMPolicy`、
`PyramidKVPolicy`（`layer_budgets(n_layer, seq_len)`）、
`BudgetAllocator(cfg=None).layer_budgets(n_layer, seq_len, mode="pyramid")`。

选型与「统一保留 sink + 窗口」的组合改进说明见 `docs/ALGORITHMS.md` A-8~A-11。

---

## 8. 量化内核（`gemm.py` / `packing.py` / `backend.py`）

### 8.1 GEMM（不变量 ① 的基准面）

| 函数 | 签名 | 说明 |
|---|---|---|
| `quantize_activation` | `(x, bits=8) -> (xq int8, x_scale)` | BitNet per-token 8-bit 激活量化 |
| `int32_reference` | `(W_ternary, xq) -> np.ndarray` | 权威整数参考 `acc[i]=Σ_j W[i,j]·xq[j]`（int32） |
| `gemm_fp32` | `(W_fp, x) -> np.ndarray` | fp32 参考 GEMM |
| `gemm_int8` | `(W_fp, x) -> np.ndarray` | int8 激活路径 |
| `gemm_ternary_tl` | `(packed, xq, w_scale, x_scale, N, K) -> np.ndarray` | TL 后端（打包域，5-bit 直接查 LUT） |
| `gemm_ternary_i2s` | `(W_ternary, xq, w_scale, x_scale)` | I2_S 后端（符号 + int8 镜像） |
| `tl_acc_i32` / `i2s_acc_i32` | `(...) -> np.ndarray` | int32 累加核（位精确断言用） |
| `tl_i2s_bitwise_equal` / `..._packed` | `(...) -> bool` | 两条整数路径逐位相等断言 |
| `build_lut_tables` | `(xq, ng) -> np.ndarray` | TL 16 项 LUT（14 项符号+索引补齐） |

### 8.2 三值 5-bit 打包（`packing.py`，不变量 ③）

`ternary_to_digits(w)`、`digits_to_ternary(d)`、`k_pad(K)`、`packed_bytes(N, K)`、
`row_wsum(w)`、`to_i8_mirror(w)`、`pack_ternary_v1(w) -> (packed, Kpad)`、
`unpack_code5(packed, N, K)`、`unpack_ternary_v1(packed, N, K)`、
`build_lut16(a)`、`lut_from_code5(lut16)`。24 权重 → 5 字节（1.67 bit/权重）。

### 8.3 后端解析（`backend.py`）

`resolve(requested: str | None = None) -> str`（返回 `"native"` / `"numpy"`）、
`native_status() -> (bool, str)`、`available_backends() -> tuple[str, ...]`。
三级回退 native → numpy → pure-python（`PHOS_BACKEND` 环境变量可强制）。

---

## 9. 加载 / 导出（`loader.py` / `export.py`）

```python
load_phos(path_or_bytes: str | Path | bytes) -> PhosFile
export_model(weights: dict[str, np.ndarray], mcfg: ModelConfig, qcfg: QuantConfig,
             tokenizer=None, path: str | Path | None = None,
             metadata: dict | None = None) -> bytes
inspect_phos(path: str | Path) -> str        # 人类可读的 .phos 摘要
```

`PhosFile` 属性：`weights`（三值 int8 / 其余 f32）、`scales`（per-channel）、
`hgf`（`name -> (A f16 [r,in], B f16 [out,r], gate)`）、`tokenizer`、`header`、
`metadata`、`raw`（**原始 5-bit 打包字节**，TL 后端直接查 LUT，避免非部署路径
掩盖真实成本）；方法 `tensor_names() -> list[str]`。

---

## 10. 配置（`config.py`）

`PhosphorusConfig(model, quant, train, decode)` → `to_dict()` / `to_json()` /
`save(path)` / `from_dict(data)` / `load(path)`；
`merge_overrides(cfg, {"model.field": ...})`。

| 配置 | 关键字段（默认） |
|---|---|
| `ModelConfig` | `d_model=384, n_layer=6, n_head=6, n_kv_head=2, d_ff=768, vocab_size=8192, max_seq_len=512, rope_theta=10000.0, act="relu2", tied_embed=True, early_exit_layer=-1(→ceil(n/2)), early_head_rank=0`；`from_scale("tiny"/"small"/"base", **kw)`；属性 `d_head` / `d_kv` / `n_params()` |
| `QuantConfig` | `scheme="ternary", w_granularity="per_channel", a_granularity="per_token", block_size=128, a_bits=8, hgf_rank=4, hgf_alpha=1.0, hgf_gate_init=0.05, hgf_dtype="f16", qat=True, quant_embed=False, kv_int8=False` |
| `TrainConfig` | `steps=300, batch_size=16, block_size=256, lr=3e-3, warmup_steps=30, mode="qat-hgf", layer_dropout_min=1.0, early_exit_weight=0.5, out="ckpt/tiny.pt"` |
| `DecodeConfig` | `max_new_tokens=128, temperature=0.8, top_k=0, top_p=0.95, seed=0, backend="auto", speculate="off", kv_block_size=16, eviction="none", kv_budget_ratio=1.0, stream=False` |

枚举：`ActType`、`QuantEnum`、`Granularity`、`TrainMode`、`BackendName`、
`SpecMode`、`EvictionStrategy`（均为 `str, Enum`，接受字符串构造）。

---

## 11. 分词器（`tokenizer.py`）

```python
Tokenizer(vocab: list[str], merges: list[tuple[str, str]], specials: dict[str, int] | None = None)
Tokenizer.train(texts_or_path, vocab_size=..., ...)      # classmethod，BPE（Sennrich 2016）
encode(text: str, add_special: bool = False) -> list[int]
decode(ids: list[int]) -> str
save(path) / load(path)                                  # 二进制 .tok 格式
save_to_bytes() / load_from_bytes(b)
vocab_size -> int
train_from_corpus(corpus_path, vocab_size=..., out=..., ...)   # 模块级便捷入口
```

字节级映射 + 256 字节种子 → CJK/emoji 无损往返（不变量 ⑤）。

---

## 12. 训练（`model.py` / `train.py` / `data.py`，需 torch）

| 符号 | 签名 | 说明 |
|---|---|---|
| `absmean` | `(w: Tensor) -> Tensor` | absmean 缩放 |
| `ternary_ste` | `(w, scale) -> Tensor` | 三值 STE 量化（不变量 ⑥） |
| `BitLinear` | `(in_f, out_f, qcfg)` | 三值线性层（QAT） |
| `RMSNorm` | `(dim, eps=1e-5)` | |
| `build_rope` / `apply_rope` | `(theta, d_head, max_len, device)` / `(q, cos, sin)` | |
| `Layer` / `LlamaModel` | `(mcfg, qcfg)` | `forward(idx, cos=None, sin=None, return_logits=True)`；`export_numpy() -> dict[str, np.ndarray]` |
| `run_train` | `(cfg: PhosphorusConfig, corpus_path, tokenizer=None, verbose=True) -> (model, tokenizer, stats)` | AdamW + cosine + grad clip + ckpt |
| `export_trained` | `(model, tokenizer, path, ...)` | ckpt → `.phos` |
| `TextDataset` | `(tokenizer, tokens, block_size)`；`from_corpus(corpus_path, tokenizer, block_size=256, cache_dir=None)`；`batch(bs, rng)` / `val_batch(n)` / `__len__` | int16 缓存，缓存文件名由文本路径 + vocab_size 派生 |

---

## 13. CLI（`cli.py` / `__main__.py`）

`python -m phosphorus <subcommand>`：

| 子命令 | 关键参数 |
|---|---|
| `train` | `--steps/--batch-size/--lr/--mode/--out` |
| `export` | `ckpt → .phos` |
| `generate` | `--backend/--kv/--speculate {off,on,auto}/--draft-layers/--K/--temperature/--top-p/--stream/--seed` |
| `sched` | `--max-seqs/--prompts`（连续批处理） |
| `bench` | `--backend/--batch/--seq-len`（单组基准；全矩阵走 `scripts/bench_matrix.py`） |
| `tokenize` | `--train/--vocab-size/--out` |

`build_parser() -> argparse.ArgumentParser`、`main(argv=None) -> int`。

---

## 14. 契约陷阱（改代码前必读）

1. **`spec_verify` 的 q/p 必须是截断后分布**。传原始 softmax 在 `top_p < 1` 时
   破坏不变量 ②（退化分布下碰巧一致会掩盖缺陷）。构造一律走
   `sampling_probs(logits, temperature, top_k, top_p)`。
2. **草稿循环必须前向每一个草稿 token（含最后一个）**。跳过末 token 前向会使
   双 cache 每轮脱同步 1 token，α 逐轮衰减至 0（实测 round3 α→0.000）。
   末次输出的 `q_K` 不参与验证，但 cache 同步是硬前提。
3. **`forward_step` 不支持驱逐**。驱逐改变各层 KV 长度，而 `forward_step` 按
   `seq_id` 融合前向假设各层一致。驱逐只用于单序列 `forward` 路径。
4. **`truncate_tail` 要求 `eviction == "none"`**。逐层预算模式下各层长度不同，
   尾部截断会整体收缩逐层预算。`shadow()` 已强制 `"none"`。
5. **滑动窗口物理跨度**。`head_offset + seq_len` 可达
   `max_seq_len + block_size − 1`；块池无瞬态预留会在窗口回绕时报
   「分页 KV 容量不足」。池与表各 +1 块。
6. **`forward` 的滑窗用逻辑坐标**。超过 `max_seq_len` 时按 `cache_len()`
   （逻辑）而非绝对坐标截断，否则驱逐/截断后 RoPE 位置错位。
7. **`max_seq_len` 守卫**。`T > max_seq_len` 时 `forward` 只保留最近
   `max_seq_len` 个 token（RoPE 表边界），这是静默行为，长 prompt 注意。
8. **`.phos` 无 tokenizer 时文本接口不可用**。`generate` / `Scheduler.generate`
   抛 `RuntimeError`，请用 token id 接口（`generate_tokens` / `generate_ids`）。
9. **单/双层模型无法 early-exit 自草稿**。`LayerSkipDrafter` 要求
   `1 ≤ draft_layers < n_layer`，否则 `ValueError`；`generate` 对
   `n_layer < 2` 自动回退串行。
10. **`NGramDrafter` 的 `model=None` 可用**（纯 n-gram 命中零模型成本），
    未命中时才需要回退模型。
11. **EAGLE / Medusa 当前是占位别名**（见 §3.2 表），勿当独立算法引用。

## 15. 相关文档

- `docs/ARCHITECTURE.md` —— 架构 + `.phos` 格式 + 不变量清单
- `docs/ALGORITHMS.md` —— 每个模块的来源论文/项目 + 选型理由 + 超越点
- `docs/TASKS.md` —— 任务分解与验收检查点
- `docs/BENCHMARKS.md` —— 实测基准（含投机矩阵 §5.6）
