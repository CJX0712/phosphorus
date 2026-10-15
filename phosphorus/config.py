"""全部配置 dataclass（L0 层，只允许依赖 stdlib）。

设计约定：
    * 所有配置都是 `@dataclass`，支持 `to_dict()` / `from_dict()` / JSON 落盘。
    * `--config xxx.json` 可覆盖任意字段（CLI 命令行参数优先级最高）。
    * 本模块**不得**导入 numpy / torch，保证 core 分组零重依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class ActType(str, Enum):
    """FFN 激活函数。`relu2` 是 BitNet b1.58 默认；`swiglu` 作为消融对照。"""

    RELU2 = "relu2"
    SWIGLU = "swiglu"


class QuantEnum(str, Enum):
    """权重量化方案枚举（对应 `.phos` header 的 `quant_enum`）。"""

    FP32 = "fp32"
    INT8 = "int8"
    TERNARY = "ternary"


class Granularity(str, Enum):
    """量化粒度。默认权重 per_channel（按输出行）、激活 per_token（BitNet 惯例）。"""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"
    PER_TOKEN = "per_token"


class TrainMode(str, Enum):
    """训练模式。`fp32` 为基线，`qat-hgf` 为 BitNet 三值 QAT + HGF 低秩校正。"""

    FP32 = "fp32"
    QAT_HGF = "qat-hgf"


class BackendName(str, Enum):
    """GEMM 后端。auto 走三级回退 native -> numpy。"""

    AUTO = "auto"
    NATIVE = "native"
    NUMPY = "numpy"
    FP32 = "fp32"
    INT8 = "int8"
    TERNARY = "ternary"


class SpecMode(str, Enum):
    """投机解码开关（ARCHITECTURE.md §6.4）。"""

    OFF = "off"
    ON = "on"
    AUTO = "auto"


class EvictionStrategy(str, Enum):
    """KV 驱逐策略。"""

    NONE = "none"
    SNAPKV = "snapkv"
    PYRAMID = "pyramid"
    H2O = "h2o"


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """LLaMA 变体结构配置。

    默认 tiny 档（ARCHITECTURE.md §11 假设 1，d_ff 由 1024 修正为 768）：
    `d_model=384, n_layer=6, n_head=6, n_kv_head=2, d_ff=768, vocab=8192`
    -> **10,813,440 参数**（tied embedding，lm_head 不重复计数）。
    """

    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    d_ff: int = 768
    vocab_size: int = 8192
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    act: ActType = ActType.RELU2
    tied_embed: bool = True
    # O1：早退层号（-1 表示无早退）。默认 ceil(n_layer/2)。
    early_exit_layer: int = -1
    # O1：早退 head 的低秩秩；0 = 复用 tied embedding（零额外参数）。
    early_head_rank: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.act, str):
            self.act = ActType(self.act)
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model({self.d_model}) 必须能被 n_head({self.n_head}) 整除")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head({self.n_head}) 必须能被 n_kv_head({self.n_kv_head}) 整除")
        if self.early_exit_layer < 0:
            self.early_exit_layer = -((-self.n_layer) // 2)  # ceil(n_layer/2)

    @property
    def d_head(self) -> int:
        """每个注意力头的维度。"""
        return self.d_model // self.n_head

    @property
    def d_kv(self) -> int:
        """KV 投影总维度（GQA：n_kv_head * d_head）。"""
        return self.n_kv_head * self.d_head

    def n_params(self) -> int:
        """返回可训练参数总量（tied embedding 时 lm_head 不重复计数）。"""
        d, nl, v = self.d_model, self.n_layer, self.vocab_size
        dkv = self.d_kv
        per_layer = (
            d * d  # q_proj
            + 2 * dkv * d  # k_proj / v_proj
            + d * d  # o_proj
            + 3 * d * self.d_ff  # gate / up / down（ReLU² 或 SwiGLU）
            + 4 * d  # attn_norm, subln_out(q/k 后 2 个), ffn_norm, mlp subln_out
        )
        total = nl * per_layer + v * d + d  # + final norm
        if not self.tied_embed:
            total += v * d
        if self.early_head_rank > 0:
            # early_head.lora_a [d, r] + lora_b [r, V]
            total += self.early_head_rank * (d + v)
        return total

    @classmethod
    def from_scale(cls, scale: str = "tiny", **kw: Any) -> ModelConfig:
        """按预设档位构造。tiny ≈ 10.8M / small ≈ 28M / base ≈ 96M。"""
        presets: dict[str, dict[str, int]] = {
            "tiny": {"d_model": 384, "n_layer": 6, "n_head": 6, "n_kv_head": 2, "d_ff": 768},
            "small": {"d_model": 512, "n_layer": 8, "n_head": 8, "n_kv_head": 2, "d_ff": 1280},
            "base": {"d_model": 768, "n_layer": 12, "n_head": 12, "n_kv_head": 4, "d_ff": 2048},
        }
        if scale not in presets:
            raise ValueError(f"未知档位 {scale!r}，可选 {sorted(presets)}")
        return cls(**{**presets[scale], **kw})


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """BitNet 量化配置（对应 `.phos` header 的 quant 字段）。"""

    scheme: QuantEnum = QuantEnum.TERNARY
    w_granularity: Granularity = Granularity.PER_CHANNEL
    a_granularity: Granularity = Granularity.PER_TOKEN
    block_size: int = 128
    a_bits: int = 8
    # O3：HGF 低秩校正秩 r（0 = 关闭）。默认 4 ≈ 15% 额外内存。
    hgf_rank: int = 4
    hgf_alpha: float = 1.0
    hgf_gate_init: float = 0.05
    hgf_dtype: str = "f16"
    qat: bool = True
    # 嵌入表与 LM head 保持 fp16（BitNet 论文惯例）
    quant_embed: bool = False
    kv_int8: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.scheme, str):
            self.scheme = QuantEnum(self.scheme)
        if isinstance(self.w_granularity, str):
            self.w_granularity = Granularity(self.w_granularity)
        if isinstance(self.a_granularity, str):
            self.a_granularity = Granularity(self.a_granularity)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """训练超参。默认组合在 16 线程 CPU 上约 10 分钟跑完。"""

    steps: int = 300
    batch_size: int = 16
    block_size: int = 256
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 30
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1337
    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    ckpt_interval: int = 0  # 0 = 只在结束时保存
    mode: TrainMode = TrainMode.QAT_HGF  # 默认走可部署的 1.58-bit 路径
    num_threads: int = 0  # 0 = 用 os.cpu_count()
    # LayerSkip（O1 前置）：层 dropout 的最深层 survival 概率
    layer_dropout_min: float = 1.0  # 1.0 = 关闭层 dropout
    early_exit_weight: float = 0.5  # 早退 head 损失权重
    grad_accum: int = 1
    out: str = "ckpt/tiny.pt"

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = TrainMode(self.mode)


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------


@dataclass
class DecodeConfig:
    """推理解码配置。"""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 0  # 0 = 不过滤
    top_p: float = 0.95
    seed: int = 0
    backend: BackendName = BackendName.AUTO
    speculate: SpecMode = SpecMode.OFF
    kv_block_size: int = 16
    eviction: EvictionStrategy = EvictionStrategy.NONE
    kv_budget_ratio: float = 1.0
    stream: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = BackendName(self.backend)
        if isinstance(self.speculate, str):
            self.speculate = SpecMode(self.speculate)
        if isinstance(self.eviction, str):
            self.eviction = EvictionStrategy(self.eviction)


# ---------------------------------------------------------------------------
# 聚合与序列化
# ---------------------------------------------------------------------------


@dataclass
class PhosphorusConfig:
    """一次完整运行所需的全部配置。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典（枚举与元组已降级为 JSON 原生类型）。"""
        out: dict[str, Any] = {}
        for sec in ("model", "quant", "train", "decode"):
            obj = getattr(self, sec)
            out[sec] = {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
        return out

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_json_default)

    def save(self, path: str | Path) -> Path:
        """落盘为 JSON。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhosphorusConfig:
        """从字典构造（未知字段忽略，便于向前兼容）。"""
        cfg = cls()
        for sec in ("model", "quant", "train", "decode"):
            if sec not in data:
                continue
            obj = getattr(cfg, sec)
            valid = {f.name for f in fields(obj)}
            for k, v in data[sec].items():
                if k in valid:
                    setattr(obj, k, v)
        # __post_init__ 不会自动重跑，这里显式规范化枚举
        cfg.model = ModelConfig(**{f.name: getattr(cfg.model, f.name) for f in fields(cfg.model)})
        cfg.quant = QuantConfig(**{f.name: getattr(cfg.quant, f.name) for f in fields(cfg.quant)})
        cfg.train = TrainConfig(**{f.name: getattr(cfg.train, f.name) for f in fields(cfg.train)})
        cfg.decode = DecodeConfig(
            **{f.name: getattr(cfg.decode, f.name) for f in fields(cfg.decode)}
        )
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> PhosphorusConfig:
        """从 JSON 文件加载。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _plain(v: Any) -> Any:
    """把枚举 / 元组降级为 JSON 原生类型，其余原样返回。"""
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, tuple):
        return list(v)
    return v


def _json_default(o: Any) -> Any:
    """JSON 兜底序列化器。"""
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"不可序列化的值: {o!r}")


def merge_overrides(cfg: PhosphorusConfig, overrides: dict[str, Any]) -> PhosphorusConfig:
    """把扁平覆盖表 `{section.field: value}` 应用到配置上。"""
    for sec in ("model", "quant", "train", "decode"):
        if sec not in overrides:
            continue
        obj = getattr(cfg, sec)
        valid = {f.name for f in fields(obj)}
        for k, v in overrides[sec].items():
            if k in valid:
                setattr(obj, k, v)
            else:
                raise KeyError(f"{sec}.{k} 不是合法字段")
    return cfg
