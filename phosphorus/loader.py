"""加载 `.phos` 模型文件（与 export.py 字节级对称，零 torch 依赖）。

解析 ARCHITECTURE.md §3 的 header + tensor index + data region + tokenizer blob。
返回纯 numpy 权重字典（三值权重已解包回 int8）、标量 scale、HGF 低秩矩阵
与 tokenizer，供 `engine.py` 推理或 `model.py` 校验复用。

借鉴来源：
    * llama.cpp `gguf` 读取器（header / index / data 三段式）— A-16
    * 本项目 export.py（同一字节布局的写入端，本模块为读取端）
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .packing import unpack_ternary_v1
from .tokenizer import Tokenizer

HEADER_BYTES = 256
INDEX_ENTRY_BYTES = 128

DT_F32 = 0
DT_F16 = 1
DT_I32 = 5
DT_TERNARY = 6

F_TIED_EMBED = 1 << 1
F_HAS_TOKENIZER = 1 << 2
F_HAS_ROW_WSUM = 1 << 3
F_HAS_HGF = 1 << 0
F_HAS_EARLY_HEAD = 1 << 4


def _u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def _u64(b: bytes, off: int) -> int:
    return struct.unpack_from("<Q", b, off)[0]


def _f32(b: bytes, off: int) -> float:
    return struct.unpack_from("<f", b, off)[0]


class PhosFile:
    """`.phos` 反序列化结果。

    属性：
        weights: name -> ndarray（三值权重为 int8，其余为 float32）
        scales:  name -> float32（per-channel 缩放，仅三值权重）
        hgf:     name -> (A f32 [r,in], B f32 [out,r], gate float)
                 （.phos 文件中为 f16，载入时升为 f32 供计算）
        tokenizer: 分词器（若文件带 tokenizer blob）
        header:  解析后的 header 字典（含结构超参与量化配置）
        raw:     三值张量的**原始 5-bit 打包字节**（供 TL 后端直接查 LUT，
                 避免「解包成 int8 再算」这条非部署路径掩盖真实成本）
    """

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        scales: dict[str, np.ndarray],
        hgf: dict[str, Any],
        tokenizer: Tokenizer | None,
        header: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        raw: dict[str, bytes] | None = None,
    ) -> None:
        self.weights = weights
        self.scales = scales
        self.hgf = hgf
        self.tokenizer = tokenizer
        self.header = header
        self.metadata = metadata or {}
        self.raw = raw or {}

    def tensor_names(self) -> list[str]:
        return sorted(self.weights)


def load_phos(path_or_bytes: str | Path | bytes) -> PhosFile:
    """从路径或字节加载 `.phos`，返回 PhosFile。"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        blob = bytes(path_or_bytes)
    else:
        blob = Path(path_or_bytes).read_bytes()
    if blob[0:4] != b"PHOS":
        raise ValueError("不是合法的 .phos 文件：magic mismatch")

    header = {
        "n_layer": _u32(blob, 32),
        "d_model": _u32(blob, 36),
        "n_head": _u32(blob, 40),
        "n_kv_head": _u32(blob, 44),
        "d_head": _u32(blob, 48),
        "d_ff": _u32(blob, 52),
        "vocab": _u32(blob, 56),
        "max_seq_len": _u32(blob, 60),
        "rope_theta": _f32(blob, 64),
        "norm_eps": _f32(blob, 68),
        "act": _u32(blob, 72),
        "quant": _u32(blob, 76),
        "w_gran": _u32(blob, 80),
        "block_size": _u32(blob, 84),
        "a_gran": _u32(blob, 88),
        "pack_scheme": _u32(blob, 92),
        "hgf_rank": _u32(blob, 96),
        "hgf_alpha": _f32(blob, 100),
        "hgf_gate_init": _f32(blob, 104),
        "kv_block_size": _u32(blob, 108),
        "n_tensor": _u32(blob, 112),
        "early_exit_layer": _u32(blob, 116),
        "tokenizer_off": _u64(blob, 128),
        "tokenizer_bytes": _u64(blob, 136),
        "metadata_off": _u64(blob, 144),
        "metadata_bytes": _u64(blob, 152),
        "data_region_off": _u64(blob, 160),
        "data_region_bytes": _u64(blob, 168),
        "flags": _u64(blob, 24),
    }
    # early_exit_layer 在导出时以 0xFFFFFFFF 表示 -1
    if header["early_exit_layer"] == 0xFFFFFFFF:
        header["early_exit_layer"] = -1

    weights: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    raw: dict[str, bytes] = {}
    hgf: dict[str, Any] = {}
    n_tensor = header["n_tensor"]
    index_off = HEADER_BYTES

    for i in range(n_tensor):
        base = index_off + i * INDEX_ENTRY_BYTES
        raw_name = blob[base : base + 32]
        name = raw_name.split(b"\x00")[0].decode("utf-8")
        dtype = blob[base + 32]
        ndim = blob[base + 33]
        # pack / tflags @ base+34 / +35
        shape = [_u32(blob, base + 36 + 4 * s) for s in range(4)][:ndim]
        n_elem = _u32(blob, base + 52)
        data_off = _u64(blob, base + 64)
        data_bytes = _u64(blob, base + 72)
        scale_off = _u64(blob, base + 80)
        scale_bytes = _u64(blob, base + 88)
        aux_off = _u64(blob, base + 96)
        aux_bytes = _u64(blob, base + 104)
        r = _u32(blob, base + 112)

        if dtype == DT_TERNARY:
            packed = np.frombuffer(blob, dtype=np.uint8, count=data_bytes, offset=data_off)
            N, K = int(shape[0]), int(shape[1])
            weights[name] = unpack_ternary_v1(packed, N, K)
            raw[name] = bytes(packed)  # 保留打包字节，供 TL 后端查 LUT
        else:
            arr = np.frombuffer(blob, dtype=np.float32, count=n_elem, offset=data_off)
            weights[name] = arr.reshape(shape).copy()

        if scale_bytes:
            s = np.frombuffer(blob, dtype=np.float32, count=scale_bytes // 4, offset=scale_off)
            scales[name] = s.copy()

        if aux_bytes:
            aux = np.frombuffer(blob, dtype=np.float16, count=aux_bytes // 2, offset=aux_off)
            in_f = int(shape[1])
            out_f = int(shape[0])
            a_len = r * in_f
            b_len = out_f * r
            A = aux[:a_len].reshape(r, in_f).astype(np.float32)
            B = aux[a_len : a_len + b_len].reshape(out_f, r).astype(np.float32)
            gate = float(aux[a_len + b_len]) if aux.size > a_len + b_len else 0.0
            hgf[name] = (A, B, gate)

    tokenizer = None
    if header["tokenizer_bytes"]:
        tb = blob[header["tokenizer_off"] : header["tokenizer_off"] + header["tokenizer_bytes"]]
        tlen = struct.unpack_from("<I", tb, 0)[0]
        tokenizer = Tokenizer.load_from_bytes(tb[4 : 4 + tlen])

    metadata: dict[str, Any] | None = None
    if header["metadata_bytes"]:
        mb = blob[header["metadata_off"] : header["metadata_off"] + header["metadata_bytes"]]
        try:
            metadata = json.loads(mb[4:].decode("utf-8"))  # 跳过 4 字节长度前缀
        except Exception:
            metadata = None

    return PhosFile(weights, scales, hgf, tokenizer, header, metadata, raw)
