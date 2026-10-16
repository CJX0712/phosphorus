"""导出 `.phos` 模型文件（ARCHITECTURE.md §3，字节级精确）。

借鉴来源：GGUF (llama.cpp) 的「header + tensor index + data region」三段式布局 — A-16。
本项目差异：张量 entry 增加三值/HGF/row_wsum 字段，整体系列化 1.58-bit 权重。

零 torch 依赖：本模块只接收 numpy 权重字典（来自 model.export_numpy()）。
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .config import ModelConfig, QuantConfig
from .packing import pack_ternary_v1

MAGIC = b"PHOS"
ALIGN = 64
HEADER_BYTES = 256
INDEX_ENTRY_BYTES = 128

DT_F32 = 0
DT_F16 = 1
DT_I32 = 5
DT_TERNARY = 6

PACK_NONE = 0
PACK_TERNARY = 1

T_PACKED = 1 << 0
T_HAS_SCALE = 1 << 1
T_HAS_HGF = 1 << 2
T_HAS_ROW_WSUM = 1 << 3

F_TIED_EMBED = 1 << 1
F_HAS_TOKENIZER = 1 << 2
F_HAS_ROW_WSUM = 1 << 3
F_HAS_HGF = 1 << 0
F_HAS_EARLY_HEAD = 1 << 4


def _ternary_weight_keys() -> tuple:
    return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def export_model(
    weights: dict[str, np.ndarray],
    mcfg: ModelConfig,
    qcfg: QuantConfig,
    tokenizer=None,
    path: str | Path | None = None,
    metadata: dict | None = None,
) -> bytes:
    """把 numpy 权重字典写成 .phos 字节流（并可选落盘）。返回字节。"""
    entries = []
    for key, arr in weights.items():
        if any(key.endswith(s) for s in (".scale", ".row_wsum", ".hgf_a", ".hgf_b", ".hgf_gate")):
            continue
        arr = np.asarray(arr)
        is_tern = "layers." in key and any(
            key.endswith(f".{p}.weight") for p in _ternary_weight_keys()
        )
        if is_tern:
            ternary = arr.astype(np.int8)
            packed, _ = pack_ternary_v1(ternary)
            # export_numpy 把 scale/hgf 存为 `key` 去掉 `.weight` 后缀，再拼 `.scale`/`.hgf_*`
            base_key = key[: -len(".weight")] if key.endswith(".weight") else key
            scale = weights.get(
                base_key + ".scale", np.ones(arr.shape[0], dtype=np.float32)
            ).astype(np.float32)
            row_wsum = ternary.astype(np.int32).sum(axis=1).astype(np.int32)
            hgf_a = weights.get(base_key + ".hgf_a")
            hgf_b = weights.get(base_key + ".hgf_b")
            if hgf_a is not None and hgf_b is not None:
                A = np.asarray(hgf_a, dtype=np.float16)
                B = np.asarray(hgf_b, dtype=np.float16)
                # 末尾追加 hgf_gate 标量（f16），保证引擎可完整复现 HGF 校正。
                gate = float(weights.get(base_key + ".hgf_gate", 0.0))
                aux = np.concatenate(
                    [A.reshape(-1), B.reshape(-1), np.array([gate], dtype=np.float16)]
                ).tobytes()
                aux_layout = 1
                hgf_rank = int(A.shape[0])
            else:
                aux, aux_layout, hgf_rank = b"", 0, 0
            entries.append(
                {
                    "name": key,
                    "dtype": DT_TERNARY,
                    "shape": tuple(arr.shape),
                    "pack": PACK_TERNARY,
                    "data": bytes(packed),
                    "scale": scale,
                    "row_wsum": row_wsum,
                    "aux": aux,
                    "aux_layout": aux_layout,
                    "hgf_rank": hgf_rank,
                    "tflags": T_PACKED
                    | T_HAS_SCALE
                    | T_HAS_ROW_WSUM
                    | (T_HAS_HGF if hgf_rank else 0),
                }
            )
        else:
            entries.append(
                {
                    "name": key,
                    "dtype": DT_F32,
                    "shape": tuple(arr.shape),
                    "pack": PACK_NONE,
                    "data": arr.astype(np.float32).tobytes(),
                    "scale": None,
                    "row_wsum": None,
                    "aux": b"",
                    "aux_layout": 0,
                    "hgf_rank": 0,
                    "tflags": 0,
                }
            )
    n_tensor = len(entries)

    buf = bytearray(HEADER_BYTES + n_tensor * INDEX_ENTRY_BYTES)

    struct.pack_into("<4s", buf, 0, MAGIC)
    buf[4] = 1
    buf[5] = 0
    buf[6] = 0
    buf[7] = ALIGN
    struct.pack_into("<I", buf, 8, HEADER_BYTES)
    struct.pack_into("<I", buf, 12, INDEX_ENTRY_BYTES)
    struct.pack_into("<I", buf, 32, mcfg.n_layer)
    struct.pack_into("<I", buf, 36, mcfg.d_model)
    struct.pack_into("<I", buf, 40, mcfg.n_head)
    struct.pack_into("<I", buf, 44, mcfg.n_kv_head)
    struct.pack_into("<I", buf, 48, mcfg.d_head)
    struct.pack_into("<I", buf, 52, mcfg.d_ff)
    struct.pack_into("<I", buf, 56, mcfg.vocab_size)
    struct.pack_into("<I", buf, 60, mcfg.max_seq_len)
    struct.pack_into("<f", buf, 64, mcfg.rope_theta)
    struct.pack_into("<f", buf, 68, mcfg.norm_eps)
    struct.pack_into("<I", buf, 72, 1 if mcfg.act.value == "relu2" else 0)
    struct.pack_into("<I", buf, 76, {"fp32": 0, "int8": 1, "ternary": 2}[qcfg.scheme.value])
    struct.pack_into(
        "<I", buf, 80, {"per_tensor": 0, "per_channel": 1, "per_block": 2}[qcfg.w_granularity.value]
    )
    struct.pack_into("<I", buf, 84, qcfg.block_size)
    struct.pack_into(
        "<I",
        buf,
        88,
        {"per_tensor": 0, "per_channel": 1, "per_block": 2, "per_token": 3}[
            qcfg.a_granularity.value
        ],
    )
    struct.pack_into("<I", buf, 92, PACK_TERNARY if qcfg.scheme.value == "ternary" else PACK_NONE)
    struct.pack_into("<I", buf, 96, qcfg.hgf_rank)
    struct.pack_into("<f", buf, 100, qcfg.hgf_alpha)
    struct.pack_into("<f", buf, 104, qcfg.hgf_gate_init)
    struct.pack_into("<I", buf, 108, 16)  # kv_block_size
    struct.pack_into("<I", buf, 112, n_tensor)
    struct.pack_into(
        "<I", buf, 116, mcfg.early_exit_layer if mcfg.early_exit_layer >= 0 else 0xFFFFFFFF
    )
    struct.pack_into("<Q", buf, 120, HEADER_BYTES)

    flags = 0
    if mcfg.tied_embed:
        flags |= F_TIED_EMBED
    if any(e["hgf_rank"] for e in entries):
        flags |= F_HAS_HGF
    if any(e["row_wsum"] is not None for e in entries):
        flags |= F_HAS_ROW_WSUM
    if "early_lora_a" in weights:
        flags |= F_HAS_EARLY_HEAD
    if tokenizer is not None:
        flags |= F_HAS_TOKENIZER
    struct.pack_into("<Q", buf, 24, flags)

    crc = zlib.crc32(bytes(buf[0:176])) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 176, crc)

    # 写 tensor index 元数据（data/scale/aux 偏移回填）
    for i, e in enumerate(entries):
        base = HEADER_BYTES + i * INDEX_ENTRY_BYTES
        name_b = e["name"].encode("utf-8")[:32]
        buf[base : base + len(name_b)] = name_b
        buf[base + 32] = e["dtype"]
        buf[base + 33] = len(e["shape"])
        buf[base + 34] = e["pack"]
        buf[base + 35] = e["tflags"]
        for si, sdim in enumerate(e["shape"]):
            struct.pack_into("<I", buf, base + 36 + si * 4, int(sdim))
        struct.pack_into("<I", buf, base + 52, int(np.prod(e["shape"])) if e["shape"] else 0)
        struct.pack_into("<I", buf, base + 56, 1 if e["scale"] is not None else 0)
        struct.pack_into("<I", buf, base + 60, len(e["scale"]) if e["scale"] is not None else 0)
        struct.pack_into("<I", buf, base + 112, e["hgf_rank"])
        struct.pack_into("<I", buf, base + 116, e["aux_layout"])

    # 数据区
    while len(buf) % ALIGN:
        buf += b"\x00"
    data_region_off = len(buf)
    for i, e in enumerate(entries):
        base = HEADER_BYTES + i * INDEX_ENTRY_BYTES
        data_off = len(buf)
        buf += e["data"]
        while len(buf) % ALIGN:
            buf += b"\x00"
        struct.pack_into("<Q", buf, base + 64, data_off)
        struct.pack_into("<Q", buf, base + 72, len(buf) - data_off)
        if e["scale"] is not None:
            scale_off = len(buf)
            buf += e["scale"].astype(np.float32).tobytes()
            while len(buf) % 4:
                buf += b"\x00"
            while len(buf) % ALIGN:
                buf += b"\x00"
            struct.pack_into("<Q", buf, base + 80, scale_off)
            struct.pack_into("<Q", buf, base + 88, len(buf) - scale_off)
        if e["aux"]:
            aux_off = len(buf)
            buf += e["aux"]
            while len(buf) % ALIGN:
                buf += b"\x00"
            struct.pack_into("<Q", buf, base + 96, aux_off)
            struct.pack_into("<Q", buf, base + 104, len(buf) - aux_off)

    # Tokenizer blob
    tokenizer_off = len(buf)
    if tokenizer is not None:
        tok_bytes = tokenizer.save_to_bytes()
        buf += struct.pack("<I", len(tok_bytes)) + tok_bytes
        while len(buf) % ALIGN:
            buf += b"\x00"
    tokenizer_bytes = len(buf) - tokenizer_off
    struct.pack_into("<Q", buf, 128, tokenizer_off)
    struct.pack_into("<Q", buf, 136, tokenizer_bytes)

    # Metadata
    metadata_off = len(buf)
    meta = {
        "author": "晨星 (CJX0712)",
        "project": "phosphorus",
        "version": "0.1.0",
        "quant": qcfg.scheme.value,
        "hgf_rank": qcfg.hgf_rank,
        **(metadata or {}),
    }
    meta_bytes = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    buf += struct.pack("<I", len(meta_bytes)) + meta_bytes
    metadata_bytes = len(buf) - metadata_off
    struct.pack_into("<Q", buf, 144, metadata_off)
    struct.pack_into("<Q", buf, 152, metadata_bytes)
    struct.pack_into("<Q", buf, 160, data_region_off)
    struct.pack_into("<Q", buf, 168, len(buf) - data_region_off)
    struct.pack_into("<Q", buf, 16, len(buf))
    blob = bytes(buf)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(blob)
    return blob


# --------------------------------------------------------------------------- #
# 检视（`phosphorus quantize --inspect`，调试 .phos 的主要工具）               #
# --------------------------------------------------------------------------- #

_DTYPE_NAMES = {DT_F32: "f32", DT_F16: "f16", DT_I32: "i32", DT_TERNARY: "ternary_packed"}
_PACK_NAMES = {PACK_NONE: "none", PACK_TERNARY: "ternary_5bit_v1"}


def inspect_phos(path: str | Path) -> str:
    """读取 .phos 并生成人类可读的结构报告（header + 张量索引 + 完整性校验）。"""
    blob = Path(path).read_bytes()
    lines: list[str] = []
    lines.append(f"文件: {path}  ({len(blob):,} bytes)")
    if blob[0:4] != MAGIC:
        return "\n".join(lines + ["错误: magic 不是 PHOS"])
    ver = f"{blob[4]}.{blob[5]}.{blob[6]}"
    n_tensor = struct.unpack_from("<I", blob, 112)[0]
    crc_stored = struct.unpack_from("<I", blob, 176)[0]
    crc_calc = zlib.crc32(bytes(blob[0:176])) & 0xFFFFFFFF
    lines.append(f"版本: {ver}  对齐: {blob[7]}  张量数: {n_tensor}")
    lines.append(
        f"结构: n_layer={struct.unpack_from('<I', blob, 32)[0]} "
        f"d_model={struct.unpack_from('<I', blob, 36)[0]} "
        f"n_head={struct.unpack_from('<I', blob, 40)[0]} "
        f"n_kv_head={struct.unpack_from('<I', blob, 44)[0]} "
        f"d_ff={struct.unpack_from('<I', blob, 52)[0]} "
        f"vocab={struct.unpack_from('<I', blob, 56)[0]}"
    )
    lines.append(
        f"量化: quant_enum={struct.unpack_from('<I', blob, 76)[0]} "
        f"pack_scheme={_PACK_NAMES.get(struct.unpack_from('<I', blob, 92)[0], '?')} "
        f"hgf_rank={struct.unpack_from('<I', blob, 96)[0]}"
    )
    lines.append(f"crc32_header: stored=0x{crc_stored:08x} calc=0x{crc_calc:08x} "
                 f"{'OK' if crc_stored == crc_calc else 'MISMATCH'}")
    lines.append("")
    lines.append(f"{'#':>3} {'name':<34} {'dtype':<15} {'shape':<16} {'offset':>10} {'bytes':>10}")
    total_packed = 0
    for i in range(n_tensor):
        base = HEADER_BYTES + i * INDEX_ENTRY_BYTES
        name = blob[base : base + 32].split(b"\x00")[0].decode("utf-8")
        dtype = blob[base + 32]
        ndim = blob[base + 33]
        shape = [struct.unpack_from("<I", blob, base + 36 + 4 * s)[0] for s in range(4)][:ndim]
        off = struct.unpack_from("<Q", blob, base + 64)[0]
        nb = struct.unpack_from("<Q", blob, base + 72)[0]
        if off + nb > len(blob):
            lines.append(f"  !! 张量 {name} 越界: off={off} bytes={nb} file={len(blob)}")
        if dtype == DT_TERNARY:
            total_packed += nb
        lines.append(
            f"{i:>3} {name:<34} {_DTYPE_NAMES.get(dtype, str(dtype)):<15} "
            f"{str(shape):<16} {off:>10} {nb:>10}"
        )
    lines.append("")
    lines.append(f"三值打包张量总字节: {total_packed:,} "
                 f"（占文件 {total_packed / max(len(blob), 1) * 100:.1f}%）")
    return "\n".join(lines)
