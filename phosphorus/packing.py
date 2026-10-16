"""三值权重 5-bit 打包 / 解包（ternary_5bit_v1，ARCHITECTURE.md §3.6）。

借鉴来源：
    * Wang et al., 2024, "BitNet: Scaling 1-bit Transformers" (absmean 三值量化) — A-1
    * Wang et al., 2025, "bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
      （TL / I2_S 两类 mpGEMM，27 项 three_lut）— A-3

数字映射：w ∈ {-1,0,1} -> d = w + 1 ∈ {0,1,2}。
base-3 码 c = d0 + 3·d1 + 9·d2 ∈ [0,26]。
对称约简 27 = 1（全零）+ 13×2（互为相反数） -> 14 个规范型，4 bit 索引足够。
编码：c == 13 -> (idx=0, sign=0)；c <= 12 -> (idx=c+1, sign=0)；c >= 14 -> (idx=27-c, sign=1)
     code5 = (sign << 4) | idx，idx ∈ [0, 13]
8 组（24 权重）-> 40 bit -> 5 字节（小端写入 u64 低 5 字节）。

我们的超越（A-3/A-4）：
    1. 符号 + 索引 **14 项 LUT** 替代 bitnet.cpp 的 27 项（part(26-c) == -part(c)）；
    2. **行宽补齐到 24 的倍数**，使每行字节数恰为 5 的整数倍，消除跨行位跨界读取。

实现约束（T3/C1）：打包与解包必须**全向量化**，禁止 Python 逐元素循环，
吞吐需 ≥ 50 MB/s。本模块所有公开函数均为 O(1) 次 NumPy 调用。
"""

from __future__ import annotations

import numpy as np

# 常量（与 csrc/phos.h 保持一致）
WEIGHTS_PER_CHUNK = 24
BYTES_PER_CHUNK = 5
GROUPS_PER_CHUNK = 8  # 8 组 × 3 权重 = 24
LUT_SIZE = 16  # 14 项有效 + 2 项补齐（16 × int8 恰好一个 XMM 寄存器）

# c -> code5 的预计算表（c ∈ [0,26]）
_C_TO_CODE5 = np.zeros(27, dtype=np.uint8)
for _c in range(27):
    if _c == 13:
        _C_TO_CODE5[_c] = 0  # idx=0, sign=0（全零组）
    elif _c <= 12:
        _C_TO_CODE5[_c] = _c + 1  # idx=c+1, sign=0
    else:
        _C_TO_CODE5[_c] = ((26 - _c) + 1) | 0x10  # idx=(26-c)+1, sign=1

# code5（5 bit，0..31）-> c 的逆表。
# 必须用 32 项而非 14 项：idx=1 同时对应 c=0(sign=0) 与 c=26(sign=1)，
# 只按 idx 建表会被后者覆盖，丢失符号区分。
_CODE5_TO_C = np.zeros(32, dtype=np.int32)
for _c in range(27):
    _CODE5_TO_C[int(_C_TO_CODE5[_c])] = _c

# digit(c, k) - 1 表（k = 0,1,2），用于构建 14 项 LUT
_DIGIT_MINUS_ONE = np.zeros((27, 3), dtype=np.int32)
for _c in range(27):
    _DIGIT_MINUS_ONE[_c] = [_c % 3 - 1, (_c // 3) % 3 - 1, _c // 9 - 1]

# 打包 / 解包用的位移常量
_SHIFT_BITS = np.arange(GROUPS_PER_CHUNK, dtype=np.uint64) * np.uint64(5)
_SHIFT_BYTES = np.arange(BYTES_PER_CHUNK, dtype=np.uint64) * np.uint64(8)


def ternary_to_digits(w: np.ndarray) -> np.ndarray:
    """w ∈ {-1,0,1} -> d ∈ {0,1,2}。"""
    return (np.asarray(w, dtype=np.int8) + 1).astype(np.int8)


def digits_to_ternary(d: np.ndarray) -> np.ndarray:
    """d ∈ {0,1,2} -> w ∈ {-1,0,1}。"""
    return (np.asarray(d, dtype=np.int8) - 1).astype(np.int8)


def k_pad(K: int) -> int:
    """行宽补齐到 24 的倍数（消除跨行位跨界读取）。"""
    return ((K + WEIGHTS_PER_CHUNK - 1) // WEIGHTS_PER_CHUNK) * WEIGHTS_PER_CHUNK


def packed_bytes(N: int, K: int) -> int:
    """打包后的理论字节数 = N · ceil(K/24) · 5。"""
    return N * (k_pad(K) // WEIGHTS_PER_CHUNK) * BYTES_PER_CHUNK


def row_wsum(w: np.ndarray) -> np.ndarray:
    """每行的三值权重之和 Σ_j W[i,j]（int32），供 I2_S 零点校正使用。"""
    return np.asarray(w, dtype=np.int32).sum(axis=1).astype(np.int32)


def to_i8_mirror(w: np.ndarray) -> np.ndarray:
    """三值权重的 int8 镜像（值域就是 {-1,0,1}），供 I2_S 后端使用。"""
    return np.asarray(w, dtype=np.int8)


# --------------------------------------------------------------------------- #
# 字节级 5-bit 打包 / 解包（与 C 内核 / .phos 格式严格一致，全向量化）        #
# --------------------------------------------------------------------------- #


def pack_ternary_v1(w: np.ndarray) -> tuple[np.ndarray, int]:
    """整矩阵三值权重 [N, K] -> (packed uint8, K_pad)。

    `len(packed) == N · (K_pad/24) · 5`。全向量化：base-3 编码 -> 查 code5 表 ->
    u64 位移合并 -> view(uint8) 取低 5 字节（依赖小端字节序，见下方注释）。
    """
    w = np.asarray(w, dtype=np.int8)
    if w.ndim != 2:
        raise ValueError(f"pack_ternary_v1 需要 2D 输入 [N, K]，收到 shape={w.shape}")
    N, K = w.shape
    Kpad = k_pad(K)
    if Kpad != K:
        w = np.concatenate([w, np.zeros((N, Kpad - K), dtype=np.int8)], axis=1)
    nchunks = Kpad // WEIGHTS_PER_CHUNK

    d = w.reshape(N, Kpad // 3, 3).astype(np.int32) + 1  # digit ∈ {0,1,2}
    codes = d[:, :, 0] + 3 * d[:, :, 1] + 9 * d[:, :, 2]  # [N, ng]，0..26
    code5 = _C_TO_CODE5[codes].astype(np.uint64).reshape(N, nchunks, GROUPS_PER_CHUNK)

    # 8 组 × 5 bit = 40 bit，合并进一个 u64 的低 40 位
    v = np.bitwise_or.reduce(code5 << _SHIFT_BITS, axis=2)  # [N, nchunks] uint64
    # view(uint8) 依赖主机小端序（x86/ARM 均为 LE）；大端机需先 byteswap。
    out = v.reshape(-1).view(np.uint8).reshape(-1, 8)[:, :BYTES_PER_CHUNK].reshape(-1)
    return np.ascontiguousarray(out), Kpad


def _codes_from_packed(packed: np.ndarray, N: int, Kpad: int) -> np.ndarray:
    """从打包字节还原 code5 数组 [N, K_pad/3]（uint8，含符号位）。

    `packed` 可以是 numpy uint8 数组，也可以是 `bytes`/`bytearray`
    （loader 的 `PhosFile.raw` 存的就是 bytes）—— `np.asarray(bytes, uint8)`
    会把它当字符串标量而抛 ValueError，必须显式 `frombuffer`。
    """
    nchunks = Kpad // WEIGHTS_PER_CHUNK
    ng = Kpad // 3
    need = N * nchunks * BYTES_PER_CHUNK
    if isinstance(packed, (bytes, bytearray, memoryview)):
        p = np.frombuffer(packed, dtype=np.uint8)
    else:
        p = np.asarray(packed, dtype=np.uint8)
    if p.size < need:
        raise ValueError(f"打包数据不足：需要 {need} 字节，实际 {p.size}")
    p = p[:need].reshape(N, nchunks, BYTES_PER_CHUNK)
    v = np.zeros((N, nchunks), dtype=np.uint64)
    for j in range(BYTES_PER_CHUNK):
        v |= p[:, :, j].astype(np.uint64) << _SHIFT_BYTES[j]
    code5 = (v[:, :, None] >> _SHIFT_BITS[None, None, :]) & np.uint64(0x1F)
    return code5.reshape(N, ng).astype(np.uint8)


def unpack_code5(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """解出 code5 数组 [N, K_pad/3]（uint8）。TL 后端直接拿它查 LUT。"""
    Kpad = k_pad(K)
    return _codes_from_packed(packed, N, Kpad)


def unpack_ternary_v1(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """pack_ternary_v1 的逆运算。返回 [N, K] 三值权重（逐位恒等）。"""
    Kpad = k_pad(K)
    code5 = _codes_from_packed(packed, N, Kpad)  # [N, ng]
    c = _CODE5_TO_C[code5]  # base-3 码 [N, ng]
    d0 = c % 3
    d1 = (c // 3) % 3
    d2 = c // 9
    w = np.stack([d0, d1, d2], axis=-1).reshape(N, Kpad).astype(np.int16) - 1
    return w[:, :K].astype(np.int8)


# --------------------------------------------------------------------------- #
# 14 项 LUT 构建（TL 内核核心；C 内核用 int16 版本，numpy 用 int32 防溢出）    #
# --------------------------------------------------------------------------- #


def build_lut16(a: np.ndarray) -> np.ndarray:
    """给定 3 个 int8 激活 a，构建 16 项部分积表（int32）。

    lut[c+1] = (d0-1)·a0 + (d1-1)·a1 + (d2-1)·a2，其中 (d0,d1,d2) = digit(c)。
    lut[0] = 0（对应 c=13 全零组）。索引 14/15 为 SIMD 对齐补齐项，恒为 0。
    """
    a = np.asarray(a, dtype=np.int32).reshape(3)
    lut = np.zeros(LUT_SIZE, dtype=np.int32)
    # 前 13 项（c = 0..12）由 _DIGIT_MINUS_ONE 一次性点乘得出，无 Python 循环
    lut[1:14] = _DIGIT_MINUS_ONE[:13] @ a
    return lut


def lut_from_code5(lut16: np.ndarray) -> np.ndarray:
    """把 16 项 LUT 展开为 32 项「按完整 code5 直接查表」的表。

    code5 = (sign << 4) | idx：sign=1 时取 -lut[idx]（利用 part(26-c) == -part(c)）。
    这样 TL 内核对每组只需**一次查表 + 一次累加**，无需先解符号再取反。
    """
    lut16 = np.asarray(lut16, dtype=np.int32).reshape(LUT_SIZE)
    table = np.zeros(32, dtype=np.int32)
    idx = np.arange(32, dtype=np.int32)
    sign = (idx >> 4) & 1
    low = idx & 0x0F
    vals = np.where(sign == 1, -lut16[low], lut16[low])
    vals[low >= 14] = 0  # idx ∈ [14,15] 是补齐项，不承载合法编码
    table[:] = vals
    return table
