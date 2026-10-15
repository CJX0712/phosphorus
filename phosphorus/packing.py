"""三值权重 5-bit 打包 / 解包（ternary_5bit_v1，ARCHITECTURE.md §3.6）。

借鉴来源：
    * Wang et al., 2024, "BitNet: Scaling 1-bit Transformers" (MA, absmean) — A-1
    * Wang et al., 2025, "bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
      (TL / I2_S 内核，27 项 three_lut) — A-3
    * 原创微创新（本项目 A-3）：符号 + 索引 14 项 LUT（替代 bitnet.cpp 27 项），
      16 项补齐恰好一个 XMM 寄存器；行宽补齐到 24 的倍数消除跨行位跨界。

数字映射：-1 -> 0, 0 -> 1, +1 -> 2（d ∈ {0,1,2}）。
base-3 码 c = d0 + 3 d1 + 9 d2 ∈ [0,26]。
对称约简 27 = 1(全零) + 13×2(互为相反数) -> 14 个规范型，4 bit 索引足矣。
8 组 (24 权重) -> 40 bit = 5 字节。
"""

from __future__ import annotations

import numpy as np

# 常量
WEIGHTS_PER_CHUNK = 24
BYTES_PER_CHUNK = 5
GROUPS_PER_CHUNK = 8  # 8 组 × 3 权重 = 24
LUT_SIZE = 16  # 14 项有效 + 2 项补齐

# c -> 规范型符号/索引 的预计算表（c ∈ [0,26] -> (index, sign)）
_C_TO_CODE5 = np.zeros(27, dtype=np.uint8)
for _c in range(27):
    if _c == 13:
        _C_TO_CODE5[_c] = 0  # index=0, sign=0
    elif _c <= 12:
        _C_TO_CODE5[_c] = _c + 1  # index=c+1, sign=0
    else:
        _C_TO_CODE5[_c] = ((26 - _c) + 1) | 0x10  # index=(26-c)+1, sign=1
# 逆表：以完整 5-bit code5（含符号位，0..31）为索引 -> c。
# 注意：c=0(idx1,sign0) 与 c=26(idx1,sign1) 共享 idx，必须以完整 code5 区分，
# 一维 (idx) 表会被后者覆盖，故用 32 项表。
_CODE5_TO_C = np.zeros(32, dtype=np.int8)
for _c in range(27):
    _CODE5_TO_C[int(_C_TO_CODE5[_c])] = _c
# digit(c,k) - 1 表（k=0,1,2）
_DIGIT_MINUS_ONE = np.zeros((27, 3), dtype=np.int8)
for _c in range(27):
    _d0, _d1, _d2 = _c % 3, (_c // 3) % 3, _c // 9
    _DIGIT_MINUS_ONE[_c] = [_d0 - 1, _d1 - 1, _d2 - 1]


def ternary_to_digits(w: np.ndarray) -> np.ndarray:
    """w ∈ {-1,0,1} -> d ∈ {0,1,2}。"""
    return (np.asarray(w, dtype=np.int8) + 1).astype(np.int8)


def digits_to_ternary(d: np.ndarray) -> np.ndarray:
    """d ∈ {0,1,2} -> w ∈ {-1,0,1}。"""
    return (np.asarray(d, dtype=np.int8) - 1).astype(np.int8)


def k_pad(K: int) -> int:
    """行宽补齐到 24 的倍数。"""
    return ((K + WEIGHTS_PER_CHUNK - 1) // WEIGHTS_PER_CHUNK) * WEIGHTS_PER_CHUNK


# --------------------------------------------------------------------------- #
# 单行长整型打包 / 解包（不走字节，纯数值；用于测试与 numpy 路径）            #
# --------------------------------------------------------------------------- #


def pack_row_digits(d: np.ndarray) -> np.ndarray:
    """一行 digit 序列 (长度 K_pad) -> base-3 码 c ∈ [0,26]（长度 K_pad/3）。

    用于 numpy 路径：c 直接索引 _DIGIT_MINUS_ONE 得到三元组权重。
    """
    d = np.asarray(d, dtype=np.int8)
    Kpad = k_pad(d.shape[0]) if d.shape[0] % WEIGHTS_PER_CHUNK else d.shape[0]
    if d.shape[0] != Kpad:
        d = np.concatenate([d, np.full(Kpad - d.shape[0], 1, dtype=np.int8)])
    c = d[0::3].astype(np.int32) + 3 * d[1::3].astype(np.int32) + 9 * d[2::3].astype(np.int32)
    return c.astype(np.int8)


def _chunks_of_3(w: np.ndarray) -> np.ndarray:
    """一行三值权重 (长度 K) -> 形状 (K_pad/3, 3) 的权重三元组。"""
    w = np.asarray(w, dtype=np.int8)
    Kpad = k_pad(w.shape[0])
    if w.shape[0] != Kpad:
        w = np.concatenate([w, np.zeros(Kpad - w.shape[0], dtype=np.int8)])
    return w.reshape(Kpad // 3, 3)


def tl_partials(w: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """纯数值 TL 部分积（不变量 ① 的权威整数路径）。

    对每个 3 权重组：partial = w0*x0 + w1*x1 + w2*x2（int32）。
    返回每行 [ng] 的部分积数组。
    """
    w = np.asarray(w, dtype=np.int8)
    xq = np.asarray(xq, dtype=np.int8)
    Kpad = k_pad(w.shape[1])
    Wg = _chunks_of_3(w.reshape(-1, w.shape[1])[0] if w.ndim == 1 else w[0])
    # 支持 2D [N, K]
    if w.ndim == 1:
        ng = Kpad // 3
        xg = xq[:Kpad].reshape(ng, 3).astype(np.int32)
        part = (Wg.astype(np.int32) * xg).sum(axis=1)
        return part
    N = w.shape[0]
    ng = Kpad // 3
    Wg = _chunks_of_3(w).reshape(N, ng, 3)
    xg = xq[:Kpad].reshape(ng, 3).astype(np.int32)
    part = (Wg.astype(np.int32) * xg[None, :, :]).sum(axis=2)  # [N, ng]
    return part


# --------------------------------------------------------------------------- #
# 字节级 5-bit 打包 / 解包（与 C 内核 / .phos 格式严格一致）                   #
# --------------------------------------------------------------------------- #


def pack_ternary_v1(w: np.ndarray) -> tuple[np.ndarray, int]:
    """整矩阵三值权重 [N, K] -> 打包字节 (length N * K_pad/24*5) + K_pad。

    返回 (packed: bytes-like uint8, K_pad)。
    """
    w = np.asarray(w, dtype=np.int8)
    assert w.ndim == 2
    N, K = w.shape
    Kpad = k_pad(K)
    if Kpad != K:
        w = np.concatenate([w, np.zeros((N, Kpad - K), dtype=np.int8)], axis=1)
    ng = Kpad // 3
    nchunks = Kpad // WEIGHTS_PER_CHUNK
    d = w.reshape(N, ng, 3).astype(np.int32) + 1  # digit ∈ {0,1,2}
    codes = d[:, :, 0] + 3 * d[:, :, 1] + 9 * d[:, :, 2]  # [N, ng] 0..26
    code5 = _C_TO_CODE5[codes]  # [N, ng] uint8（index+sign）
    out = np.zeros(N * nchunks * BYTES_PER_CHUNK, dtype=np.uint8)
    # 逐 chunk（8 组 = 40 bit）打包
    for row in range(N):
        for ch in range(nchunks):
            v = 0
            for g in range(GROUPS_PER_CHUNK):
                v |= int(code5[row, ch * GROUPS_PER_CHUNK + g]) << (5 * g)
            start = (row * nchunks + ch) * BYTES_PER_CHUNK
            out[start : start + BYTES_PER_CHUNK] = np.frombuffer(
                v.to_bytes(BYTES_PER_CHUNK, "little"), dtype=np.uint8
            )
    return out, Kpad


def unpack_ternary_v1(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """pack_ternary_v1 的逆运算。返回 [N, K] 三值权重。"""
    packed = np.asarray(packed, dtype=np.uint8)
    Kpad = k_pad(K)
    nchunks = Kpad // WEIGHTS_PER_CHUNK
    ng = Kpad // 3
    codes = np.zeros((N, ng), dtype=np.int8)
    for row in range(N):
        for ch in range(nchunks):
            start = (row * nchunks + ch) * BYTES_PER_CHUNK
            v = int.from_bytes(bytes(packed[start : start + BYTES_PER_CHUNK]), "little")
            for g in range(GROUPS_PER_CHUNK):
                code5 = (v >> (5 * g)) & 0x1F
                codes[row, ch * GROUPS_PER_CHUNK + g] = _CODE5_TO_C[code5]
    # codes -> digit d ∈ {0,1,2} -> 三值权重
    w = np.zeros((N, ng, 3), dtype=np.int8)
    w[:, :, 0] = codes % 3
    w[:, :, 1] = (codes // 3) % 3
    w[:, :, 2] = codes // 9
    w = digits_to_ternary(w.reshape(N, ng * 3))
    return w[:, :K].copy()


# --------------------------------------------------------------------------- #
# 14 项 LUT 构建（供 C 内核与 numpy 对照）                                    #
# --------------------------------------------------------------------------- #


def build_lut16(a: np.ndarray) -> np.ndarray:
    """给定 3 个 int8 激活 a，构建 16 项部分积表。

    lut[c+1] = (d0-1)*a0 + (d1-1)*a1 + (d2-1)*a2，其中 (d0,d1,d2)=digit(c)。
    lut[0] = 0（c=13 全零）。返回 int16[16]。
    """
    a = np.asarray(a, dtype=np.int16)
    lut = np.zeros(LUT_SIZE, dtype=np.int16)
    for c in range(13):
        d0, d1, d2 = _DIGIT_MINUS_ONE[c]
        lut[c + 1] = int(d0) * int(a[0]) + int(d1) * int(a[1]) + int(d2) * int(a[2])
    return lut
