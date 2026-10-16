/*
 * phos_pack.c —— 三值 5-bit 打包 / 解包（packing.py 的 C 版）。
 *
 * 编码（与 phosphorus/packing.py 逐字节一致，权威）：
 *   digit d = w + 1 ∈ {0,1,2}；c = d0 + 3·d1 + 9·d2 ∈ [0,26]
 *   code5 = (c == 13) ? 0 : (c <= 12 ? c + 1 : ((26 - c) + 1) | 0x10)
 *   8 组 code5 = 40 bit = 5 字节，组成一个小端整数；组 g 占据 bit [5g, 5g+5)。
 *
 * 顺带还清 T3 C1 的欠账：`packing.pack_ternary_v1` 是 Python 三重循环
 * （违反"打包吞吐 ≥ 50 MB/s，禁逐元素循环"），本文件提供向量化友好的 C 版。
 */

#include "phos.h"

/* code5 -> base-3 码 c（32 项；非法码 -> 13 即全零组）。与 phos_gemm.c 同表。 */
static const int8_t CODE5_TO_C[32] = {
    /*0x00*/ 13, /*0x01*/ 0,  /*0x02*/ 1,  /*0x03*/ 2,  /*0x04*/ 3,  /*0x05*/ 4,
    /*0x06*/ 5,  /*0x07*/ 6,  /*0x08*/ 7,  /*0x09*/ 8,  /*0x0A*/ 9,  /*0x0B*/ 10,
    /*0x0C*/ 11, /*0x0D*/ 12, /*0x0E*/ 13, /*0x0F*/ 13, /*0x10*/ 13, /*0x11*/ 26,
    /*0x12*/ 25, /*0x13*/ 24, /*0x14*/ 23, /*0x15*/ 22, /*0x16*/ 21, /*0x17*/ 20,
    /*0x18*/ 19, /*0x19*/ 18, /*0x1A*/ 17, /*0x1B*/ 16, /*0x1C*/ 15, /*0x1D*/ 14,
    /*0x1E*/ 13, /*0x1F*/ 13,
};

static inline uint8_t c_to_code5(int c) {
    if (c == 13) return 0;
    if (c <= 12) return (uint8_t)(c + 1);
    return (uint8_t)(((26 - c) + 1) | 0x10);
}

PHOS_API int phos_pack5(const int8_t *w, uint8_t *packed, int n_groups) {
    uint64_t acc = 0;
    int bits = 0;
    int ob = 0;
    int g;
    if (!w || !packed) return PHOS_ERR_NULL;
    if (n_groups < 0) return PHOS_ERR_RANGE;
    for (g = 0; g < n_groups; ++g) {
        int d0 = w[3 * g + 0] + 1;
        int d1 = w[3 * g + 1] + 1;
        int d2 = w[3 * g + 2] + 1;
        int c;
        if (d0 < 0 || d0 > 2 || d1 < 0 || d1 > 2 || d2 < 0 || d2 > 2) return PHOS_ERR_RANGE;
        c = d0 + 3 * d1 + 9 * d2;
        acc |= (uint64_t)c_to_code5(c) << bits;
        bits += 5;
        while (bits >= 8) {
            packed[ob++] = (uint8_t)(acc & 0xFFu);
            acc >>= 8;
            bits -= 8;
        }
    }
    if (bits > 0) packed[ob++] = (uint8_t)(acc & 0xFFu);
    return PHOS_OK;
}

PHOS_API int phos_unpack5(const uint8_t *packed, int8_t *out, int n_groups) {
    uint64_t acc = 0;
    int bits = 0;
    int ib = 0;
    int g;
    if (!packed || !out) return PHOS_ERR_NULL;
    if (n_groups < 0) return PHOS_ERR_RANGE;
    for (g = 0; g < n_groups; ++g) {
        int c;
        while (bits < 5) {
            acc |= (uint64_t)packed[ib++] << bits;
            bits += 8;
        }
        c = CODE5_TO_C[(int)(acc & 0x1Fu)];
        acc >>= 5;
        bits -= 5;
        out[3 * g + 0] = (int8_t)((c % 3) - 1);
        out[3 * g + 1] = (int8_t)(((c / 3) % 3) - 1);
        out[3 * g + 2] = (int8_t)((c / 9) - 1);
    }
    return PHOS_OK;
}
