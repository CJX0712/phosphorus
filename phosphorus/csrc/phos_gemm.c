/*
 * phos_gemm.c —— 三值 mpGEMM 内核（TL / I2_S），标量权威 + AVX2 优化。
 *
 * 位精确性契约（不变量 ①，docs/ARCHITECTURE.md §5）：
 *   三条路径在 int32 域必须逐位相等：
 *     phos_tl_gemm_i32  ==  phos_i2s_gemm_i32  ==  numpy int32_reference
 *   整数加法满足交换律/结合律且无溢出（|acc| ≤ K·127 ≤ 4096·127 ≈ 5.2e5），
 *   因此 SIMD 的 lane 分块与标量的顺序累加必然逐位相等 —— 这是可证明的，
 *   不是"实测差不多"。
 *
 * 输出索引约定：`y` / `y32` 按**绝对行号**寻址（调用方传整块缓冲或整块视图），
 * 行块 [n0, n1) 各线程写互斥区间 —— 天然无数据竞争（见 parallel.py）。
 *
 * 借鉴来源：
 *   - I2_S 的 AVX2 路径移植自 Microsoft BitNet.cpp（MIT）的
 *     `src/ggml-bitnet-mad.cpp::ggml_vec_dot_i2_i8_s*`：
 *     `_mm256_maddubs_epi16`(u8×i8→i16, 两两饱和相加) + `_mm256_madd_epi16`
 *     (i16→i32 加宽)。三元权重保证 |W|≤1，故 pair 和 ≤ 510，**不会触发
 *     maddubs 的 int16 饱和** —— 这是该恒等式对三值权重安全的原因。
 *   - TL 的 chunk 解码（一次读 5B → 24 个权重 → 直线 MAC）来自 T4 实测：
 *     实测 2.3×～3.5×（AVX2），而 per-group gather 的 16 项 LUT 只有
 *     1.9×～2.5×、逐 group 解 pack 更是 0.47×～0.67×（性能倒退）。
 */

#include "phos.h"

#include <string.h>

/* ------------------------------------------------------------------------- */
/* code5 -> base-3 码 c 的逆表（32 项）                                        */
/* ------------------------------------------------------------------------- */
/*
 * 编码（packing.py，权威）：
 *   digit d = w + 1 ∈ {0,1,2}；c = d0 + 3·d1 + 9·d2 ∈ [0,26]
 *   code5 = (c == 13) ? 0 : (c <= 12 ? c + 1 : ((26 - c) + 1) | 0x10)
 * 注意 c=0 与 c=26 共享 idx=1，必须以完整 5 bit 区分 —— 故表必须 32 项，
 * 不能用 16 项的一维 idx 表（后者会被 sign=1 的项覆盖）。
 * 非法 code5（0x0E/0x0F/0x10/0x1E/0x1F）映射到 c=13（全零组，贡献 0）。
 */
static const int8_t CODE5_TO_C[32] = {
    /*0x00*/ 13, /*0x01*/ 0,  /*0x02*/ 1,  /*0x03*/ 2,  /*0x04*/ 3,  /*0x05*/ 4,
    /*0x06*/ 5,  /*0x07*/ 6,  /*0x08*/ 7,  /*0x09*/ 8,  /*0x0A*/ 9,  /*0x0B*/ 10,
    /*0x0C*/ 11, /*0x0D*/ 12, /*0x0E*/ 13, /*0x0F*/ 13, /*0x10*/ 13, /*0x11*/ 26,
    /*0x12*/ 25, /*0x13*/ 24, /*0x14*/ 23, /*0x15*/ 22, /*0x16*/ 21, /*0x17*/ 20,
    /*0x18*/ 19, /*0x19*/ 18, /*0x1A*/ 17, /*0x1B*/ 16, /*0x1C*/ 15, /*0x1D*/ 14,
    /*0x1E*/ 13, /*0x1F*/ 13,
};

/*
 * code5 -> 三个三值权重（编译期常量）：把解码里的 `% 3` / `/ 3` / `/ 9`
 * 全部换成 3 次字节加载。T4 实测：标量 TL 由此快 ~2 倍。
 */
static const int8_t CODE5_TO_W3[32][3] = {
    {0,0,0}, /*0x00 c=13*/
    {-1,-1,-1}, /*0x01 c=0*/
    {0,-1,-1}, /*0x02 c=1*/
    {1,-1,-1}, /*0x03 c=2*/
    {-1,0,-1}, /*0x04 c=3*/
    {0,0,-1}, /*0x05 c=4*/
    {1,0,-1}, /*0x06 c=5*/
    {-1,1,-1}, /*0x07 c=6*/
    {0,1,-1}, /*0x08 c=7*/
    {1,1,-1}, /*0x09 c=8*/
    {-1,-1,0}, /*0x0A c=9*/
    {0,-1,0}, /*0x0B c=10*/
    {1,-1,0}, /*0x0C c=11*/
    {-1,0,0}, /*0x0D c=12*/
    {0,0,0}, /*0x0E c=13*/
    {0,0,0}, /*0x0F c=13*/
    {0,0,0}, /*0x10 c=13*/
    {1,1,1}, /*0x11 c=26*/
    {0,1,1}, /*0x12 c=25*/
    {-1,1,1}, /*0x13 c=24*/
    {1,0,1}, /*0x14 c=23*/
    {0,0,1}, /*0x15 c=22*/
    {-1,0,1}, /*0x16 c=21*/
    {1,-1,1}, /*0x17 c=20*/
    {0,-1,1}, /*0x18 c=19*/
    {-1,-1,1}, /*0x19 c=18*/
    {1,1,0}, /*0x1A c=17*/
    {0,1,0}, /*0x1B c=16*/
    {-1,1,0}, /*0x1C c=15*/
    {1,0,0}, /*0x1D c=14*/
    {0,0,0}, /*0x1E c=13*/
    {0,0,0}, /*0x1F c=13*/
};

/* ------------------------------------------------------------------------- */
/* 小端 40-bit chunk 读取                                                      */
/* ------------------------------------------------------------------------- */
/*
 * 组必然跨字节（8 组 × 5 bit = 40 bit 装在 5 字节里），不能按字节对齐处理。
 * 一次读 6 字节再移位；最后一个 chunk 之后可能不足 6 字节可读，逐字节兜底，
 * **绝不越界读**（probe 版曾在此处有 1 字节越界，虽能跑但是 UB）。
 */
static inline uint64_t load6_le(const uint8_t *p, int avail) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    if (avail >= 6) {
        uint64_t v = 0;
        memcpy(&v, p, 6); /* LE 平台：小端解释即所需布局 */
        return v;
    }
#endif
    {
        uint64_t v = 0;
        int i;
        int n = avail < 6 ? avail : 6;
        for (i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8 * i);
        return v;
    }
}

/* fp32 缩放：必须放在 double 域左结合，才能与 numpy 参考逐位相等（见 phos.h）。 */
static inline float scale_out(int32_t acc, float wscale, double sx) {
    return (float)((double)acc * (double)wscale * (double)sx);
}

/* ------------------------------------------------------------------------- */
/* 标量权威路径                                                                */
/* ------------------------------------------------------------------------- */

static void tl_rows_i32_scalar(const uint8_t *Wp, const int8_t *xq, int32_t *y32, int K_pad,
                               int n0, int n1) {
    const int nchunks = K_pad / PHOS_WEIGHTS_PER_CHUNK;
    const int row_bytes = nchunks * PHOS_BYTES_PER_CHUNK;
    int i, cid, g;
    for (i = n0; i < n1; ++i) {
        const uint8_t *row = Wp + (size_t)i * (size_t)row_bytes;
        int32_t acc = 0;
        for (cid = 0; cid < nchunks; ++cid) {
            const int off = cid * PHOS_BYTES_PER_CHUNK;
            const uint64_t v = load6_le(row + off, row_bytes - off);
            for (g = 0; g < PHOS_GROUPS_PER_CHUNK; ++g) {
                const int8_t *w3 = CODE5_TO_W3[(v >> (5 * g)) & 0x1Fu];
                const int8_t *a = xq + cid * PHOS_WEIGHTS_PER_CHUNK + g * 3;
                acc += (int32_t)w3[0] * (int32_t)a[0];
                acc += (int32_t)w3[1] * (int32_t)a[1];
                acc += (int32_t)w3[2] * (int32_t)a[2];
            }
        }
        y32[i] = acc;
    }
}

static void i2s_rows_i32_scalar(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                                int32_t *y32, int K, int n0, int n1) {
    int i, j;
    for (i = n0; i < n1; ++i) {
        const int8_t *row = Wi8 + (size_t)i * (size_t)K;
        int32_t acc = 0;
        for (j = 0; j < K; ++j) acc += (int32_t)row[j] * (int32_t)xu8[j];
        y32[i] = acc - 128 * row_wsum[i];
    }
}

static void tl_rows_f32_scalar(const uint8_t *Wp, const int8_t *xq, const float *wscale, double sx,
                               float *y, int K_pad, int n0, int n1) {
    const int nchunks = K_pad / PHOS_WEIGHTS_PER_CHUNK;
    const int row_bytes = nchunks * PHOS_BYTES_PER_CHUNK;
    int i, cid, g;
    for (i = n0; i < n1; ++i) {
        const uint8_t *row = Wp + (size_t)i * (size_t)row_bytes;
        int32_t acc = 0;
        for (cid = 0; cid < nchunks; ++cid) {
            const int off = cid * PHOS_BYTES_PER_CHUNK;
            const uint64_t v = load6_le(row + off, row_bytes - off);
            for (g = 0; g < PHOS_GROUPS_PER_CHUNK; ++g) {
                const int8_t *w3 = CODE5_TO_W3[(v >> (5 * g)) & 0x1Fu];
                const int8_t *a = xq + cid * PHOS_WEIGHTS_PER_CHUNK + g * 3;
                acc += (int32_t)w3[0] * (int32_t)a[0];
                acc += (int32_t)w3[1] * (int32_t)a[1];
                acc += (int32_t)w3[2] * (int32_t)a[2];
            }
        }
        y[i] = scale_out(acc, wscale[i], sx);
    }
}

static void i2s_rows_f32_scalar(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                                const float *wscale, double sx, float *y, int K, int n0, int n1) {
    int i, j;
    for (i = n0; i < n1; ++i) {
        const int8_t *row = Wi8 + (size_t)i * (size_t)K;
        int32_t acc = 0;
        for (j = 0; j < K; ++j) acc += (int32_t)row[j] * (int32_t)xu8[j];
        y[i] = scale_out(acc - 128 * row_wsum[i], wscale[i], sx);
    }
}

/* ------------------------------------------------------------------------- */
/* AVX2 优化路径（仅 x86）                                                      */
/* ------------------------------------------------------------------------- */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PHOS_HAVE_X86 1
#include <immintrin.h>

/*
 * 运行时 AVX2 探测：CPUID.1:ECX[27](OSXSAVE) & [28](AVX) + XGETBV(0)&0x6==0x6
 * （OS 已启用 XMM/YMM 状态）+ CPUID.7.0:EBX[5](AVX2)。
 * 不用 `__builtin_cpu_supports`：它依赖 libgcc 的 `__cpu_model`，而 zig 的
 * windows-gnu 目标不链接 libgcc（实测 lld-link: undefined symbol: __cpu_model）。
 */
static inline void phos_cpuid(unsigned leaf, unsigned subleaf, unsigned abcd[4]) {
#if defined(_MSC_VER)
    __cpuidex((int *)abcd, (int)leaf, (int)subleaf);
#else
    __asm__ volatile("cpuid"
                     : "=a"(abcd[0]), "=b"(abcd[1]), "=c"(abcd[2]), "=d"(abcd[3])
                     : "a"(leaf), "c"(subleaf));
#endif
}

static inline unsigned long long phos_xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int eax, edx;
    __asm__ volatile(".byte 0x0F, 0x01, 0xD0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
#endif
}

static int cpu_detect_avx2(void) {
    unsigned a[4];
    phos_cpuid(1, 0, a);
    const int osxsave = (a[2] >> 27) & 1;
    const int avx = (a[2] >> 28) & 1;
    if (!osxsave || !avx) return 0;
    if ((phos_xgetbv0() & 0x6u) != 0x6u) return 0; /* XMM + YMM 状态 */
    phos_cpuid(7, 0, a);
    return (a[1] >> 5) & 1; /* EBX bit 5 = AVX2 */
}

/* 8 个 int32 lane 水平求和。 */
__attribute__((target("avx2"))) static inline int32_t hsum8_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E)); /* [2,3,0,1] */
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1)); /* [1,0,3,2] */
    return _mm_cvtsi128_si32(s);
}

/*
 * I2_S AVX2：u8 × i8 -> i16（maddubs，两两饱和相加）-> i32（madd 加宽）-> 累加。
 * 安全性：|xu8| ≤ 255、|W| ≤ 1 -> pair 和 ≤ 510 << 32767，不会触发 int16 饱和。
 */
__attribute__((target("avx2"))) static void i2s_rows_i32_avx2(const int8_t *Wi8,
                                                              const uint8_t *xu8,
                                                              const int32_t *row_wsum,
                                                              int32_t *y32, int K, int n0,
                                                              int n1) {
    const __m256i ones = _mm256_set1_epi16(1);
    int i, j;
    for (i = n0; i < n1; ++i) {
        const int8_t *row = Wi8 + (size_t)i * (size_t)K;
        __m256i accv = _mm256_setzero_si256();
        int32_t acc;
        for (j = 0; j + 32 <= K; j += 32) {
            __m256i xv = _mm256_loadu_si256((const __m256i *)(const void *)(xu8 + j));
            __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(row + j));
            __m256i p16 = _mm256_maddubs_epi16(xv, wv); /* 16 × i16 */
            accv = _mm256_add_epi32(accv, _mm256_madd_epi16(p16, ones));
        }
        acc = hsum8_i32(accv);
        for (; j < K; ++j) acc += (int32_t)row[j] * (int32_t)xu8[j];
        y32[i] = acc - 128 * row_wsum[i];
    }
}

__attribute__((target("avx2"))) static void i2s_rows_f32_avx2(const int8_t *Wi8,
                                                              const uint8_t *xu8,
                                                              const int32_t *row_wsum,
                                                              const float *wscale, double sx,
                                                              float *y, int K, int n0, int n1) {
    const __m256i ones = _mm256_set1_epi16(1);
    int i, j;
    for (i = n0; i < n1; ++i) {
        const int8_t *row = Wi8 + (size_t)i * (size_t)K;
        __m256i accv = _mm256_setzero_si256();
        int32_t acc;
        for (j = 0; j + 32 <= K; j += 32) {
            __m256i xv = _mm256_loadu_si256((const __m256i *)(const void *)(xu8 + j));
            __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(row + j));
            __m256i p16 = _mm256_maddubs_epi16(xv, wv);
            accv = _mm256_add_epi32(accv, _mm256_madd_epi16(p16, ones));
        }
        acc = hsum8_i32(accv);
        for (; j < K; ++j) acc += (int32_t)row[j] * (int32_t)xu8[j];
        y[i] = scale_out(acc - 128 * row_wsum[i], wscale[i], sx);
    }
}

/*
 * TL 的 AVX2 版本：**同一份简单标量循环** + `__attribute__((target("avx2")))`，
 * 让 clang 在 AVX2 目标下自动向量化。
 * 为什么不用手写 intrinsic：T4 实测手写 cvt+madd 版本（0.64x~1.42x）反而不敌
 * 自动向量化 —— 解码（每 chunk 8 次表查找）才是瓶颈，手写 SIMD 只优化了
 * 已经很快的 MAC 部分。零重写原则：让编译器做它擅长的。
 */
__attribute__((target("avx2"))) static void tl_rows_i32_avx2(const uint8_t *Wp,
                                                             const int8_t *xq, int32_t *y32,
                                                             int K_pad, int n0, int n1) {
    const int nchunks = K_pad / PHOS_WEIGHTS_PER_CHUNK;
    const int row_bytes = nchunks * PHOS_BYTES_PER_CHUNK;
    int i, cid, g;
    for (i = n0; i < n1; ++i) {
        const uint8_t *row = Wp + (size_t)i * (size_t)row_bytes;
        int32_t acc = 0;
        for (cid = 0; cid < nchunks; ++cid) {
            const int off = cid * PHOS_BYTES_PER_CHUNK;
            const uint64_t v = load6_le(row + off, row_bytes - off);
            for (g = 0; g < PHOS_GROUPS_PER_CHUNK; ++g) {
                const int8_t *w3 = CODE5_TO_W3[(v >> (5 * g)) & 0x1Fu];
                const int8_t *a = xq + cid * PHOS_WEIGHTS_PER_CHUNK + g * 3;
                acc += (int32_t)w3[0] * (int32_t)a[0];
                acc += (int32_t)w3[1] * (int32_t)a[1];
                acc += (int32_t)w3[2] * (int32_t)a[2];
            }
        }
        y32[i] = acc;
    }
}

__attribute__((target("avx2"))) static void tl_rows_f32_avx2(const uint8_t *Wp,
                                                             const int8_t *xq,
                                                             const float *wscale, double sx,
                                                             float *y, int K_pad, int n0,
                                                             int n1) {
    const int nchunks = K_pad / PHOS_WEIGHTS_PER_CHUNK;
    const int row_bytes = nchunks * PHOS_BYTES_PER_CHUNK;
    int i, cid, g;
    for (i = n0; i < n1; ++i) {
        const uint8_t *row = Wp + (size_t)i * (size_t)row_bytes;
        int32_t acc = 0;
        for (cid = 0; cid < nchunks; ++cid) {
            const int off = cid * PHOS_BYTES_PER_CHUNK;
            const uint64_t v = load6_le(row + off, row_bytes - off);
            for (g = 0; g < PHOS_GROUPS_PER_CHUNK; ++g) {
                const int8_t *w3 = CODE5_TO_W3[(v >> (5 * g)) & 0x1Fu];
                const int8_t *a = xq + cid * PHOS_WEIGHTS_PER_CHUNK + g * 3;
                acc += (int32_t)w3[0] * (int32_t)a[0];
                acc += (int32_t)w3[1] * (int32_t)a[1];
                acc += (int32_t)w3[2] * (int32_t)a[2];
            }
        }
        y[i] = scale_out(acc, wscale[i], sx);
    }
}

#endif /* x86 */

/* ------------------------------------------------------------------------- */
/* 导出函数                                                                    */
/* ------------------------------------------------------------------------- */

PHOS_API int phos_cpu_has_avx2(void) {
#if defined(PHOS_HAVE_X86)
    return cpu_detect_avx2();
#else
    return 0;
#endif
}

PHOS_API int phos_abi_version(void) { return PHOS_ABI_VERSION; }

PHOS_API const char *phos_build_info(void) { return "phos-native abi=1 zig-cc c99"; }

static inline int want_avx2(unsigned flags) {
    if (flags & PHOS_FLAG_SCALAR) return 0;
    if (!(flags & PHOS_FLAG_AVX2)) return 0;
    return phos_cpu_has_avx2();
}

static inline int check_gemm_common(int N, int n0, int n1) {
    if (N <= 0 || n0 < 0 || n1 > N || n0 >= n1) return PHOS_ERR_RANGE;
    return PHOS_OK;
}

PHOS_API int phos_tl_gemm_i32(const uint8_t *Wp, const int8_t *xq, int32_t *y32, int N,
                              int K_pad, int n0, int n1, unsigned flags) {
    int rc = check_gemm_common(N, n0, n1);
    if (rc != PHOS_OK) return rc;
    if (!Wp || !xq || !y32) return PHOS_ERR_NULL;
    if (K_pad <= 0 || K_pad % PHOS_WEIGHTS_PER_CHUNK) return PHOS_ERR_ALIGN;
#if defined(PHOS_HAVE_X86)
    if (want_avx2(flags)) {
        tl_rows_i32_avx2(Wp, xq, y32, K_pad, n0, n1);
        return PHOS_OK;
    }
#endif
    tl_rows_i32_scalar(Wp, xq, y32, K_pad, n0, n1);
    return PHOS_OK;
}

PHOS_API int phos_i2s_gemm_i32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               int32_t *y32, int N, int K, int n0, int n1, unsigned flags) {
    int rc = check_gemm_common(N, n0, n1);
    if (rc != PHOS_OK) return rc;
    if (!Wi8 || !xu8 || !row_wsum || !y32) return PHOS_ERR_NULL;
    if (K <= 0) return PHOS_ERR_RANGE;
#if defined(PHOS_HAVE_X86)
    if (want_avx2(flags)) {
        i2s_rows_i32_avx2(Wi8, xu8, row_wsum, y32, K, n0, n1);
        return PHOS_OK;
    }
#endif
    i2s_rows_i32_scalar(Wi8, xu8, row_wsum, y32, K, n0, n1);
    return PHOS_OK;
}

PHOS_API int phos_tl_gemm_f32(const uint8_t *Wp, const int8_t *xq, const float *wscale, double sx,
                              float *y, int N, int K_pad, int n0, int n1, unsigned flags) {
    int rc = check_gemm_common(N, n0, n1);
    if (rc != PHOS_OK) return rc;
    if (!Wp || !xq || !wscale || !y) return PHOS_ERR_NULL;
    if (K_pad <= 0 || K_pad % PHOS_WEIGHTS_PER_CHUNK) return PHOS_ERR_ALIGN;
#if defined(PHOS_HAVE_X86)
    if (want_avx2(flags)) {
        tl_rows_f32_avx2(Wp, xq, wscale, sx, y, K_pad, n0, n1);
        return PHOS_OK;
    }
#endif
    tl_rows_f32_scalar(Wp, xq, wscale, sx, y, K_pad, n0, n1);
    return PHOS_OK;
}

PHOS_API int phos_i2s_gemm_f32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               const float *wscale, double sx, float *y, int N, int K, int n0,
                               int n1, unsigned flags) {
    int rc = check_gemm_common(N, n0, n1);
    if (rc != PHOS_OK) return rc;
    if (!Wi8 || !xu8 || !row_wsum || !wscale || !y) return PHOS_ERR_NULL;
    if (K <= 0) return PHOS_ERR_RANGE;
#if defined(PHOS_HAVE_X86)
    if (want_avx2(flags)) {
        i2s_rows_f32_avx2(Wi8, xu8, row_wsum, wscale, sx, y, K, n0, n1);
        return PHOS_OK;
    }
#endif
    i2s_rows_f32_scalar(Wi8, xu8, row_wsum, wscale, sx, y, K, n0, n1);
    return PHOS_OK;
}
