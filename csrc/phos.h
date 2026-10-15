/*
 * phos.h —— Phosphorus C99 原生内核的公共 ABI 草案（T4 落地，T1 先定义）
 *
 * 设计约束（见 docs/ARCHITECTURE.md §0.3 与 TASKS.md T4）：
 *   1. 纯 C99，不依赖 OpenMP / pthread：内核必须是纯函数、可重入、无全局状态。
 *      并行由 Python 侧 ThreadPoolExecutor 在 row-block 粒度切分，
 *      ctypes 调用自动释放 GIL。k0/k1 参数即用于行块切分。
 *   2. 不用系统编译器：统一由 `python -m ziglang cc -O2` 编译（O4）。
 *   3. 所有导出函数返回 int 错误码，输出一律写调用方提供的缓冲区
 *      （避免跨 FFI 边界的内存所有权问题）。
 *   4. 标量路径是「位精确权威实现」；SIMD 路径只在测试中与之对比。
 *
 * 打包布局（ternary_5bit_v1，ARCHITECTURE.md §3.6）：
 *   24 个三值权重 -> 8 组 5 bit -> 5 字节；行宽补齐到 24 的倍数。
 */
#ifndef PHOS_H
#define PHOS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* 错误码                                                                     */
/* ------------------------------------------------------------------------- */
#define PHOS_OK 0
#define PHOS_ERR_NULL 1
#define PHOS_ERR_RANGE 2
#define PHOS_ERR_ALIGN 3

/* 打包常量：8 组 x 5 bit = 40 bit = 5 字节，对应 24 个三值权重。 */
#define PHOS_GROUPS_PER_CHUNK 8
#define PHOS_WEIGHTS_PER_CHUNK 24
#define PHOS_BYTES_PER_CHUNK 5
#define PHOS_LUT_SIZE 16 /* 14 项有效 + 2 项补齐，恰好 16（SIMD 对齐） */

#ifdef _WIN32
#define PHOS_API __declspec(dllexport)
#else
#define PHOS_API __attribute__((visibility("default")))
#endif

/* ------------------------------------------------------------------------- */
/* 三值 mpGEMM                                                                */
/* ------------------------------------------------------------------------- */

/*
 * TL（Ternary LUT）GEMM：查表法，无乘法，内存受限场景（decode / GEMV）首选。
 * 符号 + 索引 14 项 LUT：part(26-c) == -part(c)，故只需 13 项 + 1 个全零项。
 *
 * Wp       [N, K_pad]  5-bit 打包权重（行宽 K_pad = ceil(K/24)*24）
 * xq       [K]         int8 量化激活（值域 [-127, 127]）
 * wscale   [N]         权重 per-channel scale
 * sx                   激活 per-token scale
 * y        [N]         输出
 * k0, k1               行块切分 [k0, k1)，用于线程池并行
 */
PHOS_API int phos_tl_gemm_f32(const uint8_t *Wp, const int8_t *xq, const float *wscale,
                              float sx, float *y, int N, int K_pad, int k0, int k1);

/*
 * I2_S（int2 + scale）GEMM：u8 x i8 点积 + 零点校正，算力受限场景（prefill）首选。
 * 数学恒等式：sum_j (xu8[j] - 128) * W[i,j] == sum_j xu8[j]*W[i,j] - 128*row_wsum[i]
 * 因此与 TL 在 int32 域逐位相等（不变量 ①）。
 */
PHOS_API int phos_i2s_gemm_f32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               const float *wscale, float sx, float *y, int N, int K,
                               int k0, int k1);

/*
 * 从 5-bit 打包流解出 n_groups 组（每组 3 个三值权重），写入 out[3*n_groups]。
 */
PHOS_API int phos_unpack5(const uint8_t *packed, int8_t *out, int n_groups);

/*
 * 反向：把 n_groups 组三值权重打包为 5-bit 流。
 */
PHOS_API int phos_pack5(const int8_t *w, uint8_t *packed, int n_groups);

/* ------------------------------------------------------------------------- */
/* 逐元素 / 归约算子                                                          */
/* ------------------------------------------------------------------------- */

/*
 * RMSNorm: y[n,:] = x[n,:] / sqrt(mean(x[n,:]^2) + eps) * w[:]
 */
PHOS_API int phos_rmsnorm_f32(const float *x, const float *w, float *y, int N, int D, float eps);

/*
 * 逐行 softmax（数值稳定：先减行最大值）。
 */
PHOS_API int phos_softmax_f32(const float *x, float *y, int N, int D);

/*
 * RoPE：对 [T, H, Dh] 的张量在位置 [pos0, pos0+T) 上施加旋转位置编码。
 * theta 默认 10000.0。相邻两两配对旋转（Su et al., arXiv:2104.09864）。
 */
PHOS_API int phos_rope_f32(const float *x, float *y, int T, int H, int Dh, int pos0, float theta);

#ifdef __cplusplus
}
#endif

#endif /* PHOS_H */
