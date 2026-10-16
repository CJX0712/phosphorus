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

/*
 * 每个内核末尾的 `flags` 位开关。
 * 为什么不用 `phos_set_simd()` 全局开关：约束 #1 要求纯函数、可重入、无全局状态，
 * 全局开关会让 row-block 线程池并行出现数据竞争，也让「标量权威路径」无法与 SIMD
 * 在同一进程内并列对比。故把开关做成显式入参（由 software-architect-2 提出）。
 */
#define PHOS_FLAG_SCALAR 1u /* 强制走标量路径（位精确权威实现） */
#define PHOS_FLAG_AVX2 2u   /* 允许 AVX2 路径（需要运行时 CPU 探测已通过） */
#define PHOS_FLAG_NEON 4u   /* 允许 ARM NEON 路径 */

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
 * y        [N]         输出（fp32）
 * n0, n1               行块切分 [n0, n1)，切的是 **N 维（输出行）**，不是 K 维
 *                      —— 切 K 维会破坏累加完整性（ARCHITECTURE.md §5.1）
 * flags                PHOS_FLAG_* 位掩码
 *
 * 注意：f32 版本无法与 NumPy 参考逐位相等（C 是 fp32 左结合，NumPy 参考是 float64）。
 * 位精确断言一律走下面的 `*_i32` 整数域版本。
 */
PHOS_API int phos_tl_gemm_f32(const uint8_t *Wp, const int8_t *xq, const float *wscale,
                              float sx, float *y, int N, int K_pad, int n0, int n1,
                              unsigned flags);

/*
 * I2_S（int2 + scale）GEMM：u8 x i8 点积 + 零点校正，算力受限场景（prefill）首选。
 * 数学恒等式：sum_j (xu8[j] - 128) * W[i,j] == sum_j xu8[j]*W[i,j] - 128*row_wsum[i]
 * 因此与 TL 在 int32 域逐位相等（不变量 ①）。
 */
PHOS_API int phos_i2s_gemm_f32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               const float *wscale, float sx, float *y, int N, int K,
                               int n0, int n1, unsigned flags);

/*
 * 整数域输出版本 —— 位精确性的可观测锚点（不变量 ①）。
 * `y32[n] = sum_j W[n,j] * xq[j]`，纯 int32 累加，不做任何浮点缩放。
 * native.tl_i32 == native.i2s_i32 == numpy int32_reference 必须是逐位相等。
 */
PHOS_API int phos_tl_gemm_i32(const uint8_t *Wp, const int8_t *xq, int32_t *y32, int N,
                              int K_pad, int n0, int n1, unsigned flags);

PHOS_API int phos_i2s_gemm_i32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               int32_t *y32, int N, int K, int n0, int n1, unsigned flags);

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
