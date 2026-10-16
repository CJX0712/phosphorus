/*
 * phos.h —— Phosphorus C99 原生内核的公共 ABI（T4 落地）
 *
 * 设计约束（见 docs/ARCHITECTURE.md §0.3 与 TASKS.md T4）：
 *   1. 纯 C99，不依赖 OpenMP / pthread：内核必须是纯函数、可重入、无全局状态。
 *      并行由 Python 侧 ThreadPoolExecutor 在 **N 维 row-block** 粒度切分
 *      （`n0`/`n1`），ctypes 调用自动释放 GIL。
 *      ⚠️ 切 K 维会破坏累加完整性（ARCHITECTURE.md §5.1），绝不允许。
 *   2. 不用系统编译器：统一由 `python -m ziglang cc` 编译（O4），显式 `-target`
 *      以 baseline CPU 模型产出（否则产物含 AVX-512，在旧机器上 SIGILL）。
 *   3. 所有导出函数返回 int 错误码，输出一律写调用方提供的缓冲区
 *      （避免跨 FFI 边界的内存所有权问题）。
 *   4. 标量路径是「位精确权威实现」；SIMD 路径只在测试中与之对比，
 *      且必须逐位相等（整数加法满足交换律/结合律）。
 *
 * 打包布局（ternary_5bit_v1，ARCHITECTURE.md §3.6）：
 *   24 个三值权重 -> 8 组 5 bit -> 5 字节；行宽补齐到 24 的倍数。
 *   8 组 code5 组成一个 40-bit 小端整数，组 g 占据 bit [5g, 5g+5)。
 *
 * 借鉴来源（零重写 / 集大成，详见 docs/ALGORITHMS.md 与 NOTICE）：
 *   - I2_S 的 AVX2 路径移植自 Microsoft BitNet.cpp（MIT）的
 *     `ggml_vec_dot_i2_i8_s`：`_mm256_maddubs_epi16`(u8×i8→i16) +
 *     `_mm256_madd_epi16`(i16→i32) 两段式加宽累加。
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

#define PHOS_ABI_VERSION 1

/*
 * 每个内核末尾的 `flags` 位开关。
 * 为什么不用 `phos_set_simd()` 全局开关：约束 #1 要求纯函数、可重入、无全局状态，
 * 全局开关会让 row-block 线程池并行出现数据竞争，也让「标量权威路径」无法与 SIMD
 * 在同一进程内并列对比。故把开关做成显式入参（由 software-architect 提出）。
 */
#define PHOS_FLAG_SCALAR 1u /* 强制走标量路径（位精确权威实现） */
#define PHOS_FLAG_AVX2 2u   /* 允许 AVX2 路径（需运行时 CPU 探测已通过） */
#define PHOS_FLAG_NEON 4u   /* 允许 ARM NEON 路径（暂未实现，见 phos_gemm.c） */

#ifdef _WIN32
#define PHOS_API __declspec(dllexport)
#else
#define PHOS_API __attribute__((visibility("default")))
#endif

/* ------------------------------------------------------------------------- */
/* 构建 / 能力探测                                                             */
/* ------------------------------------------------------------------------- */

/* ABI 版本号：bindings.py 加载时校验，不匹配即视为不可用并降级。 */
PHOS_API int phos_abi_version(void);

/* 构建信息（zig 版本 + 目标三元组 + 编译开关），用于缓存核对与 issue 复现。 */
PHOS_API const char *phos_build_info(void);

/* 运行时 CPU 探测：产物是否为 baseline 之外的 SIMD 路径做好了准备。 */
PHOS_API int phos_cpu_has_avx2(void);

/* ------------------------------------------------------------------------- */
/* 三值 mpGEMM                                                                */
/* ------------------------------------------------------------------------- */

/*
 * TL（Ternary LUT）GEMM：查表法，无乘法，内存受限场景（decode / GEMV）首选。
 * 符号 + 索引 14 项 LUT：part(26-c) == -part(c)，故只需 13 项 + 1 个全零项。
 *
 * Wp       [N, K_pad]  5-bit 打包权重（行宽 K_pad = ceil(K/24)*24，行步长
 *                      = (K_pad/24)*5 字节；**不保证 64B 对齐**）
 * xq       [K]         int8 量化激活（值域 [-127, 127]，调用方负责 pad 到 K_pad
 *                      且 pad 位填 0 —— pad 权重为 0，乘任何数都是 0）
 * wscale   [N]         权重 per-channel scale
 * sx                   激活 per-token scale
 * y        [N]         输出（fp32）
 * n0, n1               行块切分 [n0, n1)，切的是 **N 维（输出行）**，不是 K 维
 *                      —— 切 K 维会破坏累加完整性（ARCHITECTURE.md §5.1）
 * flags                PHOS_FLAG_* 位掩码
 *
 * ⚠️ 关于 f32 版本与 NumPy 的位精确性（2026-10 实测修正）：
 * 早期注释声称「f32 版本无法与 NumPy 逐位相等」——**该结论不成立**。
 * 实测（software-engineer-2，7 个变种全部 bit-exact=True）证明：只要 C 侧把
 * `acc * wscale * sx` 放在 **double 域左结合**计算、最后一次性 cast 回 float，
 * 就能与 NumPy 参考逐位相等：
 *     y[i] = (float)((double)acc * (double)wscale[i] * (double)sx)
 * 若改用 fp32 左结合累加才会丢失位精确性。整数域断言仍以 `*_i32` 为锚点。
 */
PHOS_API int phos_tl_gemm_f32(const uint8_t *Wp, const int8_t *xq, const float *wscale,
                              double sx, float *y, int N, int K_pad, int n0, int n1,
                              unsigned flags);

/*
 * I2_S（int2 + scale）GEMM：u8 x i8 点积 + 零点校正，算力受限场景（prefill）首选。
 * 数学恒等式：sum_j (xu8[j] - 128) * W[i,j] == sum_j xu8[j]*W[i,j] - 128*row_wsum[i]
 * 因此与 TL 在 int32 域逐位相等（不变量 ①）。
 *
 * Wi8      [N, K] int8 解包权重（值域 {-1,0,1}，即 loader 侧的 int8 镜像）
 * xu8      [K]     uint8 激活 = xq + 128（xq ∈ [-127,127] -> xu8 ∈ [1,255]）
 * row_wsum [N]     int32 = Σ_j W[i,j]（.phos 未序列化，由 loader 现算）
 */
PHOS_API int phos_i2s_gemm_f32(const int8_t *Wi8, const uint8_t *xu8, const int32_t *row_wsum,
                               const float *wscale, double sx, float *y, int N, int K,
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
 * 流式语义：不感知行边界，按需读字节（**不会**越界读）。
 */
PHOS_API int phos_unpack5(const uint8_t *packed, int8_t *out, int n_groups);

/*
 * 反向：把 n_groups 组三值权重打包为 5-bit 流（ceil(5*n_groups/8) 字节）。
 */
PHOS_API int phos_pack5(const int8_t *w, uint8_t *packed, int n_groups);

/* ------------------------------------------------------------------------- */
/* 逐元素 / 归约算子                                                          */
/* ------------------------------------------------------------------------- */

/*
 * RMSNorm: y[n,:] = x[n,:] / sqrt(mean(x[n,:]^2) + eps) * w[:]
 * 与 engine._rmsnorm 的数值语义一致（fp64 中间累加）。
 */
PHOS_API int phos_rmsnorm_f32(const float *x, const float *w, float *y, int N, int D, float eps,
                              unsigned flags);

/*
 * 逐行 softmax（数值稳定：先减行最大值）。
 */
PHOS_API int phos_softmax_f32(const float *x, float *y, int N, int D, unsigned flags);

/*
 * RoPE：对 [T, H, Dh] 的张量在位置 [pos0, pos0+T) 上施加旋转位置编码。
 * theta 默认 10000.0。相邻两两配对旋转（Su et al., arXiv:2104.09864），
 * 与 engine._build_rope/_apply_rope 的公式一致。
 */
PHOS_API int phos_rope_f32(const float *x, float *y, int T, int H, int Dh, int pos0, float theta,
                           unsigned flags);

#ifdef __cplusplus
}
#endif

#endif /* PHOS_H */
