/*
 * phos_ops.c —— 逐元素 / 归约算子（rmsnorm / softmax / rope）。
 *
 * 这三个算子的数值语义逐一对齐 engine.py 的 numpy 参考：
 *   _rmsnorm  : ss 用 fp64 累加，inv 转 fp32 后逐元素乘
 *   softmax   : 减行最大值后 exp / sum（fp64 内部累加，保精度）
 *   _apply_rope: inv_freq 在 fp64 域计算，cos/sin 转 fp32 后与激活做 fp32 旋转
 * 验收容差：与 numpy 参考 abs < 1e-6（T4 C2 的 L3 级，非位精确）。
 */

#include "phos.h"

#include <math.h>

PHOS_API int phos_rmsnorm_f32(const float *x, const float *w, float *y, int N, int D, float eps,
                              unsigned flags) {
    int n, d;
    (void)flags;
    if (!x || !w || !y) return PHOS_ERR_NULL;
    if (N <= 0 || D <= 0) return PHOS_ERR_RANGE;
    for (n = 0; n < N; ++n) {
        const float *row = x + (size_t)n * (size_t)D;
        float *out = y + (size_t)n * (size_t)D;
        double ss = 0.0;
        float inv;
        for (d = 0; d < D; ++d) ss += (double)row[d] * (double)row[d];
        inv = (float)(1.0 / sqrt(ss / (double)D + (double)eps));
        for (d = 0; d < D; ++d) out[d] = (float)((double)row[d] * (double)inv) * w[d];
    }
    return PHOS_OK;
}

PHOS_API int phos_softmax_f32(const float *x, float *y, int N, int D, unsigned flags) {
    int n, d;
    (void)flags;
    if (!x || !y) return PHOS_ERR_NULL;
    if (N <= 0 || D <= 0) return PHOS_ERR_RANGE;
    for (n = 0; n < N; ++n) {
        const float *row = x + (size_t)n * (size_t)D;
        float *out = y + (size_t)n * (size_t)D;
        float mx = row[0];
        double sum = 0.0;
        for (d = 1; d < D; ++d)
            if (row[d] > mx) mx = row[d];
        for (d = 0; d < D; ++d) {
            double e = exp((double)row[d] - (double)mx);
            out[d] = (float)e;
            sum += e;
        }
        for (d = 0; d < D; ++d) out[d] = (float)((double)out[d] / sum);
    }
    return PHOS_OK;
}

PHOS_API int phos_rope_f32(const float *x, float *y, int T, int H, int Dh, int pos0, float theta,
                           unsigned flags) {
    int t, h, i;
    (void)flags;
    if (!x || !y) return PHOS_ERR_NULL;
    if (T <= 0 || H <= 0 || Dh <= 0 || (Dh % 2)) return PHOS_ERR_RANGE;
    for (t = 0; t < T; ++t) {
        for (h = 0; h < H; ++h) {
            const float *row = x + ((size_t)t * (size_t)H + (size_t)h) * (size_t)Dh;
            float *out = y + ((size_t)t * (size_t)H + (size_t)h) * (size_t)Dh;
            for (i = 0; i < Dh / 2; ++i) {
                double inv = 1.0 / pow((double)theta, (double)(2 * i) / (double)Dh);
                double freq = (double)(pos0 + t) * inv;
                float c = (float)cos(freq);
                float s = (float)sin(freq);
                float q1 = row[2 * i];
                float q2 = row[2 * i + 1];
                out[2 * i] = q1 * c - q2 * s;
                out[2 * i + 1] = q2 * c + q1 * s;
            }
        }
    }
    return PHOS_OK;
}
