/*
 * zig cc 编译链冒烟测试（O4：零编译器依赖的原生内核构建）
 * 参考：ziglang PyPI 包，`python -m ziglang cc` 自带 clang 前端，无需 MSVC/gcc。
 *
 * 这里刻意只使用 C99 标准库 + 位运算，验证：
 *   1) 编译器可用且能产出可执行文件；
 *   2) 整数语义（int32 累加、位域提取）与目标平台一致；
 *   3) 三值 5-bit 解 pack 的标量参考实现可编译通过（T4 内核的权威参考）。
 *
 * 编译： python -m ziglang cc -O2 -o smoke scripts/smoke_zig.c
 * 期望输出： PHOS_ZIG_OK
 */
#include <stdint.h>
#include <stdio.h>

/* 从 5-bit 打包流中取出第 g 组的 code5（见 ARCHITECTURE.md §3.6 Step 5）。 */
static int unpack_code5(const uint8_t *row, int group) {
    int byte_off = (group / 8) * 5;
    int bit_off = (group % 8) * 5;
    uint64_t v = 0;
    int i;
    for (i = 0; i < 6; ++i) {          /* 读 6 字节防止跨字节边界截断 */
        v |= (uint64_t)row[byte_off + i] << (8 * i);
    }
    return (int)((v >> bit_off) & 0x1Fu);
}

/* 由 code5 还原 3 个三值权重（符号 + 索引 14 项编码的逆运算）。 */
static void decode3(int code5, int8_t *out) {
    int index = code5 & 0x0F;
    int sign = (code5 >> 4) & 1;
    int c, k;
    if (index == 0) {
        out[0] = out[1] = out[2] = 0;
        return;
    }
    c = sign ? (27 - index) : (index - 1);
    for (k = 0; k < 3; ++k) {
        int d = c % 3;
        c /= 3;
        out[k] = (int8_t)(d - 1);
    }
    if (sign) {
        for (k = 0; k < 3; ++k) out[k] = (int8_t)(-out[k]);
    }
}

int main(void) {
    /* 手工构造：24 个权重 = 8 组 = 5 字节。第 0 组编码 (1,0,-1)。 */
    uint8_t row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_t w[3] = {0, 0, 0};
    int32_t acc = 0;
    int i;

    /* c for (1,0,-1): d = (2,1,0) -> c = 2 + 3*1 + 9*0 = 5 -> index=6, sign=0 */
    row[0] = (uint8_t)(((0u << 4) | 6u) & 0x1Fu);

    decode3(unpack_code5(row, 0), w);
    if (w[0] != 1 || w[1] != 0 || w[2] != -1) {
        printf("PHOS_ZIG_FAIL decode3 got %d %d %d\n", w[0], w[1], w[2]);
        return 1;
    }

    /* 32 位整数累加不溢出检查：512 * 127 = 65024 < INT32_MAX */
    for (i = 0; i < 512; ++i) acc += (int32_t)127;
    if (acc != 65024) {
        printf("PHOS_ZIG_FAIL acc=%d\n", (int)acc);
        return 1;
    }

    printf("PHOS_ZIG_OK\n");
    return 0;
}
