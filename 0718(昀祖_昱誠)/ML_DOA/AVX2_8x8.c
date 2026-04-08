// tranpose_8_8_avx2.c

#include <stdio.h>
#include <immintrin.h>
#define V_ELEMS 8

static inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    // 将两个256位整数向量v0和v1按照指定顺序合并为两个低位和高位的256位整数向量
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb); // 低位
    *vh = _mm256_unpackhi_epi32(va, vb); // 高位
}

static inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    // 将两个256位整数向量v0和v1按照指定顺序合并为两个低位和高位的256位整数向量
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb); // 低位
    *vh = _mm256_unpackhi_epi64(va, vb); // 高位
}

static inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    // 将两个256位整数向量v0和v1按照指定顺序合并为两个低位和高位的256位整数向量
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0)); // 低位
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1)); // 高位
}

//
// Transpose_8_8
//
// 原地转置8x8整数数组
//

static void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;

    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);

    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);

    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);
}

int main(void)
{
    int buff[V_ELEMS][V_ELEMS] __attribute__ ((aligned(32)));
    int i, j;
    int k = 0;

    // 初始化 buff 数组
    for (i = 0; i < V_ELEMS; ++i)
    {
        for (j = 0; j < V_ELEMS; ++j)
        {
            buff[i][j] = k++;
        }
    }

    // 打印初始的 buff 数组
    printf("\n初始状态:\n");
    for (i = 0; i < V_ELEMS; ++i)
    {
        for (j = 0; j < V_ELEMS; ++j)
        {
            printf("%4d", buff[i][j]);
        }
        printf("\n");
    }

    // 转置操作
    Transpose_8_8((__m256i *)buff[0], (__m256i *)buff[1], (__m256i *)buff[2], (__m256i *)buff[3], (__m256i *)buff[4], (__m256i *)buff[5], (__m256i *)buff[6], (__m256i *)buff[7]);

    // 打印转置后的 buff 数组
    printf("\n转置后:\n");
    for (i = 0; i < V_ELEMS; ++i)
    {
        for (j = 0; j < V_ELEMS; ++j)
        {
            printf("%4d", buff[i][j]);
        }
        printf("\n");
    }

    // 再次进行转置操作
    Transpose_8_8((__m256i *)buff[0], (__m256i *)buff[1], (__m256i *)buff[2], (__m256i *)buff[3], (__m256i *)buff[4], (__m256i *)buff[5], (__m256i *)buff[6], (__m256i *)buff[7]);

    // 打印再次转置后的 buff 数组
    printf("\n再次转置后:\n");
    for (i = 0; i < V_ELEMS; ++i)
    {
        for (j = 0; j < V_ELEMS; ++j)
        {
            printf("%4d", buff[i][j]);
        }
        printf("\n");
    }

    return 0;
}