#include <stdint.h>
#include <immintrin.h>
#include "q_format_config.h"
#ifndef SPATIAL_SPECTRUM_INT32_H
#define SPATIAL_SPECTRUM_INT32_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;
extern int search_count;

int q_shift_DOA;
typedef struct
{
    int Rx_M_x;
    int Rx_M_y;
    int Rx_M;
    float d;
    float kc;
    int32_t *Pn_re;
    int32_t *Pn_im;
} SearchConst_int32;


static inline
void float_to_q_int32_avx512(int32_t *restrict out_re,
                             int32_t *restrict out_im,
                             const float *restrict in_re,
                             const float *restrict in_im,
                             size_t      len)
{
    /* 1. 把 2^q_shift 先 broadcast 成 512-bit 常量 */
    const __m512 scale_ps = _mm512_set1_ps((float)(1 << Q_SHIFT));

    size_t i         = 0;
    const size_t vec = len & ~15UL;     /* 向量化整段 (16 floats/iter) */

    /* 2-A. 主迴圈：每次處理 16 筆 */
    for (; i < vec; i += 16) {
        /* 2-A-1. 載入 16 個實部 / 虛部 float */
        __m512 v_re = _mm512_loadu_ps(in_re + i);
        __m512 v_im = _mm512_loadu_ps(in_im + i);

        /* 2-A-2. 乘上 2^q_shift → 固定點放大 */
        v_re = _mm512_mul_ps(v_re, scale_ps);
        v_im = _mm512_mul_ps(v_im, scale_ps);

        /* 2-A-3. _mm512_cvtps_epi32 直接做「四捨五入」轉 int32 */
        __m512i q_re = _mm512_cvtps_epi32(v_re);
        __m512i q_im = _mm512_cvtps_epi32(v_im);

        /* 2-A-4. 寫回結果 */
        _mm512_storeu_si512(out_re + i, q_re);
        _mm512_storeu_si512(out_im + i, q_im);
    }

    /* 2-B. 尾段不足 16 筆 → 用 mask 處理 */
    if (i < len) {
        const __mmask16 tail = (1u << (len - i)) - 1u;

        __m512 v_re = _mm512_maskz_loadu_ps(tail, in_re + i);
        __m512 v_im = _mm512_maskz_loadu_ps(tail, in_im + i);

        v_re = _mm512_mul_ps(v_re, scale_ps);
        v_im = _mm512_mul_ps(v_im, scale_ps);

        __m512i q_re = _mm512_cvtps_epi32(v_re);
        __m512i q_im = _mm512_cvtps_epi32(v_im);

        _mm512_mask_storeu_epi32(out_re + i, tail, q_re);
        _mm512_mask_storeu_epi32(out_im + i, tail, q_im);
    }
}

static inline
void q_int32_to_float_avx512(float   *restrict out_re,
                             float   *restrict out_im,
                             const int32_t *restrict in_re,
                             const int32_t *restrict in_im,
                             size_t      len)
{
    /* 1. 製作 1/(2^q_shift) 常量向量 */
    const float  inv_scale = 1.0f / (float)(1 << Q_SHIFT);
    const __m512 inv_ps    = _mm512_set1_ps(inv_scale);

    size_t i         = 0;
    const size_t vec = len & ~15UL;   /* 16 elements 對齊邊界 */

    /* 2-A. 主迴圈：每次處理 16 個複數元素 */
    for (; i < vec; i += 16) {
        /* 2-A-1. 讀 16 個 int32 → 轉 float */
        __m512 v_re = _mm512_cvtepi32_ps(_mm512_loadu_si512(in_re + i));
        __m512 v_im = _mm512_cvtepi32_ps(_mm512_loadu_si512(in_im + i));

        /* 2-A-2. 乘上 1/(2^q_shift) → 還原浮點值 */
        v_re = _mm512_mul_ps(v_re, inv_ps);
        v_im = _mm512_mul_ps(v_im, inv_ps);

        /* 2-A-3. 寫回結果 */
        _mm512_storeu_ps(out_re + i, v_re);
        _mm512_storeu_ps(out_im + i, v_im);
    }

    /* 2-B. 尾段 (<16 筆) 用 mask 處理 */
    if (i < len) {
        const __mmask16 tail = (1u << (len - i)) - 1u;

        __m512 v_re = _mm512_cvtepi32_ps(
                         _mm512_maskz_loadu_epi32(tail, in_re + i));
        __m512 v_im = _mm512_cvtepi32_ps(
                         _mm512_maskz_loadu_epi32(tail, in_im + i));

        v_re = _mm512_mul_ps(v_re, inv_ps);
        v_im = _mm512_mul_ps(v_im, inv_ps);

        _mm512_mask_storeu_ps(out_re + i, tail, v_re);
        _mm512_mask_storeu_ps(out_im + i, tail, v_im);
    }
}

void spatial_spectrum_value_int32(int32_t *a_vector_re, int32_t *a_vector_im, int32_t *Pn_re, int32_t *Pn_im, int M, float *result_re, float *result_im);


void calculate_spatial_spectrum_3D_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB    
);
void calculate_spatial_spectrum_3D_multiBeam_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB,
    float search_step_theta 
);
#endif