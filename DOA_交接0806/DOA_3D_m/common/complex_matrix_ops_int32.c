#define PI acos(-1)
//--------------------
#define AVX 16
//--------------------
#include <immintrin.h>
// C
#include <complex.h>
#include <assert.h>
#include "color.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

#include "complex_matrix_ops_int32.h"
#include "q_format_config.h"


//=====================================================================
void matrix_transpose_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA)
{
    // simple out-of-place transpose (temp buffer) for clarity
    int32_t *temp_re = (int32_t*)malloc(sizeof(int32_t)*rowA*colA);
    int32_t *temp_im = (int32_t*)malloc(sizeof(int32_t)*rowA*colA);
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            temp_re[j*rowA+i] = matA_re[i*colA+j];
            temp_im[j*rowA+i] = matA_im[i*colA+j];
        }
    }
    memcpy(matA_re, temp_re,sizeof(int32_t)*rowA*colA);
    memcpy(matA_im, temp_im,sizeof(int32_t)*rowA*colA);

    free(temp_re);free(temp_im);
}

void complex_matrix_conjugate_transpose_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA)
{
    // simple out-of-place conjugate transpose (temp buffer) for clarity
    int32_t *temp_re = (int32_t*)malloc(sizeof(int32_t)*rowA*colA);
    int32_t *temp_im = (int32_t*)malloc(sizeof(int32_t)*rowA*colA);
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            temp_re[j*rowA+i] = matA_re[i*colA+j];
            temp_im[j*rowA+i] = -matA_im[i*colA+j]; // conjugate
        }
    }
    memcpy(matA_re, temp_re,sizeof(int32_t)*rowA*colA);
    memcpy(matA_im, temp_im,sizeof(int32_t)*rowA*colA);

    free(temp_re);free(temp_im);
}

void complex_matrix_multiplication_int32_ijk(int32_t *matA_re,
                         int32_t *matA_im,
                         int32_t *matB_re,
                         int32_t *matB_im,
                         int32_t       *matC_re,
                         int32_t       *matC_im,
                         int rowA, int rowB, int colB)
{
    matrix_transpose_int32(matB_re, matB_im, rowB, colB);
    memset(matC_re, 0, (size_t)rowA * colB * sizeof(int32_t));
    memset(matC_im, 0, (size_t)rowA * colB * sizeof(int32_t));

    const int chunks   = rowB / AVX;
    const int rem      = rowB % AVX;
    const __mmask16 m  = (1u << rem) - 1u;
    __m512i acc_re, acc_im, a_re, a_im, b_re, b_im, p_re, p_im;
    for (int i = 0; i < rowA; ++i) {

        const int32_t *ai_re = matA_re + (size_t)i * rowB;
        const int32_t *ai_im = matA_im + (size_t)i * rowB;

        for (int j = 0; j < colB; ++j) {

            const int32_t *bj_re = matB_re + (size_t)j * rowB;
            const int32_t *bj_im = matB_im + (size_t)j * rowB;

            acc_re = _mm512_setzero_si512();
            acc_im = _mm512_setzero_si512();

            for (int k = 0; k < chunks; ++k) {

                a_re = _mm512_loadu_si512(ai_re + k * AVX);
                a_im = _mm512_loadu_si512(ai_im + k * AVX);
                b_re = _mm512_loadu_si512(bj_re + k * AVX);
                b_im = _mm512_loadu_si512(bj_im + k * AVX);

                p_re = _mm512_sub_epi32(
                                _mm512_mullo_epi32(a_re, b_re),
                                _mm512_mullo_epi32(a_im, b_im));
                p_im = _mm512_add_epi32(
                                _mm512_mullo_epi32(a_re, b_im),
                                _mm512_mullo_epi32(a_im, b_re));

                p_re = _mm512_srai_epi32(p_re, Q_SHIFT);
                p_im = _mm512_srai_epi32(p_im, Q_SHIFT);

                acc_re = _mm512_add_epi32(acc_re, p_re);
                acc_im = _mm512_add_epi32(acc_im, p_im);
            }

            if (rem) {
                a_re = _mm512_maskz_loadu_epi32(m, ai_re + chunks * AVX);
                a_im = _mm512_maskz_loadu_epi32(m, ai_im + chunks * AVX);
                b_re = _mm512_maskz_loadu_epi32(m, bj_re + chunks * AVX);
                b_im = _mm512_maskz_loadu_epi32(m, bj_im + chunks * AVX);

                p_re = _mm512_sub_epi32(
                                _mm512_mullo_epi32(a_re, b_re),
                                _mm512_mullo_epi32(a_im, b_im));
                p_im = _mm512_add_epi32(
                                _mm512_mullo_epi32(a_re, b_im),
                                _mm512_mullo_epi32(a_im, b_re));

                p_re = _mm512_srai_epi32(p_re, Q_SHIFT);
                p_im = _mm512_srai_epi32(p_im, Q_SHIFT);

                acc_re = _mm512_add_epi32(acc_re, p_re);
                acc_im = _mm512_add_epi32(acc_im, p_im);
            }

            matC_re[i * colB + j] = _mm512_reduce_add_epi32(acc_re);
            matC_im[i * colB + j] = _mm512_reduce_add_epi32(acc_im);
        }
    }
    matrix_transpose_int32(matB_re, matB_im, colB, rowB);
}

static int64_t acc_re[1000000] __attribute__((aligned(64)));
static int64_t acc_im[1000000] __attribute__((aligned(64)));
void complex_matrix_multiplication_int32(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int32_t *matC_re, int32_t *matC_im, int rowA, int rowB, int colB)
{
    // Parameter initialize
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication 
    gettimeofday(&start_multiply, NULL);                           // start
    // Temporarily store C using int64_t accumulators
    memset(acc_re, 0, rowA * colB * sizeof(int64_t)); // Initial acc_re = 0
    memset(acc_im, 0, rowA * colB * sizeof(int64_t)); // Initial acc_im = 0
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    matrix_transpose_int32(matB_re, matB_im, rowB, colB);              // Matrix transpose
    //----------------------------------------------------------
    __m512i re_A, re_B, re_C; 
    __m512i im_A, im_B, im_C;
    //----------------------------------------------------------
    int full_chunks = rowB / AVX;
    int remainder   = rowB % AVX;
    
    for (int k = 0; k < full_chunks; k++) {
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_loadu_si512(&matA_re[i * rowB + k * AVX]);
            im_A = _mm512_loadu_si512(&matA_im[i * rowB + k * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_loadu_si512(&matB_re[j * rowB + k * AVX]);
                im_B = _mm512_loadu_si512(&matB_im[j * rowB + k * AVX]);
    
                re_C = _mm512_sub_epi32(_mm512_mullo_epi32(re_A, re_B), _mm512_mullo_epi32(im_A, im_B));
                im_C = _mm512_add_epi32(_mm512_mullo_epi32(re_A, im_B), _mm512_mullo_epi32(im_A, re_B));
                int idx = i * colB + j;
                acc_re[idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                acc_im[idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
            }
        }
    }
    __mmask16 mask;
    if (remainder > 0) {
        mask = (1 << remainder) - 1;
    
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_maskz_loadu_epi32(mask, &matA_re[i * rowB + full_chunks * AVX]);
            im_A = _mm512_maskz_loadu_epi32(mask, &matA_im[i * rowB + full_chunks * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_maskz_loadu_epi32(mask, &matB_re[j * rowB + full_chunks * AVX]);
                im_B = _mm512_maskz_loadu_epi32(mask, &matB_im[j * rowB + full_chunks * AVX]);
    
                re_C = _mm512_sub_epi32(_mm512_mullo_epi32(re_A, re_B), _mm512_mullo_epi32(im_A, im_B));
                im_C = _mm512_add_epi32(_mm512_mullo_epi32(re_A, im_B), _mm512_mullo_epi32(im_A, re_B));
    
                int idx = i * colB + j;
                acc_re[idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                acc_im[idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
            }
        }
    }
    //-------------------------------------------------------------------
    matrix_transpose_int32(matB_re, matB_im, colB, rowB); // Matrix transpse -> back to origin version
    // Right shift back to int32_t
    for (int i = 0; i < rowA * colB; ++i) {
        //printf("acc_re[%d] = %ld, acc_im[%d] = %ld\n", i, acc_re[i], i, acc_im[i]);
        matC_re[i] = (int32_t)(acc_re[i] >> Q_SHIFT);
        matC_im[i] = (int32_t)(acc_im[i] >> Q_SHIFT);
        // printf("matC_re[%d] = %d, matC_im[%d] = %d\n", i, matC_re[i], i, matC_im[i]);
    }

    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}


void float_matrix_to_q_format(int32_t *out_re, int32_t *out_im, float *in_re, float *in_im, int row, int col)
{
    int scale = (1 << Q_SHIFT);
    for (int i = 0; i < row * col; ++i) {
        out_re[i] = (int32_t)(in_re[i] * scale);
        out_im[i] = (int32_t)(in_im[i] * scale);
    }
}


void q_format_to_float_matrix(float *out_re, float *out_im, int32_t *in_re, int32_t *in_im, int row, int col)
{
    float scale = 1.0f / (1 << Q_SHIFT);
    for (int i = 0; i < row * col; ++i) {
        out_re[i] = (float)in_re[i] * scale;
        out_im[i] = (float)in_im[i] * scale;
    }
}


void print_complex_matrix_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%d ", matA_re[i * colA + j]);
            printf("+ %di", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}