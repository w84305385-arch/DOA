// 
// 做int32(AVX512乘法) MGS-QR
// g++ -mavx512f -g -o float_int_try  float_int_try.c -Wall -Wextra -std=c++14 math_func.a
// ./float_int_try
// #define DATA_CSV_MODE 1
#define PI acos(-1)
#define BLOCK_SIZE 16
#define PRINT_RESULT 1
#define PLOT_RESULT 0
//--------------------
#define AVX 16
#define M_Antenna 64
#define ND 512
//--------------------
#include <immintrin.h>
#include "math_func.h"
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
//----------------------global variable---------------------------
static float total_test = 0;
static float total_multiply_time = 0;
static float total_pre_transpose_time = 0;
//----------------------------------------------------------------
void matrix_transpose(float *matA_re, float *matA_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));

    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = temp_im[i * colA + j];
            // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
        }
    }
    free(temp_re);
    free(temp_im);
}
void matrix_transpose_i(int16_t *matA_re, int16_t *matA_im, int16_t rowA, int16_t colA)
{
    int16_t *temp_re = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    int16_t *temp_im = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int16_t)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int16_t)));

    for (int16_t i = 0; i < rowA; ++i)
    {
        for (int16_t j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = temp_im[i * colA + j];
            //printf("(%d + %di), ", matA_re[j * rowA + i], matA_im[j * rowA + i]);
        }
    }
    free(temp_re);
    free(temp_im);
}
void matrix_transpose_i32(int *matA_re, int *matA_im, int rowA, int colA)
{
    int *temp_re = (int *)malloc(colA * rowA * sizeof(int));
    int *temp_im = (int *)malloc(colA * rowA * sizeof(int));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int)));

    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = temp_im[i * colA + j];
            //printf("(%d + %di), ", matA_re[j * rowA + i], matA_im[j * rowA + i]);
        }
    }
    free(temp_re);
    free(temp_im);
}

void print_complex_matrix(float *matA_re, float *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%.1f", matA_re[i * colA + j]);
            printf("+%.1fi", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}
void print_complex_matrix_i(int16_t *matA_re, int16_t *matA_im, int16_t rowA, int16_t colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%hd", matA_re[i * colA + j]);
            printf("+%hdi", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}
void print_complex_matrix_i32(int *matA_re, int *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%d", matA_re[i * colA + j]);
            printf("+%di", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}
void print_complex_matrix_R_xx(float *matA_re, float *matA_im, int rowA, int colA)
{
    int a=0;
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("R_xx_re[%d]=%.2f; ",a , matA_re[i * colA + j]);
            printf("R_xx_im[%d]=%.2f;",a , matA_im[i * colA + j]);
            a=a+1;
        }
        printf("\n");
    }
}

// complex matrix addition
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            // matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            // matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
            matA_re[i * colA + j] += matB_re[i * colA + j];
            matA_im[i * colA + j] += matB_im[i * colA + j];
        }
    }
}
void complex_matrix_addition_i(int16_t *matA_re, int16_t *matA_im, int16_t *matB_re, int16_t *matB_im, int16_t rowA, int16_t colA)
{
    for (int16_t i = 0; i < rowA; ++i)
    {
        for (int16_t j = 0; j < colA; ++j)
        {
            // matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            // matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
            matA_re[i * colA + j] += matB_re[i * colA + j];
            matA_im[i * colA + j] += matB_im[i * colA + j];
        }
    }
}
void complex_matrix_addition_i32(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            // matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            // matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
            matA_re[i * colA + j] += matB_re[i * colA + j];
            matA_im[i * colA + j] += matB_im[i * colA + j];
        }
    }
}

// complex matrix subtraction
void complex_matrix_subtraction(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[i * colA + j] -= matB_re[i * colA + j];
            matA_im[i * colA + j] -= matB_im[i * colA + j];
        }
    }
}
void complex_matrix_subtraction_i(int16_t *matA_re, int16_t *matA_im, int16_t *matB_re, int16_t *matB_im, int16_t rowA, int16_t colA)
{
    for (int16_t i = 0; i < rowA; ++i)
    {
        for (int16_t j = 0; j < colA; ++j)
        {
            matA_re[i * colA + j] -= matB_re[i * colA + j];
            matA_im[i * colA + j] -= matB_im[i * colA + j];
        }
    }
}
void complex_matrix_subtraction_i32(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[i * colA + j] -= matB_re[i * colA + j];
            matA_im[i * colA + j] -= matB_im[i * colA + j];
        }
    }
}

__attribute__((aligned(32))) float matC_Re_sum[30000000] = {0.0}; // Real
__attribute__((aligned(32))) float matC_Im_sum[30000000] = {0.0}; // Imaginary
//__attribute__((aligned(32))) float matA_re[100000] = {0.0};     // re_A
//__attribute__((aligned(32))) float matA_im[100000] = {0.0};     // im_A
//-------------------------------------------------------------------
//__attribute__((aligned(32))) float matB_re[100000] = {0.0}; // re_B
//__attribute__((aligned(32))) float matB_im[100000] = {0.0}; // im_B
//-------------------------------------------------------------------
__attribute__((aligned(32))) float matC_Real[30000000] = {0.0}; // re_C
__attribute__((aligned(32))) float matC_Imag[30000000] = {0.0}; // im_C

void complex_matrix_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timespec start, end;
    struct timeval time_data_start, time_data_end, time_data_diff; // time initial
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    struct timeval start_transpose, end_transpose, diff_transpose; // transpose variable
    gettimeofday(&start_multiply, NULL);                           // start

    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);                        // start
    matrix_transpose(matB_re, matB_im, rowB, colB);              // Matrix transpose
    gettimeofday(&end_transpose, NULL);                          // end
    timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    total_pre_transpose_time += diff_transpose.tv_usec;          // global variable can store transpose time
    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    memset(matC_Re_sum, 0, rowA * colB * sizeof(double));
    memset(matC_Im_sum, 0, rowA * colB * sizeof(double));
    // memset(matA_re, 0, rowA * colB * sizeof(float));
    // memset(matA_im, 0, rowA * colB * sizeof(float));
    // memset(matB_re, 0, rowA * colB * sizeof(float));
    // memset(matB_im, 0, rowA * colB * sizeof(float));
    memset(matC_Real, 0, rowA * colB * sizeof(double));
    memset(matC_Imag, 0, rowA * colB * sizeof(double));
    //----------------------------------------------------------
    //----------------------------------------------------------

    float time_used;
    int colA = rowB;

    __m512 re_A, re_B, re_C; // simd 256 for matrix real part
    __m512 im_A, im_B, im_C; // simd 256 for matrix Imaginary part
    __m512 simd_add, simd_sub;
    memset(matC_re, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Real)
    memset(matC_im, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Imaginary)
    //------------------------------------------------------------------- 32x32 take ~= 11ms
    // printf("matA_re:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     printf("%.2f, ", matA_re[a]);
    // }
    // printf("\nmatA_im:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     printf("%.2f, ", matA_im[a]);
    // }
    // printf("\n");
    // printf("matB_re:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     printf("%.2f, ", matB_re[a]);
    // }
    // printf("\nmatB_im:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     printf("%.2f, ", matB_im[a]);
    // }
    // printf("\n");
    //-------------------------------------------------------------------
    // gettimeofday(&start_multiply, NULL);
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            for (int k = 0; k < ((rowB - 1) / AVX) + 1; k++)
            {
                re_A = _mm512_loadu_ps(&matA_re[i * rowB + AVX * k]);
                im_A = _mm512_loadu_ps(&matA_im[i * rowB + AVX * k]);
                re_B = _mm512_loadu_ps(&matB_re[j * rowB + AVX * k]);
                im_B = _mm512_loadu_ps(&matB_im[j * rowB + AVX * k]);
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B)); // Re{ac-bd}
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B)); // Im{ad+bc}

                _mm512_storeu_ps(&matC_Real[(i * colB * rowB + j * rowB) + AVX * k], re_C); // store Re value
                _mm512_storeu_ps(&matC_Imag[(i * colB * rowB + j * rowB) + AVX * k], im_C); // store Im value
            }
        }
    }
    // gettimeofday(&end_multiply, NULL);
    //  timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate
    //  printf(L_PURPLE "\nElapsed AVX512 multiply time: %ld(us)\n" CLOSE, (long int)diff_multiply.tv_usec);

    //-------------------------------------------------------------------

    for (int i = 0; i < rowA * colB; i++)
    {
        // printf("※ matC_Real[%d] = %.2f, matC_Imag[%d] = %.2f\n", i * rowB, matC_Real[i * rowB], i * rowB, matC_Imag[i * rowB]);
        for (int a = 0; a < rowB; a++)
        {
            matC_Re_sum[i] += matC_Real[i * rowB + a];
            matC_Im_sum[i] += matC_Imag[i * rowB + a];
            // printf("matC_Re_sum[%d] = %.2f, ", i, matC_Re_sum[i]);
            //    printf("matC_Im_sum[%d] = %.0f, ", i * colB + a, matC_Im_sum[i]);
        }
        // printf(" \n");
    }
    // printf("\n");

    for (int i = 0; i < rowA * colB; i++)
    {
        // matC[i] = {matC_Real[i], matC_Imag[i]};
        //  matC[i] = matC_Imag[i];
        matC_re[i] = matC_Re_sum[i];
        matC_im[i] = matC_Im_sum[i];
    }
    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);           // start
    matrix_transpose(matB_re, matB_im, rowB, colB); // Matrix transpse -> back to origin version
    gettimeofday(&end_transpose, NULL);             // end
    //-------------------------------------------------------------------
    timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    total_pre_transpose_time += diff_transpose.tv_usec;          // global variable can store transpose time
    //------------------------------------------------------------
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}
void complex_matrix_multiplication_iii(int16_t *matA_re, int16_t *matA_im, int16_t *matB_re, int16_t *matB_im, int16_t *matC_re, int16_t *matC_im, int16_t rowA, int16_t rowB, int16_t colB)
{
    memset(matC_re, 0, rowA * colB * sizeof(int16_t));
    memset(matC_im, 0, rowA * colB * sizeof(int16_t));
    __m512i simd_matA_re, simd_matB_re, simd_matC_re;
    __m512i simd_matA_im, simd_matB_im, simd_matC_im;

    for (int16_t i = 0; i < rowA; ++i)
    {
        for (int16_t j = 0; j < colB; ++j)
        {
            for (int16_t k = 0; k < rowB; ++k)
            {
                matC_re[i * colB + j] += matA_re[i * rowB + k] * matB_re[k * colB + j]-matA_im[i * rowB + k] * matB_im[k * colB + j];
                matC_im[i * colB + j] += matA_re[i * rowB + k] * matB_im[k * colB + j]+matA_im[i * rowB + k] * matB_re[k * colB + j];
            }
        }
    }
}

__attribute__((aligned(32))) int matC_Re_sumi[30000000] = {0}; // Real
__attribute__((aligned(32))) int matC_Im_sumi[30000000] = {0}; // Imaginary
__attribute__((aligned(32))) int matC_Reali[30000000] = {0}; // re_C
__attribute__((aligned(32))) int matC_Imagi[30000000] = {0}; // im_C


void complex_matrix_multiplication_iii32(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int *matC_re, int *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timespec start, end;
    struct timeval time_data_start, time_data_end, time_data_diff; // time initial
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    struct timeval start_transpose, end_transpose, diff_transpose; // transpose variable
    gettimeofday(&start_multiply, NULL);                           // start

    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);                        // start
    matrix_transpose_i32(matB_re, matB_im, rowB, colB);              // Matrix transpose
    gettimeofday(&end_transpose, NULL);                          // end
    timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    total_pre_transpose_time += diff_transpose.tv_usec;          // global variable can store transpose time
    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    memset(matC_Re_sumi, 0, rowA * colB * sizeof(int));
    memset(matC_Im_sumi, 0, rowA * colB * sizeof(int));
    // memset(matA_re, 0, rowA * colB * sizeof(float));
    // memset(matA_im, 0, rowA * colB * sizeof(float));
    // memset(matB_re, 0, rowA * colB * sizeof(float));
    // memset(matB_im, 0, rowA * colB * sizeof(float));
    memset(matC_Reali, 0, rowA * colB * sizeof(int));
    memset(matC_Imagi, 0, rowA * colB * sizeof(int));
    //----------------------------------------------------------
    //----------------------------------------------------------

    float time_used;
    int colA = rowB;

    __m512i re_A, re_B, re_C; // simd 256 for matrix real part
    __m512i im_A, im_B, im_C; // simd 256 for matrix Imaginary part
    __m512i simd_add, simd_sub;
    memset(matC_re, 0, rowA * colB * sizeof(int)); // Initial matC = 0 (Real)
    memset(matC_im, 0, rowA * colB * sizeof(int)); // Initial matC = 0 (Imaginary)
    //------------------------------------------------------------------- 32x32 take ~= 11ms
    //printf("matA_re:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     printf("%d, ", matA_re[a]);
    // }
    // printf("\nmatA_im:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     printf("%d, ", matA_im[a]);
    // }
    // printf("\n");
    // printf("matB_re:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     printf("%d, ", matB_re[a]);
    // }
    // printf("\nmatB_im:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     printf("%d, ", matB_im[a]);
    // }
    // printf("\n");
    //-------------------------------------------------------------------
    // gettimeofday(&start_multiply, NULL);
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            for (int k = 0; k < ((rowB - 1) / AVX) + 1; k++)
            {
                re_A = _mm512_loadu_si512(&matA_re[i * rowB + AVX * k]);
                im_A = _mm512_loadu_si512(&matA_im[i * rowB + AVX * k]);
                re_B = _mm512_loadu_si512(&matB_re[j * rowB + AVX * k]);
                im_B = _mm512_loadu_si512(&matB_im[j * rowB + AVX * k]);
                re_C = _mm512_sub_epi32(_mm512_mullo_epi32(re_A, re_B), _mm512_mullo_epi32(im_A, im_B)); // Re{ac-bd}
                im_C = _mm512_add_epi32(_mm512_mullo_epi32(re_A, im_B), _mm512_mullo_epi32(im_A, re_B)); // Im{ad+bc}

                _mm512_storeu_si512(&matC_Reali[(i * colB * rowB + j * rowB) + AVX * k], re_C); // store Re value
                _mm512_storeu_si512(&matC_Imagi[(i * colB * rowB + j * rowB) + AVX * k], im_C); // store Im value
            }
        }
    }
    // gettimeofday(&end_multiply, NULL);
    //  timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate
    //  printf(L_PURPLE "\nElapsed AVX512 multiply time: %ld(us)\n" CLOSE, (long int)diff_multiply.tv_usec);

    //-------------------------------------------------------------------

    for (int i = 0; i < rowA * colB; i++)
    {
         //printf("※ matC_Real[%d] = %d, matC_Imag[%d] = %d\n", i * rowB, matC_Real[i * rowB], i * rowB, matC_Imag[i * rowB]);
        for (int a = 0; a < rowB; a++)
        {
            matC_Re_sumi[i] += matC_Reali[i * rowB + a];
            matC_Im_sumi[i] += matC_Imagi[i * rowB + a];
            // printf("matC_Re_sum[%d] = %d, ", i, matC_Re_sum[i]);
            //    printf("matC_Im_sum[%d] = %.0f, ", i * colB + a, matC_Im_sum[i]);
        }
        // printf(" \n");
    }
    // printf("\n");

    for (int i = 0; i < rowA * colB; i++)
    {
        // matC[i] = {matC_Real[i], matC_Imag[i]};
        //  matC[i] = matC_Imag[i];
        matC_re[i] = matC_Re_sumi[i];
        matC_im[i] = matC_Im_sumi[i];
    }
    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);           // start
    matrix_transpose_i32(matB_re, matB_im, rowB, colB); // Matrix transpse -> back to origin version
    gettimeofday(&end_transpose, NULL);             // end
    //-------------------------------------------------------------------
    timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    total_pre_transpose_time += diff_transpose.tv_usec;          // global variable can store transpose time
    //------------------------------------------------------------
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}

// get complex matrix by column
void complex_matrix_get_columns(float *matA_re, float *matA_im, float *matCol_re, float *matCol_im, int rowA, int colA, int colTarget)
{
    for (int i = 0; i < rowA; ++i)
    {
        matCol_re[i] = matA_re[i * colA + colTarget];
        matCol_im[i] = matA_im[i * colA + colTarget];
    }
}
void complex_matrix_get_columns_i(int16_t *matA_re, int16_t *matA_im, int16_t *matCol_re, int16_t *matCol_im, int16_t rowA, int16_t colA, int16_t colTarget)
{
    for (int16_t i = 0; i < rowA; ++i)
    {
        matCol_re[i] = matA_re[i * colA + colTarget];
        matCol_im[i] = matA_im[i * colA + colTarget];
    }
}
void complex_matrix_get_columns_i32(int *matA_re, int *matA_im, int *matCol_re, int *matCol_im, int rowA, int colA, int colTarget)
{
    for (int i = 0; i < rowA; ++i)
    {
        matCol_re[i] = matA_re[i * colA + colTarget];
        matCol_im[i] = matA_im[i * colA + colTarget];
    }
}
// get complex matrix by row
void complex_matrix_get_rows(float *matA_re, float *matA_im, float *matRow_re, float *matRow_im, int rowA, int colA, int rowTarget)
{
    for (int i = 0; i < colA; ++i)
    {
        matRow_re[i] = matA_re[rowTarget * colA + i];
        matRow_im[i] = matA_im[rowTarget * colA + i];
    }
}
void complex_matrix_get_rows_i(int16_t *matA_re, int16_t *matA_im, int16_t *matRow_re, int16_t *matRow_im, int16_t rowA, int16_t colA, int16_t rowTarget)
{
    for (int16_t i = 0; i < colA; ++i)
    {
        matRow_re[i] = matA_re[rowTarget * colA + i];
        matRow_im[i] = matA_im[rowTarget * colA + i];
    }
}
void complex_matrix_get_rows_i32(int *matA_re, int *matA_im, int *matRow_re, int *matRow_im, int rowA, int colA, int rowTarget)
{
    for (int i = 0; i < colA; ++i)
    {
        matRow_re[i] = matA_re[rowTarget * colA + i];
        matRow_im[i] = matA_im[rowTarget * colA + i];
    }
}

void complex_matrix_conjugate_transpose(float *matA_re, float *matA_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));

    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = -temp_im[i * colA + j];
            // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
        }
    }
    free(temp_re);
    free(temp_im);
}
void complex_matrix_conjugate_transpose_i(int16_t *matA_re, int16_t *matA_im, int16_t rowA, int16_t colA)
{
    int16_t *temp_re = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    int16_t *temp_im = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int16_t)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int16_t)));

    for (int16_t i = 0; i < rowA; ++i)
    {
        for (int16_t j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = -temp_im[i * colA + j];
            // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
        }
    }
    free(temp_re);
    free(temp_im);
}
void complex_matrix_conjugate_transpose_i32(int *matA_re, int *matA_im, int rowA, int colA)
{
    int *temp_re = (int *)malloc(colA * rowA * sizeof(int));
    int *temp_im = (int *)malloc(colA * rowA * sizeof(int));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int)));

    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA_re[j * rowA + i] = temp_re[i * colA + j];
            matA_im[j * rowA + i] = -temp_im[i * colA + j];
            // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
        }
    }
    free(temp_re);
    free(temp_im);
}

void complex_matrix_conjugate_transpose_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    complex_matrix_conjugate_transpose(temp_re, temp_im, rowA, colA);
    complex_matrix_multiplication(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);

    free(temp_re);
    free(temp_im);
}
void complex_matrix_conjugate_transpose_multiplication_iii(int16_t *matA_re, int16_t *matA_im, int16_t *matB_re, int16_t *matB_im, int16_t rowA, int16_t colA)
{
    int16_t *temp_re = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    int16_t *temp_im = (int16_t *)malloc(colA * rowA * sizeof(int16_t));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int16_t)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int16_t)));
    complex_matrix_conjugate_transpose_i(temp_re, temp_im, rowA, colA);
    //print_complex_matrix_i(matA_re, matA_im, rowA,colA );
    //print_complex_matrix_i(temp_re, temp_im, colA,rowA );
    complex_matrix_multiplication_iii(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);
    //print_complex_matrix_i(matB_re, matB_im, 1, 1);
    free(temp_re);
    free(temp_im);
}
void complex_matrix_conjugate_transpose_multiplication_iii32(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
{
    int *temp_re = (int *)malloc(colA * rowA * sizeof(int));
    int *temp_im = (int *)malloc(colA * rowA * sizeof(int));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int)));
    complex_matrix_conjugate_transpose_i32(temp_re, temp_im, rowA, colA);
    //print_complex_matrix_i(matA_re, matA_im, rowA,colA );
    //print_complex_matrix_i(temp_re, temp_im, colA,rowA );
    complex_matrix_multiplication_iii32(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);
    //print_complex_matrix_i(matB_re, matB_im, 1, 1);
    free(temp_re);
    free(temp_im);
}

void compute_Pn(float *Pn_re, float *Pn_im, float *vet_noise_re, float *vet_noise_im, int M, int len_t_theta)
{
    //---------------------------------------------------------------
    float *vet_noise_temp_re = (float *)malloc(M * sizeof(float));
    float *vet_noise_temp_im = (float *)malloc(M * sizeof(float));
    float *Pn_temp_re = (float *)malloc(M * M * sizeof(float));
    float *Pn_temp_im = (float *)malloc(M * M * sizeof(float));
    //---------------------------------------------------------------
    // print_complex_matrix(vet_noise_re, vet_noise_im, M, M - len_t_theta);
    // printf("debug vet_noise_re\n");
    // for (int i = 0; i < M * (M - len_t_theta); i++)
    // {
    //     printf("\t(%f,%f)\n", vet_noise_re[i], vet_noise_im[i]);
    // }
    for (int i = 0; i < M - len_t_theta; ++i)
    {
        complex_matrix_get_columns(vet_noise_re, vet_noise_im, vet_noise_temp_re, vet_noise_temp_im, M, M - len_t_theta, i);
        complex_matrix_conjugate_transpose_multiplication(vet_noise_temp_re, vet_noise_temp_im, Pn_temp_re, Pn_temp_im, M, 1);
        complex_matrix_addition(Pn_re, Pn_im, Pn_temp_re, Pn_temp_im, M, M);
    }
    free(vet_noise_temp_re);
    free(vet_noise_temp_im);
    free(Pn_temp_re);
    free(Pn_temp_im);
}
void compute_Pn_i(int16_t *Pn_re, int16_t *Pn_im, int16_t *vet_noise_re, int16_t *vet_noise_im, int16_t M, int16_t len_t_theta)
{
    //---------------------------------------------------------------
    int16_t *vet_noise_temp_re = (int16_t *)malloc(M * sizeof(int16_t));
    int16_t *vet_noise_temp_im = (int16_t *)malloc(M * sizeof(int16_t));
    int16_t *Pn_temp_re = (int16_t *)malloc(M * M * sizeof(int16_t));
    int16_t *Pn_temp_im = (int16_t *)malloc(M * M * sizeof(int16_t));
    //---------------------------------------------------------------
    // print_complex_matrix(vet_noise_re, vet_noise_im, M, M - len_t_theta);
    // printf("debug vet_noise_re\n");
    // for (int i = 0; i < M * (M - len_t_theta); i++)
    // {
    //     printf("\t(%f,%f)\n", vet_noise_re[i], vet_noise_im[i]);
    // }
    for (int16_t i = 0; i < M - len_t_theta; ++i)
    {
        complex_matrix_get_columns_i(vet_noise_re, vet_noise_im, vet_noise_temp_re, vet_noise_temp_im, M, M - len_t_theta, i);
        complex_matrix_conjugate_transpose_multiplication_iii(vet_noise_temp_re, vet_noise_temp_im, Pn_temp_re, Pn_temp_im, M, 1);
        complex_matrix_addition_i(Pn_re, Pn_im, Pn_temp_re, Pn_temp_im, M, M);
    }
    free(vet_noise_temp_re);
    free(vet_noise_temp_im);
    free(Pn_temp_re);
    free(Pn_temp_im);
}

void compute_S_MUSIC(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *music_Real, float *music_Imag)
{
    //---------------------------------------------------------------
    float *Pn_a_vector_temp_re = (float *)malloc(M * sizeof(float));
    float *Pn_a_vector_temp_im = (float *)malloc(M * sizeof(float));
    float *S_MUSIC_temp_re = (float *)malloc(M * sizeof(float));
    float *S_MUSIC_temp_im = (float *)malloc(M * sizeof(float));
    float real = 1.0;
    float imag = 0.0;
    float *temp_re = &real;
    float *temp_im = &imag;
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    complex_matrix_multiplication(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector_re, a_vector_im, M, 1);
    complex_matrix_multiplication(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, S_MUSIC_temp_re, S_MUSIC_temp_im, 1, M, 1);
    cpp_division2(1, 0, &S_MUSIC_temp_re[0], &S_MUSIC_temp_im[0], music_Real, music_Imag);

    //printf("music = (%.5f,%.5f)\n", *music_Real, *music_Real);
}
// QR decomposer for c code
void qr(float *A_re, float *A_im, float *Q_re, float *Q_im, float *R_re, float *R_im, int row, int col)
{
    float *Q_col_temp_re = (float *)malloc(row * 1 * sizeof(float));
    float *Q_col_temp_im = (float *)malloc(row * 1 * sizeof(float));
    //--------------------------------------------------------------
    float *Q_col_re = (float *)malloc(row * 1 * sizeof(float));
    float *Q_col_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    memset(Q_col_re, 0, row * 1 * sizeof(float));
    memset(Q_col_im, 0, row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *vector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *vector_cur_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *power_val_re = (float *)malloc(sizeof(float));
    float *power_val_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    int X1 = 64;   //原始輸入放大X1倍
    
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_re[i] = 1; // value 1 (unit matrix)
        R_re[i] = 1; // value 1 (unit matrix)
    }
    for (int i = 0; i < col; ++i)
    {
        for (int m = 0; m < row; ++m)
        {
            Q_re[m * col + i] = A_re[m * col + i];
            Q_im[m * col + i] = A_im[m * col + i];
        }
    }
    //print_complex_matrix(A_re, A_im, row, col);
    //printf("A:\n");
    //print_complex_matrix(A_re, A_im, row, col );
    //printf(YELLOW"---------\n"CLOSE);
    for (int i = 0; i < col; ++i)
    {
        //printf(YELLOW"-----i=(%d)----\n"CLOSE,i);
        //printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        //printf("一開始Q:\n");
        //print_complex_matrix(Q_re, Q_im, row, col );
        
        complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
        complex_matrix_get_columns(Q_re, Q_im, Q_col_temp_re, Q_col_temp_im, row, col, i);
        //printf("Q_col歸一前,v(%d)\n",i);
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        /*
        if(i==0){    
            for(int16_t w=0;w<col;w++ ){ //為了不讓power_value超過32768 縮小X2倍
                Q_col_temp_re[w]=Q_col_temp_re[w]/X2;
                Q_col_temp_im[w]=Q_col_temp_im[w]/X2;
            }
        }  
        else{
            for(int16_t w=0;w<col;w++ ){ //為了不讓power_value超過32768
                Q_col_temp_re[w]=Q_col_temp_re[w]/2;
                Q_col_temp_im[w]=Q_col_temp_im[w]/2;
            }
        }  
        */
        complex_matrix_conjugate_transpose(Q_col_temp_re, Q_col_temp_im, row, 1);
        //printf("Q_col_temp^H\n");
        //print_complex_matrix(Q_col_temp_re, Q_col_temp_im,1 ,row );
        //printf("Q_col_temp\n");
        //print_complex_matrix(Q_col_temp_re, Q_col_temp_im, row, 1 );
        complex_matrix_conjugate_transpose_multiplication(Q_col_temp_re, Q_col_temp_im, power_val_re, power_val_im, 1, row); //v(i)長度的平方
        //printf("power_val開根號前\n");
        //print_complex_matrix(power_val_re, power_val_im, 1, 1);
        cpp_sqrt(&power_val_re[0], &power_val_im[0]);
        //printf("power_val開根號後:放到R對角線上\n");
        //print_complex_matrix(power_val_re, power_val_im, 1, 1);
        R_re[i * col + i] = power_val_re[0];
        R_im[i * col + i] = power_val_im[0]; //給R對角線
        /*
        if(i==0){ //給R補償後的power_value，才能使R是128倍
            R_re[i * col + i] = power_val_re[0]*X2;
            R_im[i * col + i] = power_val_im[0]*X2; //給R對角線
        }
        else{
            R_re[i * col + i] = power_val_re[0]*2;
            R_im[i * col + i] = power_val_im[0]*2; //給R對角線
        }
        */
        
        //printf("除法分子放大前\n");
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        for(int w=0;w<col;w++){
            Q_col_re[w]=Q_col_re[w]*X1;
            Q_col_im[w]=Q_col_im[w]*X1;
        }
        //printf("除法分子放大後\n");
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
       
        //printf("除法前\n");
        //print_complex_matrix(Q_re, Q_im, row, col);
        for (int16_t m = 0; m < row; ++m)
        {
            Q_re[m * col + i]=Q_re[m * col + i]*X1;
            Q_im[m * col + i]=Q_im[m * col + i]*X1;
        }
        /*
        if(i==0){
            for (int m = 0; m < row; ++m)//因為除法所以Q_re要乘以X1,Q_temp是1/X2倍->power_value是1/X2倍，除完的Q會是X2倍
            {   
                Q_re[m * col + i]=Q_re[m * col + i]/2; //合併原本乘X1再除以X2補償
                Q_im[m * col + i]=Q_im[m * col + i]/2;
            }
        }
        else{
            for (int m = 0; m < row; ++m)
            {   
                Q_re[m * col + i]=Q_re[m * col + i]*32; //合併原本乘X1再除以2
                Q_im[m * col + i]=Q_im[m * col + i]*32;
            }
        }
        */
        complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
        //complex_matrix_conjugate_transpose(Q_col_re, Q_col_im, 1, row); //v^H -> v : 1*row -> row*1
        //printf("分子放大後\n");
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        //*/ 
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&Q_col_re[m], &Q_col_im[m], &power_val_re[0], &power_val_im[0]); //Q_col=q(i)=v(i)除rii : row*1除長度//i=0:q0，i=1:q1...給後續計算用
            cpp_division(&Q_re[m * col + i], &Q_im[m * col + i], &power_val_re[0], &power_val_im[0]);//只為了存一開始歸一化的v(i)除rii進Q i=0時歸一第0行，i=1時歸一第1行...
        }
        int a3=64;
        //*把Q_col放大
        for(int w=0;w<col;w++){
            Q_col_re[w]=Q_col_re[w]*a3;
            Q_col_im[w]=Q_col_im[w]*a3;
        }
        //*/
        /*
        for(int w=0;w<col;w++){ //讓Q_col從32變成128
            Q_col_re[w]=Q_col_re[w]*4;
            Q_col_im[w]=Q_col_im[w]*4;
        }
        */
        ///*
        //printf("除完的Q_col(未調整)\n");
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        
        //printf("歸一化後(調整過)q,(*a3:%d) \n",a3);
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        complex_matrix_conjugate_transpose( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_re,Q_im(%d):\n",i);
        //print_complex_matrix(Q_re, Q_im, row, col);
        //printf(BLUE"---\n"CLOSE);
        if(i<col-1)//i=0,1,2進////i=0,1,2,3,4,5,6進
        {
            int size = (col-(i+1));
            float *Q_sub_re = (float *)malloc( row* size * sizeof(float));
            float *Q_sub_im = (float *)malloc( row* size * sizeof(float));
            memset(Q_sub_re, 0, row * size * sizeof(float));
            memset(Q_sub_im, 0, row * size * sizeof(float));
            float *Q_col_proj_re = (float *)malloc(row * size * sizeof(float));
            float *Q_col_proj_im = (float *)malloc(row * size * sizeof(float));
            memset(Q_col_proj_re, 0, row * size * sizeof(float));
            memset(Q_col_proj_im, 0, row * size * sizeof(float));
            float *proj_vector_re = (float *)malloc( 1 * size *sizeof(float));
            float *proj_vector_im = (float *)malloc( 1 * size *sizeof(float));
            memset(proj_vector_re, 0, 1 * size * sizeof(float));
            memset(proj_vector_im, 0, 1 * size * sizeof(float));
            //printf("要取的Q:\n");
            //print_complex_matrix(Q_re, Q_im, row, col );   
            //printf(YELLOW "j=%d\n" CLOSE,j);
        
            for (int m = 0; m < row; ++m)
            {
                for (int j = i+1; j < col; ++j)
                {
                    Q_col_proj_re[m * (col-(i+1)) + j-(i+1)] = Q_re[m * col + j]; //i=0: 8*7//i=1: 4*2//i=2: 4*1//i=3: 4*0
                    Q_col_proj_im[m * (col-(i+1)) + j-(i+1)] = Q_im[m * col + j];
                    // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }
            //printf("q:Q_col^H\n");
            //print_complex_matrix(Q_col_re, Q_col_im, 1, row);
            //printf("v:Q_col_proj縮小前(%d):\n",i);
            //print_complex_matrix(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            
            //* 讓Q_col_proj縮小別太小 total三塊程式要改(含此塊)[備註以免漏掉]
            int a=64;
            //if(i>0){
                for(int16_t w=0;w<row*size;w++){ //乘法 放大a
                    Q_col_proj_re[w]=Q_col_proj_re[w]*a;
                    Q_col_proj_im[w]=Q_col_proj_im[w]*a;
                }
            //}
            //printf("v:Q_col_proj先放大(*a:%d):\n",a);
            //print_complex_matrix(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            //*/
           
            for(int16_t w=0;w<row*size;w++){ //乘法 縮小X1
                Q_col_proj_re[w]=Q_col_proj_re[w]/X1;
                Q_col_proj_im[w]=Q_col_proj_im[w]/X1;
            }   
            
            //printf("v:Q_col_proj:\n");
        
            //print_complex_matrix(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            complex_matrix_multiplication(Q_col_re, Q_col_im, Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector \n");
            //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1));
            
            //*配合讓Q_col_proj別太小 
            //if(i>0){
                for(int w=0;w<size;w++){
                    proj_vector_re[w]=proj_vector_re[w]/a;
                    proj_vector_im[w]=proj_vector_im[w]/a;
                }
                //printf("q*v:r:Q_col x Q_col_proj = proj_vector ,給R右上,再縮小a的 \n");
                //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1));
            //}
            //*/ 

            for (int j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re[j-(i+1)]; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im[j-(i+1)];
            }
            
            for (int16_t j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re[j-(i+1)]/a3; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im[j-(i+1)]/a3;
            }
            //print_complex_matrix(R_re, R_im, row, col );
            //*/
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose( Q_col_re, Q_col_im, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("Q_col(%d):\n",i);
            //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
            
            int a2=16;
            //* 讓proj_vector縮小別太小 total四塊程式要改(含此塊)[備註以免漏掉]
            //if(i>0){
                for(int w=0;w<col-(i+1);w++){ 
                proj_vector_re[w]=proj_vector_re[w]*a2;
                proj_vector_im[w]=proj_vector_im[w]*a2;
                }
            //}
            //*/
            //printf("proj_vector(*a2:%d)\n",a2);
            //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1)); 
            for(int w=0;w<col-(i+1);w++){ //乘法 縮小X
                proj_vector_re[w]=proj_vector_re[w]/X1;
                proj_vector_im[w]=proj_vector_im[w]/X1;
            }
            //printf("proj_vector\n");
            //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1)); 
            complex_matrix_multiplication( Q_col_re, Q_col_im, proj_vector_re, proj_vector_im, Q_sub_re, Q_sub_im, row, 1, col-(i+1));// row*col-(i+1)
            //printf("Q_sub = Q_col x proj_vector\n");
            //print_complex_matrix(Q_sub_re, Q_sub_im,  row, col-(i+1));
            
            int a33=a3*a3;
            //*配合除法後把Q_col放大
            for(int w=0;w<row*size;w++){
                Q_sub_re[w]=Q_sub_re[w]/a33;
                Q_sub_im[w]=Q_sub_im[w]/a33;
            }
            //printf("Q_sub 除a33後\n");
            //print_complex_matrix(Q_sub_re, Q_sub_im, row, col-(i+1));
            //printf("Q 減前\n");
            //print_complex_matrix(Q_re, Q_im, row, col);
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {    
                //printf(RED"進for分別減,i=%d,j=%d\n"CLOSE,i,j);
                complex_matrix_get_columns(Q_re, Q_im, vector_cur_re, vector_cur_im, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns(Q_sub_re, Q_sub_im, Q_col_re, Q_col_im, row, col-(i+1), j-(i+1));
                
                //* 配合讓proj_vector縮小別太小
                //if(i>0){
                    for(int m=0;m<col;m++){
                        vector_cur_re[m]=vector_cur_re[m]*a2;
                        vector_cur_im[m]=vector_cur_im[m]*a2;
                    }
                //}
                //*/
                
                //printf("vector_cur:\n");
                //print_complex_matrix(vector_cur_re, vector_cur_im,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix(Q_col_re, Q_col_im,  row, 1);
                complex_matrix_subtraction(vector_cur_re, vector_cur_im, Q_col_re, Q_col_im, row, 1);
                ///* 配合讓proj_vector縮小別太小
                //if(i>0){
                    for(int m=0;m<col;m++){
                    vector_cur_re[m]=vector_cur_re[m]/a2;
                    vector_cur_im[m]=vector_cur_im[m]/a2;
                    }
                //}  
                //*/
                //printf("減完的vector_cur\n");
                //print_complex_matrix(vector_cur_re, vector_cur_im,  row, 1);
                for (int m = 0; m < row; m++)
                {
                    Q_re[m * col + j] = vector_cur_re[m];
                    Q_im[m * col + j] = vector_cur_im[m];
                }
            } 
            //printf(YELLOW"---------\n"CLOSE);
        free(Q_sub_re);
        free(Q_sub_im);
        free(Q_col_proj_re);
        free(Q_col_proj_im);
        free(proj_vector_re); 
        free(proj_vector_im); 
        }   
    }
    //printf(RED "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //printf("Q最終:\n");
    //print_complex_matrix(Q_re, Q_im, row, col );   
    //printf("R最終:\n");
    //print_complex_matrix(R_re, R_im, row, col );  
    //printf("A:\n");
    //print_complex_matrix(A_re, A_im, row, col );
    //float *QxR_re = (float *)malloc(row * col * sizeof(float));
    //float *QxR_im = (float *)malloc(row * col * sizeof(float));
    //printf("Q*R:\n");
    //complex_matrix_multiplication( Q_re, Q_im, R_re, R_im, QxR_re, QxR_im, row, col, col); 
    //print_complex_matrix(QxR_re, QxR_im, row, col );
    //free(QxR_re);
    //free(QxR_im);
    free(Q_col_re);
    free(Q_col_im);
    free(vector_cur_re);
    free(vector_cur_im);
}

void hybrid_qr(int *A_re, int *A_im, int *Q_re, int *Q_im, int *R_re, int *R_im, int row, int col)
{
    int X1 = 16;   //原始輸入放大X1倍，配合Rxx的放大記得改
    //--------------------------------------------------------------
    int *Q_col_re = (int *)malloc(row * 1 * sizeof(int));
    int *Q_col_im = (int *)malloc(row * 1 * sizeof(int));
    //---------------------------------------------------------------
    float *Q_col_re_f = (float *)malloc(row * 1 * sizeof(float));
    float *Q_col_im_f = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    memset(Q_col_re, 0, row * 1 * sizeof(int));
    memset(Q_col_im, 0, row * 1 * sizeof(int));
    //---------------------------------------------------------------
    int *vector_cur_re = (int *)malloc(row * 1 * sizeof(int));
    int *vector_cur_im = (int *)malloc(row * 1 * sizeof(int));
    //---------------------------------------------------------------
    float *vector_cur_re_f = (float *)malloc(row * 1 * sizeof(float));
    float *vector_cur_im_f = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    int *Q_col_temp_re = (int *)malloc(row * 1 * sizeof(int));
    int *Q_col_temp_im = (int *)malloc(row * 1 * sizeof(int));
    //---------------------------------------------------------------
    float *Q_col_temp_re_f = (float *)malloc(row * 1 * sizeof(float));
    float *Q_col_temp_im_f = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    int *power_val_re = (int *)malloc(sizeof(int));
    int *power_val_im = (int *)malloc(sizeof(int));
    //---------------------------------------------------------------
    float *power_val_re_f = (float *)malloc(sizeof(float));
    float *power_val_im_f = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *Q_re_f = (float *)malloc(row*col*sizeof(float));
    float *Q_im_f = (float *)malloc(row*col*sizeof(float));
    //---------------------------------------------------------------
    
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_re[i] = 1; // value 1 (unit matrix)
        R_re[i] = 1; // value 1 (unit matrix)
    }
    for (int i = 0; i < col; ++i)
    {
        for (int m = 0; m < row; ++m)
        {
            Q_re[m * col + i] = A_re[m * col + i];
            Q_im[m * col + i] = A_im[m * col + i];
        }
    }
    for (int i = 0; i < col; ++i)
    {
        for (int m = 0; m < row; ++m)
        {
            Q_re_f[m * col + i] = A_re[m * col + i];
            Q_im_f[m * col + i] = A_im[m * col + i];
        }
    }
    //printf("A:\n");
    for (int i = 0; i < 1; ++i)
    {
        //printf(YELLOW"-----i=(%d)----\n"CLOSE,i);
        printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        printf("一開始Q:\n");
        //print_complex_matrix(Q_re_f, Q_im_f, row, col );
        
        complex_matrix_get_columns(Q_re_f, Q_im_f, Q_col_re_f, Q_col_im_f, row, col, i);
        complex_matrix_get_columns(Q_re_f, Q_im_f, Q_col_temp_re_f, Q_col_temp_im_f, row, col, i);
        
        complex_matrix_conjugate_transpose(Q_col_temp_re_f, Q_col_temp_im_f, row, 1);
        //printf("Q_col_temp^H\n");
        //print_complex_matrix(Q_col_temp_re_f, Q_col_temp_im_f,1 ,row );
        //printf("Q_col_temp\n");
        //print_complex_matrix(Q_col_temp_re_f, Q_col_temp_im_f, row, 1 );
        complex_matrix_conjugate_transpose_multiplication(Q_col_temp_re_f, Q_col_temp_im_f, power_val_re_f, power_val_im_f, 1, row); //v(i)長度的平方
        //printf("power_val開根號前\n");
        //print_complex_matrix(power_val_re_f, power_val_im_f, 1, 1);
        cpp_sqrt(&power_val_re_f[0], &power_val_im_f[0]);
        //printf("power_val開根號後:放到R對角線上\n");
        //print_complex_matrix(power_val_re_f, power_val_im_f, 1, 1);
        R_re[i * col + i] = power_val_re_f[0];
        R_im[i * col + i] = power_val_im_f[0];
        
        //printf("除法分子放大前\n");
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
        for(int w=0;w<col;w++){
            Q_col_re_f[w]=Q_col_re_f[w]*X1;
            Q_col_im_f[w]=Q_col_im_f[w]*X1;
        }
        
        //printf("除法分子放大後\n");
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
       
        //printf("除法分子放大前\n");
        //print_complex_matrix(Q_re_f, Q_im_f, row, col);
        for (int m = 0; m < row; ++m)
        {
            Q_re_f[m * col + i]=Q_re_f[m * col + i]*X1;
            Q_im_f[m * col + i]=Q_im_f[m * col + i]*X1;
        }
        
        //printf("除法分子放大後\n");
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
        
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&Q_col_re_f[m], &Q_col_im_f[m], &power_val_re_f[0], &power_val_im_f[0]); //Q_col=q(i)=v(i)除rii : row*1除長度//i=0:q0，i=1:q1...給後續計算用
            cpp_division(&Q_re_f[m * col + i], &Q_im_f[m * col + i], &power_val_re_f[0], &power_val_im_f[0]);//只為了存一開始歸一化的v(i)除rii進Q i=0時歸一第0行，i=1時歸一第1行...
        }
        ///*

        int a3=1;
        //*把Q_col放大
        for(int w=0;w<col;w++){
            Q_col_re_f[w]=Q_col_re_f[w]*a3;
            Q_col_im_f[w]=Q_col_im_f[w]*a3;
        }
        //*/
        printf(BLUE"歸一化後(調整過)q,(%d): a3倍 \n"CLOSE,a3);
        print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
        complex_matrix_conjugate_transpose( Q_col_re_f, Q_col_im_f, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_re,Q_im(%d):\n",i);
        //print_complex_matrix(Q_re_f, Q_im_f, row, col);
        //printf(BLUE"---\n"CLOSE);
        if(i<col-1)//i=0,1,2進////i=0,1,2,3,4,5,6進
        {
            int size = (col-(i+1));
            float *Q_sub_re_f = (float *)malloc( row* size * sizeof(float));
            float *Q_sub_im_f = (float *)malloc( row* size * sizeof(float));
            memset(Q_sub_re_f, 0, row * size * sizeof(float));
            memset(Q_sub_im_f, 0, row * size * sizeof(float));
            float *Q_col_proj_re_f = (float *)malloc(row * size * sizeof(float));
            float *Q_col_proj_im_f = (float *)malloc(row * size * sizeof(float));
            memset(Q_col_proj_re_f, 0, row * size * sizeof(float));
            memset(Q_col_proj_im_f, 0, row * size * sizeof(float));
            float *proj_vector_re_f = (float *)malloc( 1 * size *sizeof(float));
            float *proj_vector_im_f = (float *)malloc( 1 * size *sizeof(float));
            memset(proj_vector_re_f, 0, 1 * size * sizeof(float));
            memset(proj_vector_im_f, 0, 1 * size * sizeof(float));
            //printf("要取的Q:\n");
            //print_complex_matrix(Q_re, Q_im, row, col );   
            //printf(YELLOW "j=%d\n" CLOSE,j);
        
            for (int m = 0; m < row; ++m)
            {
                for (int j = i+1; j < col; ++j)
                {
                    Q_col_proj_re_f[m * (col-(i+1)) + j-(i+1)] = Q_re_f[m * col + j]; //i=0: 8*7//i=1: 4*2//i=2: 4*1//i=3: 4*0
                    Q_col_proj_im_f[m * (col-(i+1)) + j-(i+1)] = Q_im_f[m * col + j];
                    // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }
            //printf("q:Q_col^H\n");
            //print_complex_matrix(Q_col_re_f, Q_col_im_f, 1, row);
            printf(BLUE"v:Q_col_proj\n"CLOSE);
            print_complex_matrix(Q_col_proj_re_f, Q_col_proj_im_f, row , col-(i+1));
            int a=1;
            //if(i>0){
                for(int w=0;w<row*size;w++){ //乘法 放大a
                    Q_col_proj_re_f[w]=Q_col_proj_re_f[w]*a;
                    Q_col_proj_im_f[w]=Q_col_proj_im_f[w]*a;
                }
            //}
            printf(BLUE"v:Q_col_proj先放大(xa)\n"CLOSE);
            print_complex_matrix(Q_col_proj_re_f, Q_col_proj_im_f, row , col-(i+1));

            for(int w=0;w<row*size;w++){ //乘法 縮小X1
                Q_col_proj_re_f[w]=Q_col_proj_re_f[w]/X1;
                Q_col_proj_im_f[w]=Q_col_proj_im_f[w]/X1;
            }   
            
            printf(BLUE"v:Q_col_proj先放大(/X1)\n"CLOSE);
            print_complex_matrix(Q_col_proj_re_f, Q_col_proj_im_f, row , col-(i+1));
            complex_matrix_multiplication(Q_col_re_f, Q_col_im_f, Q_col_proj_re_f, Q_col_proj_im_f, proj_vector_re_f, proj_vector_im_f, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            
            printf(BLUE"q*v:r:Q_col x Q_col_proj = proj_vector \n"CLOSE);
            print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1));
            
            for(int w=0;w<size;w++){
                proj_vector_re_f[w]=proj_vector_re_f[w]/a;
                proj_vector_im_f[w]=proj_vector_im_f[w]/a;
            }
            printf(BLUE" proj_vector縮小a的 \n"CLOSE);
            print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1));

            for (int j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re_f[j-(i+1)]/a3; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im_f[j-(i+1)]/a3;
            }
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose( Q_col_re_f, Q_col_im_f, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("q(%d):\n",i);
            //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
            int a2=1;
            //* 讓proj_vector縮小別太小 total四塊程式要改(含此塊)[備註以免漏掉]
            for(int w=0;w<col-(i+1);w++){ 
                proj_vector_re_f[w]=proj_vector_re_f[w]*a2;
                proj_vector_im_f[w]=proj_vector_im_f[w]*a2;
            }
            //*/
            
            printf(BLUE"q*v:r:proj_vector放大(*a2:%d)\n"CLOSE,a2);
            print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1)); 
            for(int w=0;w<col-(i+1);w++){ //乘法 縮小X
                proj_vector_re_f[w]=proj_vector_re_f[w]/X1;
                proj_vector_im_f[w]=proj_vector_im_f[w]/X1;
            }
            //printf("q*v:r:proj_vector\n");
            printf(BLUE"Q_col 最終:\n"CLOSE);
            print_complex_matrix( Q_col_re_f, Q_col_im_f, row, 1);
            printf(BLUE"q*v:r:proj_vector 最終:\n"CLOSE);
            print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1)); 
            complex_matrix_multiplication( Q_col_re_f, Q_col_im_f, proj_vector_re_f, proj_vector_im_f, Q_sub_re_f, Q_sub_im_f, row, 1, col-(i+1));// row*col-(i+1)
            printf(BLUE"Q_sub=多個r*q = Q_col x proj_vector\n"CLOSE);
            print_complex_matrix(Q_sub_re_f, Q_sub_im_f,  row, col-(i+1));
            int a33=a3*a3;
            //*配合除法後把Q_col放大
            for(int w=0;w<row*size;w++){
                Q_sub_re_f[w]=Q_sub_re_f[w]/a33;
                Q_sub_im_f[w]=Q_sub_im_f[w]/a33;
            }
            //*/
            printf(BLUE"最終Q_sub:\n"CLOSE);
            print_complex_matrix(Q_sub_re_f, Q_sub_im_f,  row, col-(i+1));
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {    
                //printf(RED"進for分別減,i=%d,j=%d\n"CLOSE,i,j);
                complex_matrix_get_columns(Q_re_f, Q_im_f, vector_cur_re_f, vector_cur_im_f, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns(Q_sub_re_f, Q_sub_im_f, Q_col_re_f, Q_col_im_f, row, col-(i+1), j-(i+1));
                
                for(int m=0;m<col;m++){
                    vector_cur_re_f[m]=vector_cur_re_f[m]*a2;
                    vector_cur_im_f[m]=vector_cur_im_f[m]*a2;
                }
                //printf("調整過的vector_cur減Q_col\n");
                //printf("vector_cur:\n");
                //print_complex_matrix(vector_cur_re_f, vector_cur_im_f,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix(Q_col_re_f, Q_col_im_f,  row, 1);
                complex_matrix_subtraction(vector_cur_re_f, vector_cur_im_f, Q_col_re_f, Q_col_im_f, row, 1);
                for(int m=0;m<col;m++){
                    vector_cur_re_f[m]=vector_cur_re_f[m]/a2;
                    vector_cur_im_f[m]=vector_cur_im_f[m]/a2;
                }
                //printf("減完的vector_cur\n");
                //print_complex_matrix(vector_cur_re_f, vector_cur_im_f,  row, 1);
                for (int m = 0; m < row; m++)
                {
                    Q_re_f[m * col + j] = vector_cur_re_f[m];
                    Q_im_f[m * col + j] = vector_cur_im_f[m];
                }
            } 
            for (int m = 0; m < row*col; m++)
            {
                Q_re[m] = Q_re_f[m];
                Q_im[m] = Q_im_f[m];
            }
            //printf(YELLOW"---------\n"CLOSE);
        free(Q_sub_re_f);
        free(Q_sub_im_f);
        free(Q_col_proj_re_f);
        free(Q_col_proj_im_f);
        free(proj_vector_re_f); 
        free(proj_vector_im_f); 
        }   
    }
    //print_complex_matrix_i(A_re, A_im, row, col ); //
    //printf(YELLOW"---------\n"CLOSE);
    for (int i = 1; i < col; ++i)
    {
        //printf(YELLOW"-----i=(%hd)----\n"CLOSE,i);
        //printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        //printf("一開始Q:\n");
        //print_complex_matrix_i32(Q_re, Q_im, row, col );
        complex_matrix_get_columns_i32(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i); //Q給Q_col相當於 v(i) : row*1
        complex_matrix_get_columns_i32(Q_re, Q_im, Q_col_temp_re, Q_col_temp_im, row, col, i);
        for(int w=0;w<col;w++){
            Q_col_temp_re_f[w]=Q_col_temp_re[w];
            Q_col_temp_im_f[w]=Q_col_temp_im[w];
            //Q_col_re_f[w]=Q_col_re[w];
            //Q_col_im_f[w]=Q_col_im[w];
        }
        //printf("Q_col_temp\n");
        //print_complex_matrix(Q_col_temp_re_f, Q_col_temp_im_f, row, 1 );
        complex_matrix_conjugate_transpose(Q_col_temp_re_f, Q_col_temp_im_f, row, 1);
        //printf("Q_col_temp^H\n");
        //print_complex_matrix(Q_col_temp_re_f, Q_col_temp_im_f,1 ,row );
        complex_matrix_conjugate_transpose_multiplication(Q_col_temp_re_f, Q_col_temp_im_f, power_val_re_f, power_val_im_f, 1, row); //v(i)長度的平方
        //printf("power_val開根號前\n");
        //print_complex_matrix(power_val_re_f, power_val_im_f, 1, 1);
        cpp_sqrt(&power_val_re_f[0], &power_val_im_f[0]);
        //printf("power_val開根號後:放到R對角線上\n");
        //print_complex_matrix(power_val_re_f, power_val_im_f, 1, 1);
        
        R_re[i * col + i] = power_val_re_f[0];
        R_im[i * col + i] = power_val_im_f[0]; //給R對角線
        
        //printf("除法放大前\n");
        //print_complex_matrix_i(Q_re, Q_im, row, col);
        for (int m = 0; m < row; ++m)
        {
            Q_re_f[m * col + i]=Q_re[m * col + i]*X1;
            Q_im_f[m * col + i]=Q_im[m * col + i]*X1;
        }
        //printf("除法放大後\n");
        //print_complex_matrix(Q_re_f, Q_im_f, row, col);
        complex_matrix_get_columns(Q_re_f, Q_im_f, Q_col_re_f, Q_col_im_f, row, col, i);
        //printf("除法分子放大後\n");
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&Q_col_re_f[m], &Q_col_im_f[m], &power_val_re_f[0], &power_val_im_f[0]); //Q_col=q(i)=v(i)除rii : row*1除長度//i=0:q0，i=1:q1...給後續計算用
            cpp_division(&Q_re_f[m * col + i], &Q_im_f[m * col + i], &power_val_re_f[0], &power_val_im_f[0]);//只為了存一開始歸一化的v(i)除rii進Q i=0時歸一第0行，i=1時歸一第1行...
            //cpp_division(&Q_col_re_f[m], &Q_col_im_f[m], &power_val_re_f[0], &power_val_im_f[0]);
        }
        //printf("除法後\n");
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
        //printf("除法後Q\n");
        //print_complex_matrix(Q_re_f, Q_im_f, row, col);
        for(int w=0;w<row*col;w++){
            Q_re[w]=Q_re_f[w];
            Q_im[w]=Q_im_f[w];
        }
        //
        
        int a3=1;
        //*把Q_col放大
        for(int w=0;w<col;w++){
            Q_col_re[w]=Q_col_re_f[w]*a3;
            Q_col_im[w]=Q_col_im_f[w]*a3;
        }
        //*/
        complex_matrix_conjugate_transpose_i32( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_col(%hd)*a3\n",i);
        //print_complex_matrix_i32(Q_col_re, Q_col_im, 1, row);
        //printf(BLUE"---\n"CLOSE);
        
        if(i<col-1)
        {
            int size = (col-(i+1));
            int *Q_sub_re = (int *)malloc( row* size * sizeof(int));
            int *Q_sub_im = (int *)malloc( row* size * sizeof(int));
            memset(Q_sub_re, 0, row * size * sizeof(int));
            memset(Q_sub_im, 0, row * size * sizeof(int));
            int *Q_col_proj_re = (int *)malloc(row * size * sizeof(int));
            int *Q_col_proj_im = (int *)malloc(row * size * sizeof(int));
            memset(Q_col_proj_re, 0, row * size * sizeof(int));
            memset(Q_col_proj_im, 0, row * size * sizeof(int));
            int *proj_vector_re = (int *)malloc( 1 * size *sizeof(int));
            int *proj_vector_im = (int *)malloc( 1 * size *sizeof(int));
            memset(proj_vector_re, 0, 1 * size * sizeof(int));
            memset(proj_vector_im, 0, 1 * size * sizeof(int));
            //printf("要取的Q:\n");
            //print_complex_matrix_i32(Q_re, Q_im, row, col );   
            //printf(YELLOW "j=%d\n" CLOSE,j);
        
            for (int m = 0; m < row; ++m)
            {
                for (int j = i+1; j < col; ++j)
                {
                    Q_col_proj_re[m * (col-(i+1)) + j-(i+1)] = Q_re[m * col + j]; //i=0: 8*7//i=1: 4*2//i=2: 4*1//i=3: 4*0 //v
                    Q_col_proj_im[m * (col-(i+1)) + j-(i+1)] = Q_im[m * col + j];
                    // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }
            //printf("Q\n");
            //print_complex_matrix_i32(Q_re, Q_im, row, col);
            //printf("q:Q_col^H\n");
            //print_complex_matrix_i32(Q_col_re, Q_col_im, 1, row);
            //printf("v:Q_col_proj\n");
            //print_complex_matrix_i32(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            
            //* 讓Q_col_proj縮小別太小 total三塊程式要改(含此塊)[備註以免漏掉]
            int a=1;
            //if(i>0){
                for(int w=0;w<row*size;w++){ //乘法 放大a
                    Q_col_proj_re[w]=Q_col_proj_re[w]*a;
                    Q_col_proj_im[w]=Q_col_proj_im[w]*a;
                }
            //}
            //printf("Q_col_proj先放大(xa):\n");
            //print_complex_matrix_i32(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            //*/

            for(int w=0;w<row*size;w++){ //乘法 縮小X
                Q_col_proj_re[w]=Q_col_proj_re[w]/X1;
                Q_col_proj_im[w]=Q_col_proj_im[w]/X1;
            } 
            //printf("v:Q_col_proj\n");
            //print_complex_matrix_i32(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            complex_matrix_multiplication_iii32(Q_col_re, Q_col_im, Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector\n");
            //print_complex_matrix_i32(proj_vector_re, proj_vector_im, 1, col-(i+1));

            //*配合讓Q_col_proj別太小 

            for(int w=0;w<size;w++){
                proj_vector_re[w]=proj_vector_re[w]/a;
                proj_vector_im[w]=proj_vector_im[w]/a;
            }
            //printf(" proj_vector縮小a的 \n");
            //print_complex_matrix_i32(proj_vector_re, proj_vector_im, 1, col-(i+1));
            //*/ 
            
            //*配合把Q_col放大
            for (int j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re[j-(i+1)]/a3; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im[j-(i+1)]/a3;
            }
            //*/
            
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose_i32( Q_col_re, Q_col_im, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("Q_col(%hd):\n",i);
            //print_complex_matrix_i(Q_col_re, Q_col_im, row, 1);
            
            int a2=1;
            //* 讓proj_vector縮小別太小 total四塊程式要改(含此塊)[備註以免漏掉]
            for(int w=0;w<col-(i+1);w++){ 
                proj_vector_re[w]=proj_vector_re[w]*a2;
                proj_vector_im[w]=proj_vector_im[w]*a2;
            }
            //*/
            //printf("q*v:r:proj_vector放大(*a2:%d)\n",a2);
            //print_complex_matrix_i32(proj_vector_re, proj_vector_im, 1, col-(i+1));
            for(int w=0;w<col-(i+1);w++){ //乘法 縮小X
                proj_vector_re[w]=proj_vector_re[w]/X1;
                proj_vector_im[w]=proj_vector_im[w]/X1;
            }
            //printf("q*v:r:proj_vector 最終:\n");
            //print_complex_matrix_i32(proj_vector_re, proj_vector_im, 1, col-(i+1));
            complex_matrix_multiplication_iii32( Q_col_re, Q_col_im, proj_vector_re, proj_vector_im, Q_sub_re, Q_sub_im, row, 1, col-(i+1));// row*col-(i+1)
            //printf("Q_sub=多個r*q = Q_col x proj_vector\n");
            //print_complex_matrix_i32(Q_sub_re, Q_sub_im,  row, col-(i+1));
            int a33=a3*a3;
            //*配合除法後把Q_col放大
            for(int w=0;w<row*size;w++){
                Q_sub_re[w]=Q_sub_re[w]/a33;
                Q_sub_im[w]=Q_sub_im[w]/a33;
            }
            //*/
            //printf("最終Q_sub:\n");
            //print_complex_matrix_i32(Q_sub_re, Q_sub_im,  row, col-(i+1));
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {   
                //printf(RED"進for分別減,i=%hd,j=%hd\n"CLOSE,i,j);
                complex_matrix_get_columns_i32(Q_re, Q_im, vector_cur_re, vector_cur_im, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns_i32(Q_sub_re, Q_sub_im, Q_col_re, Q_col_im, row, col-(i+1), j-(i+1));
                //printf("vector_cur放大前:\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                //* 配合讓proj_vector縮小別太小
                
                for(int m=0;m<col;m++){
                    vector_cur_re[m]=vector_cur_re[m]*a2;
                    vector_cur_im[m]=vector_cur_im[m]*a2;
                }
                //*/

                //printf("調整過\n");
                //printf("vector_cur:\n");
                //print_complex_matrix_i32(vector_cur_re, vector_cur_im,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix_i(Q_col_re, Q_col_im,  row, 1);
                complex_matrix_subtraction_i32(vector_cur_re, vector_cur_im, Q_col_re, Q_col_im, row, 1);

                //printf("減完的vector_cur減完\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                //* 配合讓proj_vector縮小別太小
                for(int m=0;m<col;m++){
                    vector_cur_re[m]=vector_cur_re[m]/a2;
                    vector_cur_im[m]=vector_cur_im[m]/a2;
                } 
                //*/

                //printf("減完的vector_cur減完&除完a2\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                for (int m = 0; m < row; m++)
                {
                    Q_re[m * col + j] = vector_cur_re[m];
                    Q_im[m * col + j] = vector_cur_im[m];
                }    
            } 
            for(int w=0;w<row*col;w++){
                Q_re_f[w]=Q_re[w];
                Q_im_f[w]=Q_im[w];
            }
        free(Q_sub_re);
        free(Q_sub_im);
        free(Q_col_proj_re);
        free(Q_col_proj_im);
        free(proj_vector_re); 
        free(proj_vector_im); 
        }   
        //printf(BLUE"Q:\n"CLOSE);
        //print_complex_matrix_i(Q_re, Q_im, row, col );
    }
    //printf(RED "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    free(Q_col_re);
    free(Q_col_im);
    free(vector_cur_re);
    free(vector_cur_im);
    
}

// compute eigen upper triangular
void eigen_upper_triangular(float *A_re, float *A_im, float *eigenvalue_re, float *eigenvalue_im, float *eigenvector_re, float *eigenvector_im, int row, int col)
{
    //---------------------------------------------------------------
    float *vector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *vector_cur_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *eigen_element_cur_re = (float *)malloc(sizeof(float));
    float *eigen_element_cur_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *vector_cur_temp_re = (float *)malloc(sizeof(float));
    float *vector_cur_temp_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *A_col_re = (float *)malloc(1 * col * sizeof(float));
    float *A_col_im = (float *)malloc(1 * col * sizeof(float));
    //---------------------------------------------------------------
    float diff_eigen_value_re = 0;
    float diff_eigen_value_im = 0;
    //---------------------------------------------------------------
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
            {
                A_re[i * col + j] = 0;
                A_im[i * col + j] = 0;
            }
            if (i == j)
            {
                eigenvalue_re[i * col + j] = A_re[i * col + j];
                eigenvalue_im[i * col + j] = A_im[i * col + j];

                eigenvector_re[i * col + j] = 1;
                // printf(PURPLE "eigenvalue[%d] = %.2f\n" CLOSE, i * col + j, eigenvalue[i * col + j]);
            }
        }
    }
    for (int i = 0; i < col; ++i)
    {
        complex_matrix_get_columns(eigenvector_re, eigenvector_im, vector_cur_re, vector_cur_im, row, col, i);

        for (int j = i - 1; j > -1; --j)
        {
            diff_eigen_value_re = eigenvalue_re[i * col + i] - eigenvalue_re[j * col + j];
            diff_eigen_value_im = eigenvalue_im[i * col + i] - eigenvalue_im[j * col + j];
            if (diff_eigen_value_re < 1e-8)
            {
                eigen_element_cur_re[0] = 0;
                eigen_element_cur_im[0] = 0;
            }
            else
            {
                complex_matrix_get_rows(A_re, A_im, A_col_re, A_col_im, row, col, j);
                complex_matrix_multiplication(A_col_re, A_col_im, vector_cur_re, vector_cur_im, eigen_element_cur_re, eigen_element_cur_im, 1, row, 1);
                cpp_division(&eigen_element_cur_re[0], &eigen_element_cur_im[0], &diff_eigen_value_re, &diff_eigen_value_im);
            }
            vector_cur_re[j] = eigen_element_cur_re[0];
            vector_cur_im[j] = eigen_element_cur_im[0];
        }
        complex_matrix_conjugate_transpose(vector_cur_re, vector_cur_im, row, 1);
        complex_matrix_conjugate_transpose_multiplication(vector_cur_re, vector_cur_im, vector_cur_temp_re, vector_cur_temp_im, 1, row);
        cpp_sqrt(&vector_cur_temp_re[0], &vector_cur_temp_im[0]); // vector_cur_temp[0] = sqrt(vector_cur_temp[0]);
        complex_matrix_conjugate_transpose(vector_cur_re, vector_cur_im, 1, row);
        // Complex Division
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&vector_cur_re[m], &vector_cur_im[m], &vector_cur_temp_re[0], &vector_cur_temp_im[0]);
            eigenvector_re[m * col + i] = vector_cur_re[m];
            eigenvector_im[m * col + i] = vector_cur_im[m];
            // printf(L_BLUE "eigenvector[%d] = %.2f\n" CLOSE, m * col + i, eigenvector_re[m * col + i]);
        }
    }
    free(vector_cur_re);
    free(vector_cur_im);
    free(eigen_element_cur_re);
    free(eigen_element_cur_im);
    free(vector_cur_temp_re);
    free(vector_cur_temp_im);
    free(A_col_re);
    free(A_col_im);
}

// compute complex eigenvector and eigenvalue for c code
void eigen(float *A_re, float *A_im, float *Ve_re, float *Ve_im, float *De_re, float *De_im, int row, int col, int iter)
{
    struct timeval start_QR, end_QR, diff_QR;
    float time_QR = 0.0;
    float *Q_re = (float *)calloc(row * col, sizeof(float));
    float *Q_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *R_re = (float *)calloc(row * col, sizeof(float));
    float *R_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *Q_temp_re = (float *)calloc(row * col, sizeof(float));
    float *Q_temp_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *Q_temp_clone_re = (float *)calloc(row * col, sizeof(float));
    float *Q_temp_clone_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_temp_re[i] = 1;
    }
    gettimeofday(&start_QR, NULL);
    for (int i = 0; i < iter; ++i)
    {
        //------------------------------Before QR------------------------------------------
        // printf(YELLOW "\n----------------Before QR-------------------\n" CLOSE);
        // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        // printf("Q = \t\n");
        // print_complex_matrix(Q_re, Q_im, row, col);
        // printf("R = \t\n");
        // print_complex_matrix(R_re, R_im, row, col);
        //-------------------------------After QR--------------------- --------------------

        qr(A_re, A_im, Q_re, Q_im, R_re, R_im, row, col);

        // printf(YELLOW "\n----------------After QR-------------------\n" CLOSE);
        // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        // printf("Q = \t\n");
        // print_complex_matrix(Q_re, Q_im, row, col);
        // printf("R = \t\n");
        // print_complex_matrix(R_re, R_im, row, col);
        //------------------------------------------------------------------------
        complex_matrix_multiplication(R_re, R_im, Q_re, Q_im, A_re, A_im, row, row, col);
        complex_matrix_multiplication(Q_temp_re, Q_temp_im, Q_re, Q_im, Q_temp_clone_re, Q_temp_clone_im, row, row, col);
        //---------------------------------------------------------------
        memcpy(Q_temp_re, Q_temp_clone_re, row * col * sizeof(float));
        memcpy(Q_temp_im, Q_temp_clone_im, row * col * sizeof(float));
    }
    gettimeofday(&end_QR, NULL);
    timersub(&end_QR, &start_QR, &diff_QR);
    time_QR = diff_QR.tv_usec;
    printf(CYAN "Elapsed QR :\t\t%.3f(ms), Iteration = %d\n" CLOSE, time_QR / 1000, iter);

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
            {
                A_re[i * col + j] = 0;
                A_im[i * col + j] = 0;
            }
        }
    }
    //---------------------------------------------------------------
    float *YY0_re = (float *)calloc(row * col, sizeof(float));
    float *YY0_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *XX0_re = (float *)calloc(row * col, sizeof(float));
    float *XX0_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    eigen_upper_triangular(A_re, A_im, YY0_re, YY0_im, XX0_re, XX0_im, row, col);
    memcpy(De_re, YY0_re, row * col * sizeof(float));
    memcpy(De_im, YY0_im, row * col * sizeof(float));
    complex_matrix_multiplication(Q_temp_re, Q_temp_im, XX0_re, XX0_im, Ve_re, Ve_im, row, row, col);

    free(Q_re);
    free(Q_im);
    free(R_re);
    free(R_im);
    free(Q_temp_re);
    free(Q_temp_im);
    free(Q_temp_clone_re);
    free(Q_temp_clone_im);
    free(YY0_re);
    free(YY0_im);
    free(XX0_re);
    free(XX0_im);
}

void hybrid_eigen(float *A_re, float *A_im, float *Ve_re, float *Ve_im, float *De_re, float *De_im, int row, int col, int iter)
{
    struct timeval start_hybrid_QR, end_hybrid_QR, diff_hybrid_QR;
    float time_hybrid_QR = 0.0;
    float *Q_re = (float *)calloc(row * col, sizeof(float));
    float *Q_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *R_re = (float *)calloc(row * col, sizeof(float));
    float *R_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *Q_temp_re = (float *)calloc(row * col, sizeof(float));
    float *Q_temp_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *Q_temp_clone_re = (float *)calloc(row * col, sizeof(float));
    float *Q_temp_clone_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    int *A_re_i = (int *)malloc(row * col* sizeof(int));
    int *A_im_i = (int *)malloc(row * col* sizeof(int));
    //---------------------------------------------------------------
    int *Q_re_i = (int *)calloc(row * col, sizeof(int));
    int *Q_im_i = (int *)calloc(row * col, sizeof(int));
    //---------------------------------------------------------------
    int *R_re_i = (int *)calloc(row * col, sizeof(int));
    int *R_im_i = (int *)calloc(row * col, sizeof(int));
    //---------------------------------------------------------------
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_temp_re[i] = 1;
    }
    
    gettimeofday(&start_hybrid_QR, NULL);
    
    for(int w=0;w<row*col;w++){
        A_re_i[w]=round(A_re[w]*256);
        A_im_i[w]=round(A_im[w]*256);
    }
    
    for (int i = 0; i < iter; ++i)
    {
        //------------------------------Before QR------------------------------------------
        // printf(YELLOW "\n----------------Before QR-------------------\n" CLOSE);
        // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        // printf("Q = \t\n");
        // print_complex_matrix(Q_re, Q_im, row, col);
        // printf("R = \t\n");
        // print_complex_matrix(R_re, R_im, row, col);
        //-------------------------------After QR--------------------- --------------------
        //printf("A_re\n");
        //print_complex_matrix(A_re,A_im,row,col);
        //printf("A_re16\n");
        //print_complex_matrix_i(A_re16,A_im16,row,col);
        hybrid_qr(A_re_i, A_im_i, Q_re_i, Q_im_i, R_re_i, R_im_i, row, col);
        for(int w=0;w<row*col;w++){
            A_re[w]=A_re_i[w];
            A_re[w]=A_im_i[w];
            Q_re[w]=Q_re_i[w];
            Q_im[w]=Q_im_i[w];
            R_re[w]=R_re_i[w];
            R_im[w]=R_im_i[w];
        }
        // printf(YELLOW "\n----------------After QR-------------------\n" CLOSE);
        // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        //printf("Q = \t\n");
        //print_complex_matrix(Q_re, Q_im, row, col);
        //printf("R = \t\n");
        //print_complex_matrix(R_re, R_im, row, col);
        //------------------------------------------------------------------------
        complex_matrix_multiplication(R_re, R_im, Q_re, Q_im, A_re, A_im, row, row, col);
        complex_matrix_multiplication(Q_temp_re, Q_temp_im, Q_re, Q_im, Q_temp_clone_re, Q_temp_clone_im, row, row, col);
        //---------------------------------------------------------------
        memcpy(Q_temp_re, Q_temp_clone_re, row * col * sizeof(float));
        memcpy(Q_temp_im, Q_temp_clone_im, row * col * sizeof(float));
    }
    gettimeofday(&end_hybrid_QR, NULL);
    timersub(&end_hybrid_QR, &start_hybrid_QR, &diff_hybrid_QR);
    time_hybrid_QR = diff_hybrid_QR.tv_usec;
    printf(CYAN "Elapsed hybrid QR :\t\t%.3f(ms), Iteration = %d\n" CLOSE, time_hybrid_QR / 1000, iter);

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
            {
                A_re[i * col + j] = 0;
                A_im[i * col + j] = 0;
            }
        }
    }
    //---------------------------------------------------------------
    float *YY0_re = (float *)calloc(row * col, sizeof(float));
    float *YY0_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    float *XX0_re = (float *)calloc(row * col, sizeof(float));
    float *XX0_im = (float *)calloc(row * col, sizeof(float));
    //---------------------------------------------------------------
    eigen_upper_triangular(A_re, A_im, YY0_re, YY0_im, XX0_re, XX0_im, row, col);
    memcpy(De_re, YY0_re, row * col * sizeof(float));
    memcpy(De_im, YY0_im, row * col * sizeof(float));
    complex_matrix_multiplication(Q_temp_re, Q_temp_im, XX0_re, XX0_im, Ve_re, Ve_im, row, row, col);

    free(Q_re);
    free(Q_im);
    free(R_re);
    free(R_im);
    free(Q_temp_re);
    free(Q_temp_im);
    free(Q_temp_clone_re);
    free(Q_temp_clone_im);
    free(YY0_re);
    free(YY0_im);
    free(XX0_re);
    free(XX0_im);
}






//---

void MUSIC_DOA_2A_CPU_test(int16_t M, int16_t qr_iter, float *angle, int16_t number_angle, float *result, int16_t SNR)
{
    float time_Musicre = 0.0;
    float timeMusicre_start, timeMusicre_end; // Total MUSIC Algorithm time
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    //-------------------------------------------------------------------
    printf("---------------\n");
    printf("--MUSIC DOA--\n");
    printf("---------------\n");
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);

    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.5;
    float kc = 2 * PI / lemda;
    const int nd = 512;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        printf("%.1f, ", angle[a]);
    }
    printf("\n");
    //---------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
            //printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
        }
    }
    //---------------------------------------------------------------
    float *t_sig_re = (float *)malloc(nd * len_t_theta * sizeof(float));
    float *t_sig_im = (float *)malloc(nd * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < len_t_theta; ++i)
    {
        for (int j = 0; j < nd; ++j)
        {
            cpp_t_sig(&t_sig_re[i * nd + j], &t_sig_im[i * nd + j]);
            //printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
        }
    }
    //---------------------------------------------------------------
    float *sig_co_re = (float *)malloc(M * nd * sizeof(float));
    float *sig_co_im = (float *)malloc(M * nd * sizeof(float));
    //---------------------------------------------------------------
    float *x_r_re = (float *)malloc(M * nd * sizeof(float));
    float *x_r_im = (float *)malloc(M * nd * sizeof(float));
    //---------------------------------------------------------------
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    //print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    gettimeofday(&time_AWGN_end, NULL);
    //for (int a = 0; a < M * nd; a++)
    //{
        //printf("\t(%f,%f)\n", x_r_re[a], x_r_im[a]);
    //}
    //---------------------------------------------------------------
    float *R_xx_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_im = (float *)malloc(M * M * sizeof(float));
    float M_re = M;
    float M_im = 0.0;
    float *M_ptr = &M_re;
    float *M_ptr_im = &M_im;

    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int16_t i = 0; i < M * M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        //printf("(%f,%f) ", R_xx_re[i], R_xx_im[i]);
    }
    //print_complex_matrix_R_xx(R_xx_re, R_xx_im, M, M);
    //printf("---");
    //print_complex_matrix( R_xx_re, R_xx_im,M,M);   //-10度時右上角左下角是負的 -5度時是正的
    //32*32 50度
    /*
    R_xx_re[0]=15.77; R_xx_im[0]=0.00;R_xx_re[1]=-11.67; R_xx_im[1]=-10.57;R_xx_re[2]=1.60; R_xx_im[2]=15.63;R_xx_re[3]=9.34; R_xx_im[3]=-12.66;R_xx_re[4]=-15.45; R_xx_im[4]=3.04;R_xx_re[5]=13.52; R_xx_im[5]=8.09;R_xx_re[6]=-4.66; R_xx_im[6]=-14.94;R_xx_re[7]=-6.61; R_xx_im[7]=14.37;R_xx_re[8]=14.47; R_xx_im[8]=-6.05;R_xx_re[9]=-14.91; R_xx_im[9]=-5.21;R_xx_re[10]=7.55; R_xx_im[10]=13.76;R_xx_re[11]=3.58; R_xx_im[11]=-15.44;R_xx_re[12]=-13.12; R_xx_im[12]=8.82;R_xx_re[13]=15.61; R_xx_im[13]=2.21;R_xx_re[14]=-10.13; R_xx_im[14]=-11.84;R_xx_re[15]=-0.29; R_xx_im[15]=15.84;R_xx_re[16]=10.89; R_xx_im[16]=-11.38;R_xx_re[17]=-15.80; R_xx_im[17]=1.18;R_xx_re[18]=12.40; R_xx_im[18]=9.81;R_xx_re[19]=-2.71; R_xx_im[19]=-15.39;R_xx_re[20]=-8.31; R_xx_im[20]=13.65;R_xx_re[21]=15.20; R_xx_im[21]=-4.23;R_xx_re[22]=-14.17; R_xx_im[22]=-7.16;R_xx_re[23]=5.72; R_xx_im[23]=14.48;R_xx_re[24]=5.51; R_xx_im[24]=-14.55;R_xx_re[25]=-13.83; R_xx_im[25]=7.24;R_xx_re[26]=15.22; R_xx_im[26]=4.18;R_xx_re[27]=-8.69; R_xx_im[27]=-13.22;R_xx_re[28]=-2.75; R_xx_im[28]=15.81;R_xx_re[29]=12.45; R_xx_im[29]=-9.70;R_xx_re[30]=-15.90; R_xx_im[30]=-1.19;R_xx_re[31]=11.28; R_xx_im[31]=11.41;
R_xx_re[32]=-11.67; R_xx_im[32]=10.57;R_xx_re[33]=15.80; R_xx_im[33]=0.00;R_xx_re[34]=-11.68; R_xx_im[34]=-10.49;R_xx_re[35]=1.58; R_xx_im[35]=15.64;R_xx_re[36]=9.40; R_xx_im[36]=-12.63;R_xx_re[37]=-15.44; R_xx_im[37]=3.09;R_xx_re[38]=13.48; R_xx_im[38]=7.94;R_xx_re[39]=-4.76; R_xx_im[39]=-15.08;R_xx_re[40]=-6.66; R_xx_im[40]=14.20;R_xx_re[41]=14.53; R_xx_im[41]=-6.15;R_xx_re[42]=-14.83; R_xx_im[42]=-5.12;R_xx_re[43]=7.71; R_xx_im[43]=13.84;R_xx_re[44]=3.80; R_xx_im[44]=-15.34;R_xx_re[45]=-13.05; R_xx_im[45]=8.84;R_xx_re[46]=15.44; R_xx_im[46]=1.96;R_xx_re[47]=-10.40; R_xx_im[47]=-11.92;R_xx_re[48]=-0.43; R_xx_im[48]=15.74;R_xx_re[49]=10.90; R_xx_im[49]=-11.47;R_xx_re[50]=-15.79; R_xx_im[50]=1.07;R_xx_re[51]=12.34; R_xx_im[51]=9.58;R_xx_re[52]=-3.01; R_xx_im[52]=-15.68;R_xx_re[53]=-8.43; R_xx_im[53]=13.33;R_xx_re[54]=15.32; R_xx_im[54]=-4.21;R_xx_re[55]=-13.97; R_xx_im[55]=-6.89;R_xx_re[56]=5.67; R_xx_im[56]=14.47;R_xx_re[57]=5.37; R_xx_im[57]=-14.65;R_xx_re[58]=-14.07; R_xx_im[58]=7.13;R_xx_re[59]=15.31; R_xx_im[59]=3.95;R_xx_re[60]=-8.58; R_xx_im[60]=-13.57;R_xx_re[61]=-2.73; R_xx_im[61]=15.54;R_xx_re[62]=12.58; R_xx_im[62]=-9.79;R_xx_re[63]=-16.01; R_xx_im[63]=-0.87;
R_xx_re[64]=1.60; R_xx_im[64]=-15.63;R_xx_re[65]=-11.68; R_xx_im[65]=10.49;R_xx_re[66]=15.79; R_xx_im[66]=0.00;R_xx_re[67]=-11.62; R_xx_im[67]=-10.55;R_xx_re[68]=1.45; R_xx_im[68]=15.64;R_xx_re[69]=9.40; R_xx_im[69]=-12.59;R_xx_re[70]=-15.30; R_xx_im[70]=3.11;R_xx_re[71]=13.58; R_xx_im[71]=8.01;R_xx_re[72]=-4.54; R_xx_im[72]=-14.97;R_xx_re[73]=-6.67; R_xx_im[73]=14.25;R_xx_re[74]=14.43; R_xx_im[74]=-6.09;R_xx_re[75]=-14.96; R_xx_im[75]=-5.13;R_xx_re[76]=7.40; R_xx_im[76]=13.90;R_xx_re[77]=3.78; R_xx_im[77]=-15.26;R_xx_re[78]=-12.76; R_xx_im[78]=8.85;R_xx_re[79]=15.68; R_xx_im[79]=1.88;R_xx_re[80]=-10.18; R_xx_im[80]=-11.95;R_xx_re[81]=-0.42; R_xx_im[81]=15.79;R_xx_re[82]=11.00; R_xx_im[82]=-11.32;R_xx_re[83]=-15.55; R_xx_im[83]=1.12;R_xx_re[84]=12.71; R_xx_im[84]=9.63;R_xx_re[85]=-2.64; R_xx_im[85]=-15.50;R_xx_re[86]=-8.53; R_xx_im[86]=13.34;R_xx_re[87]=14.95; R_xx_im[87]=-4.22;R_xx_re[88]=-13.88; R_xx_im[88]=-6.94;R_xx_re[89]=5.79; R_xx_im[89]=14.45;R_xx_re[90]=5.70; R_xx_im[90]=-14.68;R_xx_re[91]=-13.99; R_xx_im[91]=7.28;R_xx_re[92]=15.41; R_xx_im[92]=4.35;R_xx_re[93]=-8.35; R_xx_im[93]=-13.35;R_xx_re[94]=-2.81; R_xx_im[94]=15.68;R_xx_re[95]=12.46; R_xx_im[95]=-10.04;
R_xx_re[96]=9.34; R_xx_im[96]=12.66;R_xx_re[97]=1.58; R_xx_im[97]=-15.64;R_xx_re[98]=-11.62; R_xx_im[98]=10.55;R_xx_re[99]=15.89; R_xx_im[99]=0.00;R_xx_re[100]=-11.61; R_xx_im[100]=-10.62;R_xx_re[101]=1.51; R_xx_im[101]=15.67;R_xx_re[102]=9.25; R_xx_im[102]=-12.61;R_xx_re[103]=-15.46; R_xx_im[103]=3.21;R_xx_re[104]=13.45; R_xx_im[104]=8.04;R_xx_re[105]=-4.63; R_xx_im[105]=-15.07;R_xx_re[106]=-6.59; R_xx_im[106]=14.24;R_xx_re[107]=14.55; R_xx_im[107]=-6.28;R_xx_re[108]=-14.87; R_xx_im[108]=-5.33;R_xx_re[109]=7.47; R_xx_im[109]=13.85;R_xx_re[110]=3.51; R_xx_im[110]=-15.15;R_xx_re[111]=-12.91; R_xx_im[111]=9.15;R_xx_re[112]=15.62; R_xx_im[112]=2.00;R_xx_re[113]=-10.34; R_xx_im[113]=-11.98;R_xx_re[114]=-0.53; R_xx_im[114]=15.81;R_xx_re[115]=10.76; R_xx_im[115]=-11.30;R_xx_re[116]=-15.92; R_xx_im[116]=1.39;R_xx_re[117]=12.39; R_xx_im[117]=9.72;R_xx_re[118]=-2.63; R_xx_im[118]=-15.63;R_xx_re[119]=-8.25; R_xx_im[119]=13.18;R_xx_re[120]=14.95; R_xx_im[120]=-4.21;R_xx_re[121]=-14.03; R_xx_im[121]=-6.83;R_xx_re[122]=5.68; R_xx_im[122]=14.72;R_xx_re[123]=5.48; R_xx_im[123]=-14.83;R_xx_re[124]=-14.32; R_xx_im[124]=7.15;R_xx_re[125]=15.16; R_xx_im[125]=4.25;R_xx_re[126]=-8.46; R_xx_im[126]=-13.49;R_xx_re[127]=-2.46; R_xx_im[127]=15.84;
R_xx_re[128]=-15.45; R_xx_im[128]=-3.04;R_xx_re[129]=9.40; R_xx_im[129]=12.63;R_xx_re[130]=1.45; R_xx_im[130]=-15.64;R_xx_re[131]=-11.61; R_xx_im[131]=10.62;R_xx_re[132]=15.96; R_xx_im[132]=0.00;R_xx_re[133]=-11.72; R_xx_im[133]=-10.55;R_xx_re[134]=1.69; R_xx_im[134]=15.54;R_xx_re[135]=9.26; R_xx_im[135]=-12.81;R_xx_re[136]=-15.35; R_xx_im[136]=3.16;R_xx_re[137]=13.61; R_xx_im[137]=7.99;R_xx_re[138]=-4.74; R_xx_im[138]=-14.97;R_xx_re[139]=-6.49; R_xx_im[139]=14.46;R_xx_re[140]=14.56; R_xx_im[140]=-6.11;R_xx_re[141]=-14.88; R_xx_im[141]=-5.17;R_xx_re[142]=7.65; R_xx_im[142]=13.54;R_xx_re[143]=3.34; R_xx_im[143]=-15.47;R_xx_re[144]=-12.88; R_xx_im[144]=9.06;R_xx_re[145]=15.73; R_xx_im[145]=1.87;R_xx_re[146]=-10.29; R_xx_im[146]=-12.04;R_xx_re[147]=-0.29; R_xx_im[147]=15.61;R_xx_re[148]=10.82; R_xx_im[148]=-11.77;R_xx_re[149]=-15.71; R_xx_im[149]=1.21;R_xx_re[150]=12.52; R_xx_im[150]=9.76;R_xx_re[151]=-2.81; R_xx_im[151]=-15.30;R_xx_re[152]=-8.21; R_xx_im[152]=13.21;R_xx_re[153]=14.96; R_xx_im[153]=-4.44;R_xx_re[154]=-14.14; R_xx_im[154]=-7.09;R_xx_re[155]=5.99; R_xx_im[155]=14.63;R_xx_re[156]=5.75; R_xx_im[156]=-14.95;R_xx_re[157]=-14.05; R_xx_im[157]=7.07;R_xx_re[158]=15.39; R_xx_im[158]=4.22;R_xx_re[159]=-8.85; R_xx_im[159]=-13.40;
R_xx_re[160]=13.52; R_xx_im[160]=-8.09;R_xx_re[161]=-15.44; R_xx_im[161]=-3.09;R_xx_re[162]=9.40; R_xx_im[162]=12.59;R_xx_re[163]=1.51; R_xx_im[163]=-15.67;R_xx_re[164]=-11.72; R_xx_im[164]=10.55;R_xx_re[165]=16.05; R_xx_im[165]=0.00;R_xx_re[166]=-11.68; R_xx_im[166]=-10.46;R_xx_re[167]=1.70; R_xx_im[167]=15.73;R_xx_re[168]=9.31; R_xx_im[168]=-12.64;R_xx_re[169]=-15.48; R_xx_im[169]=3.18;R_xx_re[170]=13.56; R_xx_im[170]=7.94;R_xx_re[171]=-4.85; R_xx_im[171]=-15.13;R_xx_re[172]=-6.73; R_xx_im[172]=14.36;R_xx_re[173]=14.52; R_xx_im[173]=-6.11;R_xx_re[174]=-14.75; R_xx_im[174]=-4.94;R_xx_re[175]=7.88; R_xx_im[175]=13.75;R_xx_re[176]=3.49; R_xx_im[176]=-15.39;R_xx_re[177]=-12.95; R_xx_im[177]=9.14;R_xx_re[178]=15.68; R_xx_im[178]=2.06;R_xx_re[179]=-10.20; R_xx_im[179]=-11.81;R_xx_re[180]=-0.14; R_xx_im[180]=15.98;R_xx_re[181]=10.91; R_xx_im[181]=-11.44;R_xx_re[182]=-15.86; R_xx_im[182]=1.14;R_xx_re[183]=12.35; R_xx_im[183]=9.49;R_xx_re[184]=-2.78; R_xx_im[184]=-15.33;R_xx_re[185]=-8.17; R_xx_im[185]=13.31;R_xx_re[186]=15.24; R_xx_im[186]=-4.24;R_xx_re[187]=-14.24; R_xx_im[187]=-6.89;R_xx_re[188]=5.79; R_xx_im[188]=14.94;R_xx_re[189]=5.73; R_xx_im[189]=-14.72;R_xx_re[190]=-14.29; R_xx_im[190]=7.14;R_xx_re[191]=15.55; R_xx_im[191]=4.00;
R_xx_re[192]=-4.66; R_xx_im[192]=14.94;R_xx_re[193]=13.48; R_xx_im[193]=-7.94;R_xx_re[194]=-15.30; R_xx_im[194]=-3.11;R_xx_re[195]=9.25; R_xx_im[195]=12.61;R_xx_re[196]=1.69; R_xx_im[196]=-15.54;R_xx_re[197]=-11.68; R_xx_im[197]=10.46;R_xx_re[198]=15.86; R_xx_im[198]=0.00;R_xx_re[199]=-11.66; R_xx_im[199]=-10.51;R_xx_re[200]=1.45; R_xx_im[200]=15.53;R_xx_re[201]=9.34; R_xx_im[201]=-12.58;R_xx_re[202]=-15.28; R_xx_im[202]=3.12;R_xx_re[203]=13.59; R_xx_im[203]=7.95;R_xx_re[204]=-4.49; R_xx_im[204]=-15.07;R_xx_re[205]=-6.73; R_xx_im[205]=14.16;R_xx_re[206]=14.21; R_xx_im[206]=-6.08;R_xx_re[207]=-14.90; R_xx_im[207]=-4.96;R_xx_re[208]=7.58; R_xx_im[208]=13.70;R_xx_re[209]=3.54; R_xx_im[209]=-15.34;R_xx_re[210]=-12.97; R_xx_im[210]=8.87;R_xx_re[211]=15.42; R_xx_im[211]=1.97;R_xx_re[212]=-10.50; R_xx_im[212]=-11.92;R_xx_re[213]=-0.50; R_xx_im[213]=15.69;R_xx_re[214]=11.01; R_xx_im[214]=-11.34;R_xx_re[215]=-15.44; R_xx_im[215]=1.14;R_xx_re[216]=12.18; R_xx_im[216]=9.52;R_xx_re[217]=-2.77; R_xx_im[217]=-15.27;R_xx_re[218]=-8.45; R_xx_im[218]=13.20;R_xx_re[219]=15.08; R_xx_im[219]=-4.34;R_xx_re[220]=-14.13; R_xx_im[220]=-7.24;R_xx_re[221]=5.52; R_xx_im[221]=14.66;R_xx_re[222]=5.84; R_xx_im[222]=-14.76;R_xx_re[223]=-14.15; R_xx_im[223]=7.34;
R_xx_re[224]=-6.61; R_xx_im[224]=-14.37;R_xx_re[225]=-4.76; R_xx_im[225]=15.08;R_xx_re[226]=13.58; R_xx_im[226]=-8.01;R_xx_re[227]=-15.46; R_xx_im[227]=-3.21;R_xx_re[228]=9.26; R_xx_im[228]=12.81;R_xx_re[229]=1.70; R_xx_im[229]=-15.73;R_xx_re[230]=-11.66; R_xx_im[230]=10.51;R_xx_re[231]=16.24; R_xx_im[231]=0.00;R_xx_re[232]=-11.58; R_xx_im[232]=-10.67;R_xx_re[233]=1.49; R_xx_im[233]=15.79;R_xx_re[234]=9.41; R_xx_im[234]=-12.66;R_xx_re[235]=-15.61; R_xx_im[235]=3.23;R_xx_re[236]=13.54; R_xx_im[236]=8.26;R_xx_re[237]=-4.53; R_xx_im[237]=-15.16;R_xx_re[238]=-6.52; R_xx_im[238]=14.19;R_xx_re[239]=14.58; R_xx_im[239]=-6.41;R_xx_re[240]=-14.96; R_xx_im[240]=-5.13;R_xx_re[241]=7.73; R_xx_im[241]=13.88;R_xx_re[242]=3.79; R_xx_im[242]=-15.45;R_xx_re[243]=-12.87; R_xx_im[243]=8.94;R_xx_re[244]=15.92; R_xx_im[244]=1.86;R_xx_re[245]=-10.22; R_xx_im[245]=-12.09;R_xx_re[246]=-0.60; R_xx_im[246]=15.96;R_xx_re[247]=10.81; R_xx_im[247]=-11.26;R_xx_re[248]=-15.58; R_xx_im[248]=1.10;R_xx_re[249]=12.40; R_xx_im[249]=9.58;R_xx_re[250]=-2.59; R_xx_im[250]=-15.67;R_xx_re[251]=-8.40; R_xx_im[251]=13.45;R_xx_re[252]=15.60; R_xx_im[252]=-4.08;R_xx_re[253]=-14.02; R_xx_im[253]=-7.29;R_xx_re[254]=5.59; R_xx_im[254]=15.01;R_xx_re[255]=5.72; R_xx_im[255]=-15.11;
R_xx_re[256]=14.47; R_xx_im[256]=6.05;R_xx_re[257]=-6.66; R_xx_im[257]=-14.20;R_xx_re[258]=-4.54; R_xx_im[258]=14.97;R_xx_re[259]=13.45; R_xx_im[259]=-8.04;R_xx_re[260]=-15.35; R_xx_im[260]=-3.16;R_xx_re[261]=9.31; R_xx_im[261]=12.64;R_xx_re[262]=1.45; R_xx_im[262]=-15.53;R_xx_re[263]=-11.58; R_xx_im[263]=10.67;R_xx_re[264]=16.05; R_xx_im[264]=0.00;R_xx_re[265]=-11.71; R_xx_im[265]=-10.52;R_xx_re[266]=1.64; R_xx_im[266]=15.56;R_xx_re[267]=9.24; R_xx_im[267]=-12.83;R_xx_re[268]=-15.42; R_xx_im[268]=3.06;R_xx_re[269]=13.47; R_xx_im[269]=8.03;R_xx_re[270]=-4.73; R_xx_im[270]=-14.79;R_xx_re[271]=-6.36; R_xx_im[271]=14.46;R_xx_re[272]=14.38; R_xx_im[272]=-6.28;R_xx_re[273]=-14.97; R_xx_im[273]=-4.99;R_xx_re[274]=7.62; R_xx_im[274]=13.79;R_xx_re[275]=3.42; R_xx_im[275]=-15.15;R_xx_re[276]=-12.89; R_xx_im[276]=9.32;R_xx_re[277]=15.59; R_xx_im[277]=1.94;R_xx_re[278]=-10.32; R_xx_im[278]=-12.04;R_xx_re[279]=-0.28; R_xx_im[279]=15.55;R_xx_re[280]=10.65; R_xx_im[280]=-11.25;R_xx_re[281]=-15.49; R_xx_im[281]=1.36;R_xx_re[282]=12.38; R_xx_im[282]=9.68;R_xx_re[283]=-2.88; R_xx_im[283]=-15.53;R_xx_re[284]=-8.57; R_xx_im[284]=13.54;R_xx_re[285]=15.19; R_xx_im[285]=-4.11;R_xx_re[286]=-14.13; R_xx_im[286]=-7.23;R_xx_re[287]=5.98; R_xx_im[287]=14.82;
R_xx_re[288]=-14.91; R_xx_im[288]=5.21;R_xx_re[289]=14.53; R_xx_im[289]=6.15;R_xx_re[290]=-6.67; R_xx_im[290]=-14.25;R_xx_re[291]=-4.63; R_xx_im[291]=15.07;R_xx_re[292]=13.61; R_xx_im[292]=-7.99;R_xx_re[293]=-15.48; R_xx_im[293]=-3.18;R_xx_re[294]=9.34; R_xx_im[294]=12.58;R_xx_re[295]=1.49; R_xx_im[295]=-15.79;R_xx_re[296]=-11.71; R_xx_im[296]=10.52;R_xx_re[297]=16.29; R_xx_im[297]=0.00;R_xx_re[298]=-11.71; R_xx_im[298]=-10.53;R_xx_re[299]=1.74; R_xx_im[299]=15.80;R_xx_re[300]=9.56; R_xx_im[300]=-12.70;R_xx_re[301]=-15.51; R_xx_im[301]=3.06;R_xx_re[302]=13.52; R_xx_im[302]=7.86;R_xx_re[303]=-4.94; R_xx_im[303]=-15.11;R_xx_re[304]=-6.55; R_xx_im[304]=14.34;R_xx_re[305]=14.57; R_xx_im[305]=-6.34;R_xx_re[306]=-15.00; R_xx_im[306]=-5.18;R_xx_re[307]=7.63; R_xx_im[307]=13.66;R_xx_re[308]=3.37; R_xx_im[308]=-15.69;R_xx_re[309]=-13.02; R_xx_im[309]=9.04;R_xx_re[310]=15.78; R_xx_im[310]=2.12;R_xx_re[311]=-10.24; R_xx_im[311]=-11.79;R_xx_re[312]=-0.37; R_xx_im[312]=15.59;R_xx_re[313]=10.70; R_xx_im[313]=-11.42;R_xx_re[314]=-15.82; R_xx_im[314]=1.06;R_xx_re[315]=12.59; R_xx_im[315]=9.68;R_xx_re[316]=-2.56; R_xx_im[316]=-15.92;R_xx_re[317]=-8.60; R_xx_im[317]=13.31;R_xx_re[318]=15.47; R_xx_im[318]=-4.18;R_xx_re[319]=-14.48; R_xx_im[319]=-7.07;
R_xx_re[320]=7.55; R_xx_im[320]=-13.76;R_xx_re[321]=-14.83; R_xx_im[321]=5.12;R_xx_re[322]=14.43; R_xx_im[322]=6.09;R_xx_re[323]=-6.59; R_xx_im[323]=-14.24;R_xx_re[324]=-4.74; R_xx_im[324]=14.97;R_xx_re[325]=13.56; R_xx_im[325]=-7.94;R_xx_re[326]=-15.28; R_xx_im[326]=-3.12;R_xx_re[327]=9.41; R_xx_im[327]=12.66;R_xx_re[328]=1.64; R_xx_im[328]=-15.56;R_xx_re[329]=-11.71; R_xx_im[329]=10.53;R_xx_re[330]=16.17; R_xx_im[330]=0.00;R_xx_re[331]=-11.75; R_xx_im[331]=-10.53;R_xx_re[332]=1.41; R_xx_im[332]=15.70;R_xx_re[333]=9.39; R_xx_im[333]=-12.58;R_xx_re[334]=-15.19; R_xx_im[334]=3.20;R_xx_re[335]=13.70; R_xx_im[335]=7.87;R_xx_re[336]=-4.72; R_xx_im[336]=-14.96;R_xx_re[337]=-6.53; R_xx_im[337]=14.33;R_xx_re[338]=14.54; R_xx_im[338]=-6.13;R_xx_re[339]=-14.75; R_xx_im[339]=-5.00;R_xx_re[340]=7.98; R_xx_im[340]=13.78;R_xx_re[341]=3.60; R_xx_im[341]=-15.32;R_xx_re[342]=-13.06; R_xx_im[342]=8.94;R_xx_re[343]=15.44; R_xx_im[343]=1.96;R_xx_re[344]=-10.09; R_xx_im[344]=-11.76;R_xx_re[345]=-0.28; R_xx_im[345]=15.51;R_xx_re[346]=10.97; R_xx_im[346]=-11.31;R_xx_re[347]=-15.73; R_xx_im[347]=1.26;R_xx_re[348]=12.54; R_xx_im[348]=9.96;R_xx_re[349]=-2.50; R_xx_im[349]=-15.58;R_xx_re[350]=-8.68; R_xx_im[350]=13.33;R_xx_re[351]=15.37; R_xx_im[351]=-4.41;
R_xx_re[352]=3.58; R_xx_im[352]=15.44;R_xx_re[353]=7.71; R_xx_im[353]=-13.84;R_xx_re[354]=-14.96; R_xx_im[354]=5.13;R_xx_re[355]=14.55; R_xx_im[355]=6.28;R_xx_re[356]=-6.49; R_xx_im[356]=-14.46;R_xx_re[357]=-4.85; R_xx_im[357]=15.13;R_xx_re[358]=13.59; R_xx_im[358]=-7.95;R_xx_re[359]=-15.61; R_xx_im[359]=-3.23;R_xx_re[360]=9.24; R_xx_im[360]=12.83;R_xx_re[361]=1.74; R_xx_im[361]=-15.80;R_xx_re[362]=-11.75; R_xx_im[362]=10.53;R_xx_re[363]=16.51; R_xx_im[363]=0.00;R_xx_re[364]=-11.60; R_xx_im[364]=-10.85;R_xx_re[365]=1.36; R_xx_im[365]=15.78;R_xx_re[366]=9.30; R_xx_im[366]=-12.62;R_xx_re[367]=-15.61; R_xx_im[367]=3.32;R_xx_re[368]=13.64; R_xx_im[368]=8.09;R_xx_re[369]=-4.75; R_xx_im[369]=-15.26;R_xx_re[370]=-6.87; R_xx_im[370]=14.41;R_xx_re[371]=14.47; R_xx_im[371]=-6.16;R_xx_re[372]=-15.27; R_xx_im[372]=-5.02;R_xx_re[373]=7.55; R_xx_im[373]=13.95;R_xx_re[374]=3.81; R_xx_im[374]=-15.52;R_xx_re[375]=-12.91; R_xx_im[375]=8.89;R_xx_re[376]=15.52; R_xx_im[376]=2.12;R_xx_re[377]=-10.21; R_xx_im[377]=-11.88;R_xx_re[378]=-0.65; R_xx_im[378]=15.90;R_xx_re[379]=11.01; R_xx_im[379]=-11.52;R_xx_re[380]=-16.09; R_xx_im[380]=0.91;R_xx_re[381]=12.37; R_xx_im[381]=9.99;R_xx_re[382]=-2.46; R_xx_im[382]=-15.90;R_xx_re[383]=-8.63; R_xx_im[383]=13.67;
R_xx_re[384]=-13.12; R_xx_im[384]=-8.82;R_xx_re[385]=3.80; R_xx_im[385]=15.34;R_xx_re[386]=7.40; R_xx_im[386]=-13.90;R_xx_re[387]=-14.87; R_xx_im[387]=5.33;R_xx_re[388]=14.56; R_xx_im[388]=6.11;R_xx_re[389]=-6.73; R_xx_im[389]=-14.36;R_xx_re[390]=-4.49; R_xx_im[390]=15.07;R_xx_re[391]=13.54; R_xx_im[391]=-8.26;R_xx_re[392]=-15.42; R_xx_im[392]=-3.06;R_xx_re[393]=9.56; R_xx_im[393]=12.70;R_xx_re[394]=1.41; R_xx_im[394]=-15.70;R_xx_re[395]=-11.60; R_xx_im[395]=10.85;R_xx_re[396]=16.49; R_xx_im[396]=0.00;R_xx_re[397]=-11.77; R_xx_im[397]=-10.55;R_xx_re[398]=1.79; R_xx_im[398]=15.55;R_xx_re[399]=9.11; R_xx_im[399]=-13.05;R_xx_re[400]=-15.49; R_xx_im[400]=3.39;R_xx_re[401]=13.85; R_xx_im[401]=7.83;R_xx_re[402]=-4.83; R_xx_im[402]=-15.16;R_xx_re[403]=-6.39; R_xx_im[403]=14.29;R_xx_re[404]=14.57; R_xx_im[404]=-6.76;R_xx_re[405]=-15.01; R_xx_im[405]=-5.01;R_xx_re[406]=7.79; R_xx_im[406]=13.90;R_xx_re[407]=3.32; R_xx_im[407]=-15.32;R_xx_re[408]=-12.76; R_xx_im[408]=9.02;R_xx_re[409]=15.58; R_xx_im[409]=1.74;R_xx_re[410]=-10.35; R_xx_im[410]=-12.00;R_xx_re[411]=-0.14; R_xx_im[411]=15.84;R_xx_re[412]=11.14; R_xx_im[412]=-11.61;R_xx_re[413]=-15.86; R_xx_im[413]=1.10;R_xx_re[414]=12.59; R_xx_im[414]=9.90;R_xx_re[415]=-3.00; R_xx_im[415]=-15.85;
R_xx_re[416]=15.61; R_xx_im[416]=-2.21;R_xx_re[417]=-13.05; R_xx_im[417]=-8.84;R_xx_re[418]=3.78; R_xx_im[418]=15.26;R_xx_re[419]=7.47; R_xx_im[419]=-13.85;R_xx_re[420]=-14.88; R_xx_im[420]=5.17;R_xx_re[421]=14.52; R_xx_im[421]=6.11;R_xx_re[422]=-6.73; R_xx_im[422]=-14.16;R_xx_re[423]=-4.53; R_xx_im[423]=15.16;R_xx_re[424]=13.47; R_xx_im[424]=-8.03;R_xx_re[425]=-15.51; R_xx_im[425]=-3.06;R_xx_re[426]=9.39; R_xx_im[426]=12.58;R_xx_re[427]=1.36; R_xx_im[427]=-15.78;R_xx_re[428]=-11.77; R_xx_im[428]=10.55;R_xx_re[429]=16.44; R_xx_im[429]=0.00;R_xx_re[430]=-11.68; R_xx_im[430]=-10.26;R_xx_re[431]=1.96; R_xx_im[431]=15.73;R_xx_re[432]=9.21; R_xx_im[432]=-12.81;R_xx_re[433]=-15.46; R_xx_im[433]=3.43;R_xx_re[434]=13.67; R_xx_im[434]=7.96;R_xx_re[435]=-4.88; R_xx_im[435]=-14.88;R_xx_re[436]=-6.35; R_xx_im[436]=14.64;R_xx_re[437]=14.45; R_xx_im[437]=-6.27;R_xx_re[438]=-15.04; R_xx_im[438]=-5.09;R_xx_re[439]=7.69; R_xx_im[439]=13.56;R_xx_re[440]=3.39; R_xx_im[440]=-15.14;R_xx_re[441]=-12.66; R_xx_im[441]=9.12;R_xx_re[442]=15.69; R_xx_im[442]=1.96;R_xx_re[443]=-10.51; R_xx_im[443]=-11.84;R_xx_re[444]=-0.51; R_xx_im[444]=16.06;R_xx_re[445]=10.98; R_xx_im[445]=-11.35;R_xx_re[446]=-15.95; R_xx_im[446]=1.07;R_xx_re[447]=12.77; R_xx_im[447]=9.70;
R_xx_re[448]=-10.13; R_xx_im[448]=11.84;R_xx_re[449]=15.44; R_xx_im[449]=-1.96;R_xx_re[450]=-12.76; R_xx_im[450]=-8.85;R_xx_re[451]=3.51; R_xx_im[451]=15.15;R_xx_re[452]=7.65; R_xx_im[452]=-13.54;R_xx_re[453]=-14.75; R_xx_im[453]=4.94;R_xx_re[454]=14.21; R_xx_im[454]=6.08;R_xx_re[455]=-6.52; R_xx_im[455]=-14.19;R_xx_re[456]=-4.73; R_xx_im[456]=14.79;R_xx_re[457]=13.52; R_xx_im[457]=-7.86;R_xx_re[458]=-15.19; R_xx_im[458]=-3.20;R_xx_re[459]=9.30; R_xx_im[459]=12.62;R_xx_re[460]=1.79; R_xx_im[460]=-15.55;R_xx_re[461]=-11.68; R_xx_im[461]=10.26;R_xx_re[462]=16.13; R_xx_im[462]=0.00;R_xx_re[463]=-11.75; R_xx_im[463]=-10.42;R_xx_re[464]=1.57; R_xx_im[464]=15.49;R_xx_re[465]=9.27; R_xx_im[465]=-12.62;R_xx_re[466]=-15.34; R_xx_im[466]=2.98;R_xx_re[467]=13.27; R_xx_im[467]=7.89;R_xx_re[468]=-4.87; R_xx_im[468]=-15.08;R_xx_re[469]=-6.63; R_xx_im[469]=14.07;R_xx_re[470]=14.42; R_xx_im[470]=-6.09;R_xx_re[471]=-14.60; R_xx_im[471]=-5.01;R_xx_re[472]=7.42; R_xx_im[472]=13.50;R_xx_re[473]=3.42; R_xx_im[473]=-15.05;R_xx_re[474]=-12.91; R_xx_im[474]=8.78;R_xx_re[475]=15.54; R_xx_im[475]=2.00;R_xx_re[476]=-10.13; R_xx_im[476]=-12.22;R_xx_re[477]=-0.73; R_xx_im[477]=15.65;R_xx_re[478]=11.14; R_xx_im[478]=-11.21;R_xx_re[479]=-15.84; R_xx_im[479]=1.09;
R_xx_re[480]=-0.29; R_xx_im[480]=-15.84;R_xx_re[481]=-10.40; R_xx_im[481]=11.92;R_xx_re[482]=15.68; R_xx_im[482]=-1.88;R_xx_re[483]=-12.91; R_xx_im[483]=-9.15;R_xx_re[484]=3.34; R_xx_im[484]=15.47;R_xx_re[485]=7.88; R_xx_im[485]=-13.75;R_xx_re[486]=-14.90; R_xx_im[486]=4.96;R_xx_re[487]=14.58; R_xx_im[487]=6.41;R_xx_re[488]=-6.36; R_xx_im[488]=-14.46;R_xx_re[489]=-4.94; R_xx_im[489]=15.11;R_xx_re[490]=13.70; R_xx_im[490]=-7.87;R_xx_re[491]=-15.61; R_xx_im[491]=-3.32;R_xx_re[492]=9.11; R_xx_im[492]=13.05;R_xx_re[493]=1.96; R_xx_im[493]=-15.73;R_xx_re[494]=-11.75; R_xx_im[494]=10.42;R_xx_re[495]=16.70; R_xx_im[495]=0.00;R_xx_re[496]=-11.64; R_xx_im[496]=-10.72;R_xx_re[497]=1.46; R_xx_im[497]=15.93;R_xx_re[498]=9.63; R_xx_im[498]=-12.64;R_xx_re[499]=-15.40; R_xx_im[499]=3.03;R_xx_re[500]=13.86; R_xx_im[500]=8.08;R_xx_re[501]=-4.49; R_xx_im[501]=-15.20;R_xx_re[502]=-6.90; R_xx_im[502]=14.41;R_xx_re[503]=14.46; R_xx_im[503]=-5.98;R_xx_re[504]=-14.81; R_xx_im[504]=-5.26;R_xx_re[505]=7.52; R_xx_im[505]=13.80;R_xx_re[506]=3.90; R_xx_im[506]=-15.41;R_xx_re[507]=-13.18; R_xx_im[507]=8.98;R_xx_re[508]=15.94; R_xx_im[508]=2.46;R_xx_re[509]=-9.97; R_xx_im[509]=-12.33;R_xx_re[510]=-0.89; R_xx_im[510]=15.98;R_xx_re[511]=11.25; R_xx_im[511]=-11.55;
R_xx_re[512]=10.89; R_xx_im[512]=11.38;R_xx_re[513]=-0.43; R_xx_im[513]=-15.74;R_xx_re[514]=-10.18; R_xx_im[514]=11.95;R_xx_re[515]=15.62; R_xx_im[515]=-2.00;R_xx_re[516]=-12.88; R_xx_im[516]=-9.06;R_xx_re[517]=3.49; R_xx_im[517]=15.39;R_xx_re[518]=7.58; R_xx_im[518]=-13.70;R_xx_re[519]=-14.96; R_xx_im[519]=5.13;R_xx_re[520]=14.38; R_xx_im[520]=6.28;R_xx_re[521]=-6.55; R_xx_im[521]=-14.34;R_xx_re[522]=-4.72; R_xx_im[522]=14.96;R_xx_re[523]=13.64; R_xx_im[523]=-8.09;R_xx_re[524]=-15.49; R_xx_im[524]=-3.39;R_xx_re[525]=9.21; R_xx_im[525]=12.81;R_xx_re[526]=1.57; R_xx_im[526]=-15.49;R_xx_re[527]=-11.64; R_xx_im[527]=10.72;R_xx_re[528]=16.59; R_xx_im[528]=0.00;R_xx_re[529]=-11.81; R_xx_im[529]=-10.61;R_xx_re[530]=1.45; R_xx_im[530]=15.76;R_xx_re[531]=9.25; R_xx_im[531]=-12.63;R_xx_re[532]=-15.63; R_xx_im[532]=3.39;R_xx_re[533]=13.62; R_xx_im[533]=8.06;R_xx_re[534]=-4.62; R_xx_im[534]=-15.21;R_xx_re[535]=-6.53; R_xx_im[535]=14.13;R_xx_re[536]=14.31; R_xx_im[536]=-6.08;R_xx_re[537]=-14.74; R_xx_im[537]=-4.97;R_xx_re[538]=7.51; R_xx_im[538]=13.88;R_xx_re[539]=3.49; R_xx_im[539]=-15.38;R_xx_re[540]=-13.28; R_xx_im[540]=8.99;R_xx_re[541]=15.67; R_xx_im[541]=2.31;R_xx_re[542]=-10.13; R_xx_im[542]=-12.28;R_xx_re[543]=-0.42; R_xx_im[543]=16.04;
R_xx_re[544]=-15.80; R_xx_im[544]=-1.18;R_xx_re[545]=10.90; R_xx_im[545]=11.47;R_xx_re[546]=-0.42; R_xx_im[546]=-15.79;R_xx_re[547]=-10.34; R_xx_im[547]=11.98;R_xx_re[548]=15.73; R_xx_im[548]=-1.87;R_xx_re[549]=-12.95; R_xx_im[549]=-9.14;R_xx_re[550]=3.54; R_xx_im[550]=15.34;R_xx_re[551]=7.73; R_xx_im[551]=-13.88;R_xx_re[552]=-14.97; R_xx_im[552]=4.99;R_xx_re[553]=14.57; R_xx_im[553]=6.34;R_xx_re[554]=-6.53; R_xx_im[554]=-14.33;R_xx_re[555]=-4.75; R_xx_im[555]=15.26;R_xx_re[556]=13.85; R_xx_im[556]=-7.83;R_xx_re[557]=-15.46; R_xx_im[557]=-3.43;R_xx_re[558]=9.27; R_xx_im[558]=12.62;R_xx_re[559]=1.46; R_xx_im[559]=-15.93;R_xx_re[560]=-11.81; R_xx_im[560]=10.61;R_xx_re[561]=16.80; R_xx_im[561]=0.00;R_xx_re[562]=-11.74; R_xx_im[562]=-10.78;R_xx_re[563]=1.56; R_xx_im[563]=15.67;R_xx_re[564]=9.41; R_xx_im[564]=-13.04;R_xx_re[565]=-15.52; R_xx_im[565]=3.14;R_xx_re[566]=13.68; R_xx_im[566]=8.24;R_xx_re[567]=-4.59; R_xx_im[567]=-14.95;R_xx_re[568]=-6.64; R_xx_im[568]=14.20;R_xx_re[569]=14.48; R_xx_im[569]=-6.24;R_xx_re[570]=-14.96; R_xx_im[570]=-5.36;R_xx_re[571]=7.73; R_xx_im[571]=13.92;R_xx_re[572]=3.93; R_xx_im[572]=-15.63;R_xx_re[573]=-13.26; R_xx_im[573]=8.79;R_xx_re[574]=15.84; R_xx_im[574]=2.36;R_xx_re[575]=-10.47; R_xx_im[575]=-12.30;
R_xx_re[576]=12.40; R_xx_im[576]=-9.81;R_xx_re[577]=-15.79; R_xx_im[577]=-1.07;R_xx_re[578]=11.00; R_xx_im[578]=11.32;R_xx_re[579]=-0.53; R_xx_im[579]=-15.81;R_xx_re[580]=-10.29; R_xx_im[580]=12.04;R_xx_re[581]=15.68; R_xx_im[581]=-2.06;R_xx_re[582]=-12.97; R_xx_im[582]=-8.87;R_xx_re[583]=3.79; R_xx_im[583]=15.45;R_xx_re[584]=7.62; R_xx_im[584]=-13.79;R_xx_re[585]=-15.00; R_xx_im[585]=5.18;R_xx_re[586]=14.54; R_xx_im[586]=6.13;R_xx_re[587]=-6.87; R_xx_im[587]=-14.41;R_xx_re[588]=-4.83; R_xx_im[588]=15.16;R_xx_re[589]=13.67; R_xx_im[589]=-7.96;R_xx_re[590]=-15.34; R_xx_im[590]=-2.98;R_xx_re[591]=9.63; R_xx_im[591]=12.64;R_xx_re[592]=1.45; R_xx_im[592]=-15.76;R_xx_re[593]=-11.74; R_xx_im[593]=10.78;R_xx_re[594]=16.79; R_xx_im[594]=0.00;R_xx_re[595]=-11.70; R_xx_im[595]=-10.43;R_xx_re[596]=1.98; R_xx_im[596]=15.91;R_xx_re[597]=9.33; R_xx_im[597]=-12.79;R_xx_re[598]=-15.72; R_xx_im[598]=3.21;R_xx_re[599]=13.55; R_xx_im[599]=7.84;R_xx_re[600]=-4.77; R_xx_im[600]=-14.90;R_xx_re[601]=-6.35; R_xx_im[601]=14.28;R_xx_re[602]=14.66; R_xx_im[602]=-6.15;R_xx_re[603]=-15.04; R_xx_im[603]=-4.96;R_xx_re[604]=7.72; R_xx_im[604]=14.22;R_xx_re[605]=3.76; R_xx_im[605]=-15.39;R_xx_re[606]=-13.23; R_xx_im[606]=8.91;R_xx_re[607]=16.05; R_xx_im[607]=1.93;
R_xx_re[608]=-2.71; R_xx_im[608]=15.39;R_xx_re[609]=12.34; R_xx_im[609]=-9.58;R_xx_re[610]=-15.55; R_xx_im[610]=-1.12;R_xx_re[611]=10.76; R_xx_im[611]=11.30;R_xx_re[612]=-0.29; R_xx_im[612]=-15.61;R_xx_re[613]=-10.20; R_xx_im[613]=11.81;R_xx_re[614]=15.42; R_xx_im[614]=-1.97;R_xx_re[615]=-12.87; R_xx_im[615]=-8.94;R_xx_re[616]=3.42; R_xx_im[616]=15.15;R_xx_re[617]=7.63; R_xx_im[617]=-13.66;R_xx_re[618]=-14.75; R_xx_im[618]=5.00;R_xx_re[619]=14.47; R_xx_im[619]=6.16;R_xx_re[620]=-6.39; R_xx_im[620]=-14.29;R_xx_re[621]=-4.88; R_xx_im[621]=14.88;R_xx_re[622]=13.27; R_xx_im[622]=-7.89;R_xx_re[623]=-15.40; R_xx_im[623]=-3.03;R_xx_re[624]=9.25; R_xx_im[624]=12.63;R_xx_re[625]=1.56; R_xx_im[625]=-15.67;R_xx_re[626]=-11.70; R_xx_im[626]=10.43;R_xx_re[627]=16.46; R_xx_im[627]=0.00;R_xx_re[628]=-11.83; R_xx_im[628]=-10.45;R_xx_re[629]=1.51; R_xx_im[629]=15.54;R_xx_re[630]=9.44; R_xx_im[630]=-12.62;R_xx_re[631]=-15.14; R_xx_im[631]=3.14;R_xx_re[632]=13.29; R_xx_im[632]=7.91;R_xx_re[633]=-4.70; R_xx_im[633]=-14.77;R_xx_re[634]=-6.67; R_xx_im[634]=14.07;R_xx_re[635]=14.36; R_xx_im[635]=-6.25;R_xx_re[636]=-14.96; R_xx_im[636]=-5.39;R_xx_re[637]=7.39; R_xx_im[637]=13.79;R_xx_re[638]=3.89; R_xx_im[638]=-15.33;R_xx_re[639]=-13.15; R_xx_im[639]=9.05;
R_xx_re[640]=-8.31; R_xx_im[640]=-13.65;R_xx_re[641]=-3.01; R_xx_im[641]=15.68;R_xx_re[642]=12.71; R_xx_im[642]=-9.63;R_xx_re[643]=-15.92; R_xx_im[643]=-1.39;R_xx_re[644]=10.82; R_xx_im[644]=11.77;R_xx_re[645]=-0.14; R_xx_im[645]=-15.98;R_xx_re[646]=-10.50; R_xx_im[646]=11.92;R_xx_re[647]=15.92; R_xx_im[647]=-1.86;R_xx_re[648]=-12.89; R_xx_im[648]=-9.32;R_xx_re[649]=3.37; R_xx_im[649]=15.69;R_xx_re[650]=7.98; R_xx_im[650]=-13.78;R_xx_re[651]=-15.27; R_xx_im[651]=5.02;R_xx_re[652]=14.57; R_xx_im[652]=6.76;R_xx_re[653]=-6.35; R_xx_im[653]=-14.64;R_xx_re[654]=-4.87; R_xx_im[654]=15.08;R_xx_re[655]=13.86; R_xx_im[655]=-8.08;R_xx_re[656]=-15.63; R_xx_im[656]=-3.39;R_xx_re[657]=9.41; R_xx_im[657]=13.04;R_xx_re[658]=1.98; R_xx_im[658]=-15.91;R_xx_re[659]=-11.83; R_xx_im[659]=10.45;R_xx_re[660]=17.23; R_xx_im[660]=0.00;R_xx_re[661]=-11.74; R_xx_im[661]=-10.97;R_xx_re[662]=1.29; R_xx_im[662]=16.06;R_xx_re[663]=9.54; R_xx_im[663]=-12.64;R_xx_re[664]=-15.48; R_xx_im[664]=2.94;R_xx_re[665]=13.58; R_xx_im[665]=8.17;R_xx_re[666]=-4.38; R_xx_im[666]=-15.39;R_xx_re[667]=-6.84; R_xx_im[667]=14.50;R_xx_re[668]=15.15; R_xx_im[668]=-5.94;R_xx_re[669]=-14.98; R_xx_im[669]=-5.70;R_xx_re[670]=7.31; R_xx_im[670]=14.42;R_xx_re[671]=3.89; R_xx_im[671]=-15.74;
R_xx_re[672]=15.20; R_xx_im[672]=4.23;R_xx_re[673]=-8.43; R_xx_im[673]=-13.33;R_xx_re[674]=-2.64; R_xx_im[674]=15.50;R_xx_re[675]=12.39; R_xx_im[675]=-9.72;R_xx_re[676]=-15.71; R_xx_im[676]=-1.21;R_xx_re[677]=10.91; R_xx_im[677]=11.44;R_xx_re[678]=-0.50; R_xx_im[678]=-15.69;R_xx_re[679]=-10.22; R_xx_im[679]=12.09;R_xx_re[680]=15.59; R_xx_im[680]=-1.94;R_xx_re[681]=-13.02; R_xx_im[681]=-9.04;R_xx_re[682]=3.60; R_xx_im[682]=15.32;R_xx_re[683]=7.55; R_xx_im[683]=-13.95;R_xx_re[684]=-15.01; R_xx_im[684]=5.01;R_xx_re[685]=14.45; R_xx_im[685]=6.27;R_xx_re[686]=-6.63; R_xx_im[686]=-14.07;R_xx_re[687]=-4.49; R_xx_im[687]=15.20;R_xx_re[688]=13.62; R_xx_im[688]=-8.06;R_xx_re[689]=-15.52; R_xx_im[689]=-3.14;R_xx_re[690]=9.33; R_xx_im[690]=12.79;R_xx_re[691]=1.51; R_xx_im[691]=-15.54;R_xx_re[692]=-11.74; R_xx_im[692]=10.97;R_xx_re[693]=16.84; R_xx_im[693]=0.00;R_xx_re[694]=-11.69; R_xx_im[694]=-10.74;R_xx_re[695]=1.61; R_xx_im[695]=15.55;R_xx_re[696]=9.23; R_xx_im[696]=-12.53;R_xx_re[697]=-15.31; R_xx_im[697]=3.26;R_xx_re[698]=13.54; R_xx_im[698]=8.14;R_xx_re[699]=-4.85; R_xx_im[699]=-15.10;R_xx_re[700]=-6.88; R_xx_im[700]=14.53;R_xx_re[701]=14.59; R_xx_im[701]=-6.01;R_xx_re[702]=-15.03; R_xx_im[702]=-5.43;R_xx_re[703]=7.83; R_xx_im[703]=13.97;
R_xx_re[704]=-14.17; R_xx_im[704]=7.16;R_xx_re[705]=15.32; R_xx_im[705]=4.21;R_xx_re[706]=-8.53; R_xx_im[706]=-13.34;R_xx_re[707]=-2.63; R_xx_im[707]=15.63;R_xx_re[708]=12.52; R_xx_im[708]=-9.76;R_xx_re[709]=-15.86; R_xx_im[709]=-1.14;R_xx_re[710]=11.01; R_xx_im[710]=11.34;R_xx_re[711]=-0.60; R_xx_im[711]=-15.96;R_xx_re[712]=-10.32; R_xx_im[712]=12.04;R_xx_re[713]=15.78; R_xx_im[713]=-2.12;R_xx_re[714]=-13.06; R_xx_im[714]=-8.94;R_xx_re[715]=3.81; R_xx_im[715]=15.52;R_xx_re[716]=7.79; R_xx_im[716]=-13.90;R_xx_re[717]=-15.04; R_xx_im[717]=5.09;R_xx_re[718]=14.42; R_xx_im[718]=6.09;R_xx_re[719]=-6.90; R_xx_im[719]=-14.41;R_xx_re[720]=-4.62; R_xx_im[720]=15.21;R_xx_re[721]=13.68; R_xx_im[721]=-8.24;R_xx_re[722]=-15.72; R_xx_im[722]=-3.21;R_xx_re[723]=9.44; R_xx_im[723]=12.62;R_xx_re[724]=1.29; R_xx_im[724]=-16.06;R_xx_re[725]=-11.69; R_xx_im[725]=10.74;R_xx_re[726]=17.12; R_xx_im[726]=0.00;R_xx_re[727]=-11.77; R_xx_im[727]=-10.42;R_xx_re[728]=1.71; R_xx_im[728]=15.60;R_xx_re[729]=9.16; R_xx_im[729]=-12.80;R_xx_re[730]=-15.54; R_xx_im[730]=3.11;R_xx_re[731]=13.86; R_xx_im[731]=7.92;R_xx_re[732]=-4.83; R_xx_im[732]=-15.50;R_xx_re[733]=-6.83; R_xx_im[733]=14.40;R_xx_re[734]=14.92; R_xx_im[734]=-6.13;R_xx_re[735]=-15.42; R_xx_im[735]=-5.14;
R_xx_re[736]=5.72; R_xx_im[736]=-14.48;R_xx_re[737]=-13.97; R_xx_im[737]=6.89;R_xx_re[738]=14.95; R_xx_im[738]=4.22;R_xx_re[739]=-8.25; R_xx_im[739]=-13.18;R_xx_re[740]=-2.81; R_xx_im[740]=15.30;R_xx_re[741]=12.35; R_xx_im[741]=-9.49;R_xx_re[742]=-15.44; R_xx_im[742]=-1.14;R_xx_re[743]=10.81; R_xx_im[743]=11.26;R_xx_re[744]=-0.28; R_xx_im[744]=-15.55;R_xx_re[745]=-10.24; R_xx_im[745]=11.79;R_xx_re[746]=15.44; R_xx_im[746]=-1.96;R_xx_re[747]=-12.91; R_xx_im[747]=-8.89;R_xx_re[748]=3.32; R_xx_im[748]=15.32;R_xx_re[749]=7.69; R_xx_im[749]=-13.56;R_xx_re[750]=-14.60; R_xx_im[750]=5.01;R_xx_re[751]=14.46; R_xx_im[751]=5.98;R_xx_re[752]=-6.53; R_xx_im[752]=-14.13;R_xx_re[753]=-4.59; R_xx_im[753]=14.95;R_xx_re[754]=13.55; R_xx_im[754]=-7.84;R_xx_re[755]=-15.14; R_xx_im[755]=-3.14;R_xx_re[756]=9.54; R_xx_im[756]=12.64;R_xx_re[757]=1.61; R_xx_im[757]=-15.55;R_xx_re[758]=-11.77; R_xx_im[758]=10.42;R_xx_re[759]=16.58; R_xx_im[759]=0.00;R_xx_re[760]=-11.39; R_xx_im[760]=-10.31;R_xx_re[761]=1.61; R_xx_im[761]=15.26;R_xx_re[762]=9.38; R_xx_im[762]=-12.47;R_xx_re[763]=-15.30; R_xx_im[763]=3.18;R_xx_re[764]=13.60; R_xx_im[764]=8.25;R_xx_re[765]=-4.41; R_xx_im[765]=-14.94;R_xx_re[766]=-6.92; R_xx_im[766]=14.22;R_xx_re[767]=14.53; R_xx_im[767]=-6.23;
R_xx_re[768]=5.51; R_xx_im[768]=14.55;R_xx_re[769]=5.67; R_xx_im[769]=-14.47;R_xx_re[770]=-13.88; R_xx_im[770]=6.94;R_xx_re[771]=14.95; R_xx_im[771]=4.21;R_xx_re[772]=-8.21; R_xx_im[772]=-13.21;R_xx_re[773]=-2.78; R_xx_im[773]=15.33;R_xx_re[774]=12.18; R_xx_im[774]=-9.52;R_xx_re[775]=-15.58; R_xx_im[775]=-1.10;R_xx_re[776]=10.65; R_xx_im[776]=11.25;R_xx_re[777]=-0.37; R_xx_im[777]=-15.59;R_xx_re[778]=-10.09; R_xx_im[778]=11.76;R_xx_re[779]=15.52; R_xx_im[779]=-2.12;R_xx_re[780]=-12.76; R_xx_im[780]=-9.02;R_xx_re[781]=3.39; R_xx_im[781]=15.14;R_xx_re[782]=7.42; R_xx_im[782]=-13.50;R_xx_re[783]=-14.81; R_xx_im[783]=5.26;R_xx_re[784]=14.31; R_xx_im[784]=6.08;R_xx_re[785]=-6.64; R_xx_im[785]=-14.20;R_xx_re[786]=-4.77; R_xx_im[786]=14.90;R_xx_re[787]=13.29; R_xx_im[787]=-7.91;R_xx_re[788]=-15.48; R_xx_im[788]=-2.94;R_xx_re[789]=9.23; R_xx_im[789]=12.53;R_xx_re[790]=1.71; R_xx_im[790]=-15.60;R_xx_re[791]=-11.39; R_xx_im[791]=10.31;R_xx_re[792]=16.50; R_xx_im[792]=0.00;R_xx_re[793]=-11.48; R_xx_im[793]=-10.24;R_xx_re[794]=1.44; R_xx_im[794]=15.52;R_xx_re[795]=9.21; R_xx_im[795]=-12.63;R_xx_re[796]=-15.57; R_xx_im[796]=3.07;R_xx_re[797]=13.30; R_xx_im[797]=8.11;R_xx_re[798]=-4.46; R_xx_im[798]=-15.08;R_xx_re[799]=-6.62; R_xx_im[799]=14.41;
R_xx_re[800]=-13.83; R_xx_im[800]=-7.24;R_xx_re[801]=5.37; R_xx_im[801]=14.65;R_xx_re[802]=5.79; R_xx_im[802]=-14.45;R_xx_re[803]=-14.03; R_xx_im[803]=6.83;R_xx_re[804]=14.96; R_xx_im[804]=4.44;R_xx_re[805]=-8.17; R_xx_im[805]=-13.31;R_xx_re[806]=-2.77; R_xx_im[806]=15.27;R_xx_re[807]=12.40; R_xx_im[807]=-9.58;R_xx_re[808]=-15.49; R_xx_im[808]=-1.36;R_xx_re[809]=10.70; R_xx_im[809]=11.42;R_xx_re[810]=-0.28; R_xx_im[810]=-15.51;R_xx_re[811]=-10.21; R_xx_im[811]=11.88;R_xx_re[812]=15.58; R_xx_im[812]=-1.74;R_xx_re[813]=-12.66; R_xx_im[813]=-9.12;R_xx_re[814]=3.42; R_xx_im[814]=15.05;R_xx_re[815]=7.52; R_xx_im[815]=-13.80;R_xx_re[816]=-14.74; R_xx_im[816]=4.97;R_xx_re[817]=14.48; R_xx_im[817]=6.24;R_xx_re[818]=-6.35; R_xx_im[818]=-14.28;R_xx_re[819]=-4.70; R_xx_im[819]=14.77;R_xx_re[820]=13.58; R_xx_im[820]=-8.17;R_xx_re[821]=-15.31; R_xx_im[821]=-3.26;R_xx_re[822]=9.16; R_xx_im[822]=12.80;R_xx_re[823]=1.61; R_xx_im[823]=-15.26;R_xx_re[824]=-11.48; R_xx_im[824]=10.24;R_xx_re[825]=16.73; R_xx_im[825]=0.00;R_xx_re[826]=-11.41; R_xx_im[826]=-10.68;R_xx_re[827]=1.55; R_xx_im[827]=15.66;R_xx_re[828]=9.70; R_xx_im[828]=-12.60;R_xx_re[829]=-15.37; R_xx_im[829]=2.79;R_xx_re[830]=13.44; R_xx_im[830]=8.38;R_xx_re[831]=-4.64; R_xx_im[831]=-15.21;
R_xx_re[832]=15.22; R_xx_im[832]=-4.18;R_xx_re[833]=-14.07; R_xx_im[833]=-7.13;R_xx_re[834]=5.70; R_xx_im[834]=14.68;R_xx_re[835]=5.68; R_xx_im[835]=-14.72;R_xx_re[836]=-14.14; R_xx_im[836]=7.09;R_xx_re[837]=15.24; R_xx_im[837]=4.24;R_xx_re[838]=-8.45; R_xx_im[838]=-13.20;R_xx_re[839]=-2.59; R_xx_im[839]=15.67;R_xx_re[840]=12.38; R_xx_im[840]=-9.68;R_xx_re[841]=-15.82; R_xx_im[841]=-1.06;R_xx_re[842]=10.97; R_xx_im[842]=11.31;R_xx_re[843]=-0.65; R_xx_im[843]=-15.90;R_xx_re[844]=-10.35; R_xx_im[844]=12.00;R_xx_re[845]=15.69; R_xx_im[845]=-1.96;R_xx_re[846]=-12.91; R_xx_im[846]=-8.78;R_xx_re[847]=3.90; R_xx_im[847]=15.41;R_xx_re[848]=7.51; R_xx_im[848]=-13.88;R_xx_re[849]=-14.96; R_xx_im[849]=5.36;R_xx_re[850]=14.66; R_xx_im[850]=6.15;R_xx_re[851]=-6.67; R_xx_im[851]=-14.07;R_xx_re[852]=-4.38; R_xx_im[852]=15.39;R_xx_re[853]=13.54; R_xx_im[853]=-8.14;R_xx_re[854]=-15.54; R_xx_im[854]=-3.11;R_xx_re[855]=9.38; R_xx_im[855]=12.47;R_xx_re[856]=1.44; R_xx_im[856]=-15.52;R_xx_re[857]=-11.41; R_xx_im[857]=10.68;R_xx_re[858]=17.14; R_xx_im[858]=0.00;R_xx_re[859]=-11.90; R_xx_im[859]=-10.56;R_xx_re[860]=1.53; R_xx_im[860]=16.00;R_xx_re[861]=9.50; R_xx_im[861]=-12.63;R_xx_re[862]=-15.71; R_xx_im[862]=3.12;R_xx_re[863]=13.97; R_xx_im[863]=7.98;
R_xx_re[864]=-8.69; R_xx_im[864]=13.22;R_xx_re[865]=15.31; R_xx_im[865]=-3.95;R_xx_re[866]=-13.99; R_xx_im[866]=-7.28;R_xx_re[867]=5.48; R_xx_im[867]=14.83;R_xx_re[868]=5.99; R_xx_im[868]=-14.63;R_xx_re[869]=-14.24; R_xx_im[869]=6.89;R_xx_re[870]=15.08; R_xx_im[870]=4.34;R_xx_re[871]=-8.40; R_xx_im[871]=-13.45;R_xx_re[872]=-2.88; R_xx_im[872]=15.53;R_xx_re[873]=12.59; R_xx_im[873]=-9.68;R_xx_re[874]=-15.73; R_xx_im[874]=-1.26;R_xx_re[875]=11.01; R_xx_im[875]=11.52;R_xx_re[876]=-0.14; R_xx_im[876]=-15.84;R_xx_re[877]=-10.51; R_xx_im[877]=11.84;R_xx_re[878]=15.54; R_xx_im[878]=-2.00;R_xx_re[879]=-13.18; R_xx_im[879]=-8.98;R_xx_re[880]=3.49; R_xx_im[880]=15.38;R_xx_re[881]=7.73; R_xx_im[881]=-13.92;R_xx_re[882]=-15.04; R_xx_im[882]=4.96;R_xx_re[883]=14.36; R_xx_im[883]=6.25;R_xx_re[884]=-6.84; R_xx_im[884]=-14.50;R_xx_re[885]=-4.85; R_xx_im[885]=15.10;R_xx_re[886]=13.86; R_xx_im[886]=-7.92;R_xx_re[887]=-15.30; R_xx_im[887]=-3.18;R_xx_re[888]=9.21; R_xx_im[888]=12.63;R_xx_re[889]=1.55; R_xx_im[889]=-15.66;R_xx_re[890]=-11.90; R_xx_im[890]=10.56;R_xx_re[891]=17.16; R_xx_im[891]=0.00;R_xx_re[892]=-11.74; R_xx_im[892]=-11.07;R_xx_re[893]=1.28; R_xx_im[893]=15.76;R_xx_re[894]=9.79; R_xx_im[894]=-12.73;R_xx_re[895]=-15.73; R_xx_im[895]=3.17;
R_xx_re[896]=-2.75; R_xx_im[896]=-15.81;R_xx_re[897]=-8.58; R_xx_im[897]=13.57;R_xx_re[898]=15.41; R_xx_im[898]=-4.35;R_xx_re[899]=-14.32; R_xx_im[899]=-7.15;R_xx_re[900]=5.75; R_xx_im[900]=14.95;R_xx_re[901]=5.79; R_xx_im[901]=-14.94;R_xx_re[902]=-14.13; R_xx_im[902]=7.24;R_xx_re[903]=15.60; R_xx_im[903]=4.08;R_xx_re[904]=-8.57; R_xx_im[904]=-13.54;R_xx_re[905]=-2.56; R_xx_im[905]=15.92;R_xx_re[906]=12.54; R_xx_im[906]=-9.96;R_xx_re[907]=-16.09; R_xx_im[907]=-0.91;R_xx_re[908]=11.14; R_xx_im[908]=11.61;R_xx_re[909]=-0.51; R_xx_im[909]=-16.06;R_xx_re[910]=-10.13; R_xx_im[910]=12.22;R_xx_re[911]=15.94; R_xx_im[911]=-2.46;R_xx_re[912]=-13.28; R_xx_im[912]=-8.99;R_xx_re[913]=3.93; R_xx_im[913]=15.63;R_xx_re[914]=7.72; R_xx_im[914]=-14.22;R_xx_re[915]=-14.96; R_xx_im[915]=5.39;R_xx_re[916]=15.15; R_xx_im[916]=5.94;R_xx_re[917]=-6.88; R_xx_im[917]=-14.53;R_xx_re[918]=-4.83; R_xx_im[918]=15.50;R_xx_re[919]=13.60; R_xx_im[919]=-8.25;R_xx_re[920]=-15.57; R_xx_im[920]=-3.07;R_xx_re[921]=9.70; R_xx_im[921]=12.60;R_xx_re[922]=1.53; R_xx_im[922]=-16.00;R_xx_re[923]=-11.74; R_xx_im[923]=11.07;R_xx_re[924]=17.92; R_xx_im[924]=0.00;R_xx_re[925]=-11.91; R_xx_im[925]=-10.90;R_xx_re[926]=1.62; R_xx_im[926]=16.18;R_xx_re[927]=9.54; R_xx_im[927]=-13.29;
R_xx_re[928]=12.45; R_xx_im[928]=9.70;R_xx_re[929]=-2.73; R_xx_im[929]=-15.54;R_xx_re[930]=-8.35; R_xx_im[930]=13.35;R_xx_re[931]=15.16; R_xx_im[931]=-4.25;R_xx_re[932]=-14.05; R_xx_im[932]=-7.07;R_xx_re[933]=5.73; R_xx_im[933]=14.72;R_xx_re[934]=5.52; R_xx_im[934]=-14.66;R_xx_re[935]=-14.02; R_xx_im[935]=7.29;R_xx_re[936]=15.19; R_xx_im[936]=4.11;R_xx_re[937]=-8.60; R_xx_im[937]=-13.31;R_xx_re[938]=-2.50; R_xx_im[938]=15.58;R_xx_re[939]=12.37; R_xx_im[939]=-9.99;R_xx_re[940]=-15.86; R_xx_im[940]=-1.10;R_xx_re[941]=10.98; R_xx_im[941]=11.35;R_xx_re[942]=-0.73; R_xx_im[942]=-15.65;R_xx_re[943]=-9.97; R_xx_im[943]=12.33;R_xx_re[944]=15.67; R_xx_im[944]=-2.31;R_xx_re[945]=-13.26; R_xx_im[945]=-8.79;R_xx_re[946]=3.76; R_xx_im[946]=15.39;R_xx_re[947]=7.39; R_xx_im[947]=-13.79;R_xx_re[948]=-14.98; R_xx_im[948]=5.70;R_xx_re[949]=14.59; R_xx_im[949]=6.01;R_xx_re[950]=-6.83; R_xx_im[950]=-14.40;R_xx_re[951]=-4.41; R_xx_im[951]=14.94;R_xx_re[952]=13.30; R_xx_im[952]=-8.11;R_xx_re[953]=-15.37; R_xx_im[953]=-2.79;R_xx_re[954]=9.50; R_xx_im[954]=12.63;R_xx_re[955]=1.28; R_xx_im[955]=-15.76;R_xx_re[956]=-11.91; R_xx_im[956]=10.90;R_xx_re[957]=17.29; R_xx_im[957]=0.00;R_xx_re[958]=-11.83; R_xx_im[958]=-10.73;R_xx_re[959]=1.94; R_xx_im[959]=15.99;
R_xx_re[960]=-15.90; R_xx_im[960]=1.19;R_xx_re[961]=12.58; R_xx_im[961]=9.79;R_xx_re[962]=-2.81; R_xx_im[962]=-15.68;R_xx_re[963]=-8.46; R_xx_im[963]=13.49;R_xx_re[964]=15.39; R_xx_im[964]=-4.22;R_xx_re[965]=-14.29; R_xx_im[965]=-7.14;R_xx_re[966]=5.84; R_xx_im[966]=14.76;R_xx_re[967]=5.59; R_xx_im[967]=-15.01;R_xx_re[968]=-14.13; R_xx_im[968]=7.23;R_xx_re[969]=15.47; R_xx_im[969]=4.18;R_xx_re[970]=-8.68; R_xx_im[970]=-13.33;R_xx_re[971]=-2.46; R_xx_im[971]=15.90;R_xx_re[972]=12.59; R_xx_im[972]=-9.90;R_xx_re[973]=-15.95; R_xx_im[973]=-1.07;R_xx_re[974]=11.14; R_xx_im[974]=11.21;R_xx_re[975]=-0.89; R_xx_im[975]=-15.98;R_xx_re[976]=-10.13; R_xx_im[976]=12.28;R_xx_re[977]=15.84; R_xx_im[977]=-2.36;R_xx_re[978]=-13.23; R_xx_im[978]=-8.91;R_xx_re[979]=3.89; R_xx_im[979]=15.33;R_xx_re[980]=7.31; R_xx_im[980]=-14.42;R_xx_re[981]=-15.03; R_xx_im[981]=5.43;R_xx_re[982]=14.92; R_xx_im[982]=6.13;R_xx_re[983]=-6.92; R_xx_im[983]=-14.22;R_xx_re[984]=-4.46; R_xx_im[984]=15.08;R_xx_re[985]=13.44; R_xx_im[985]=-8.38;R_xx_re[986]=-15.71; R_xx_im[986]=-3.12;R_xx_re[987]=9.79; R_xx_im[987]=12.73;R_xx_re[988]=1.62; R_xx_im[988]=-16.18;R_xx_re[989]=-11.83; R_xx_im[989]=10.73;R_xx_re[990]=17.75; R_xx_im[990]=0.00;R_xx_re[991]=-12.22; R_xx_im[991]=-10.69;
R_xx_re[992]=11.28; R_xx_im[992]=-11.41;R_xx_re[993]=-16.01; R_xx_im[993]=0.87;R_xx_re[994]=12.46; R_xx_im[994]=10.04;R_xx_re[995]=-2.46; R_xx_im[995]=-15.84;R_xx_re[996]=-8.85; R_xx_im[996]=13.40;R_xx_re[997]=15.55; R_xx_im[997]=-4.00;R_xx_re[998]=-14.15; R_xx_im[998]=-7.34;R_xx_re[999]=5.72; R_xx_im[999]=15.11;R_xx_re[1000]=5.98; R_xx_im[1000]=-14.82;R_xx_re[1001]=-14.48; R_xx_im[1001]=7.07;R_xx_re[1002]=15.37; R_xx_im[1002]=4.41;R_xx_re[1003]=-8.63; R_xx_im[1003]=-13.67;R_xx_re[1004]=-3.00; R_xx_im[1004]=15.85;R_xx_re[1005]=12.77; R_xx_im[1005]=-9.70;R_xx_re[1006]=-15.84; R_xx_im[1006]=-1.09;R_xx_re[1007]=11.25; R_xx_im[1007]=11.55;R_xx_re[1008]=-0.42; R_xx_im[1008]=-16.04;R_xx_re[1009]=-10.47; R_xx_im[1009]=12.30;R_xx_re[1010]=16.05; R_xx_im[1010]=-1.93;R_xx_re[1011]=-13.15; R_xx_im[1011]=-9.05;R_xx_re[1012]=3.89; R_xx_im[1012]=15.74;R_xx_re[1013]=7.83; R_xx_im[1013]=-13.97;R_xx_re[1014]=-15.42; R_xx_im[1014]=5.14;R_xx_re[1015]=14.53; R_xx_im[1015]=6.23;R_xx_re[1016]=-6.62; R_xx_im[1016]=-14.41;R_xx_re[1017]=-4.64; R_xx_im[1017]=15.21;R_xx_re[1018]=13.97; R_xx_im[1018]=-7.98;R_xx_re[1019]=-15.73; R_xx_im[1019]=-3.17;R_xx_re[1020]=9.54; R_xx_im[1020]=13.29;R_xx_re[1021]=1.94; R_xx_im[1021]=-15.99;R_xx_re[1022]=-12.22; R_xx_im[1022]=10.69;R_xx_re[1023]=17.82; R_xx_im[1023]=0.00;
    */
    /*60度
    R_xx_re[0]=7700.22949;
    R_xx_im[0]=0;
    R_xx_re[1]=-6949.64111;
    R_xx_im[1]=-3128.52173;
    R_xx_re[2]=5121.00830;
    R_xx_im[2]=5737.69238;
    R_xx_re[3]=-2278.65601;
    R_xx_im[3]=-7329.69482;
    R_xx_re[4]=-6949.64111;
    R_xx_im[4]=3128.52173;
    R_xx_re[5]=7633.87012;
    R_xx_im[5]=0;
    R_xx_re[6]=-6968.37842;
    R_xx_im[6]=-3114.18311;
    R_xx_re[7]=5040.33936;
    R_xx_im[7]=5700.48779;
    R_xx_re[8]=5121.00830;
    R_xx_im[8]=-5737.69238;
    R_xx_re[9]=-6968.37842;
    R_xx_im[9]=3114.18311;
    R_xx_re[10]=7816.82861;
    R_xx_im[10]=0;
    R_xx_re[11]=-6991.15869;
    R_xx_im[11]=-3190.54102;
    R_xx_re[12]=-2278.65601;
    R_xx_im[12]=7329.69482;
    R_xx_re[13]=5040.33936;
    R_xx_im[13]=-5700.48779;
    R_xx_re[14]=-6991.15869;
    R_xx_im[14]=3190.54102;
    R_xx_re[15]=7839.22266;
    R_xx_im[15]=0;
    */    
   /* 16*16 30度
    R_xx_re[0]=30.89; R_xx_im[0]=0.00;R_xx_re[1]=21.68; R_xx_im[1]=-21.72;R_xx_re[2]=-0.09; R_xx_im[2]=-30.72;R_xx_re[3]=-21.84; R_xx_im[3]=-21.69;R_xx_re[4]=-30.33; R_xx_im[4]=-0.02;R_xx_re[5]=-21.63; R_xx_im[5]=21.76;R_xx_re[6]=-0.30; R_xx_im[6]=30.51;R_xx_re[7]=21.83; R_xx_im[7]=21.79;R_xx_re[8]=30.68; R_xx_im[8]=-0.16;R_xx_re[9]=21.81; R_xx_im[9]=-21.75;R_xx_re[10]=-0.06; R_xx_im[10]=-30.70;R_xx_re[11]=-21.49; R_xx_im[11]=-22.19;R_xx_re[12]=-31.10; R_xx_im[12]=0.12;R_xx_re[13]=-21.77; R_xx_im[13]=21.66;R_xx_re[14]=0.72; R_xx_im[14]=30.86;R_xx_re[15]=22.10; R_xx_im[15]=21.77;
R_xx_re[16]=21.68; R_xx_im[16]=21.72;R_xx_re[17]=30.89; R_xx_im[17]=0.00;R_xx_re[18]=21.65; R_xx_im[18]=-21.70;R_xx_re[19]=-0.07; R_xx_im[19]=-30.71;R_xx_re[20]=-21.36; R_xx_im[20]=-21.42;R_xx_re[21]=-30.62; R_xx_im[21]=0.08;R_xx_re[22]=-21.76; R_xx_im[22]=21.24;R_xx_re[23]=-0.02; R_xx_im[23]=30.74;R_xx_re[24]=21.71; R_xx_im[24]=21.54;R_xx_re[25]=30.73; R_xx_im[25]=0.08;R_xx_re[26]=21.61; R_xx_im[26]=-21.65;R_xx_re[27]=0.50; R_xx_im[27]=-30.81;R_xx_re[28]=-22.07; R_xx_im[28]=-21.85;R_xx_re[29]=-30.63; R_xx_im[29]=-0.10;R_xx_re[30]=-21.25; R_xx_im[30]=22.28;R_xx_re[31]=0.20; R_xx_im[31]=30.92;
R_xx_re[32]=-0.09; R_xx_im[32]=30.72;R_xx_re[33]=21.65; R_xx_im[33]=21.70;R_xx_re[34]=31.15; R_xx_im[34]=0.00;R_xx_re[35]=21.74; R_xx_im[35]=-21.75;R_xx_re[36]=0.07; R_xx_im[36]=-30.31;R_xx_re[37]=-21.68; R_xx_im[37]=-21.65;R_xx_re[38]=-30.41; R_xx_im[38]=-0.38;R_xx_re[39]=-21.82; R_xx_im[39]=21.75;R_xx_re[40]=0.10; R_xx_im[40]=30.67;R_xx_re[41]=21.65; R_xx_im[41]=21.83;R_xx_re[42]=30.61; R_xx_im[42]=-0.02;R_xx_re[43]=22.20; R_xx_im[43]=-21.42;R_xx_re[44]=-0.08; R_xx_im[44]=-31.10;R_xx_re[45]=-21.59; R_xx_im[45]=-21.74;R_xx_re[46]=-30.77; R_xx_im[46]=0.65;R_xx_re[47]=-21.74; R_xx_im[47]=22.01;
R_xx_re[48]=-21.84; R_xx_im[48]=21.69;R_xx_re[49]=-0.07; R_xx_im[49]=30.71;R_xx_re[50]=21.74; R_xx_im[50]=21.75;R_xx_re[51]=31.48; R_xx_im[51]=0.00;R_xx_re[52]=21.51; R_xx_im[52]=-21.37;R_xx_re[53]=-0.04; R_xx_im[53]=-30.72;R_xx_re[54]=-21.27; R_xx_im[54]=-21.85;R_xx_re[55]=-30.89; R_xx_im[55]=-0.11;R_xx_re[56]=-21.60; R_xx_im[56]=21.77;R_xx_re[57]=-0.15; R_xx_im[57]=30.99;R_xx_re[58]=21.68; R_xx_im[58]=21.79;R_xx_re[59]=30.88; R_xx_im[59]=0.65;R_xx_re[60]=21.98; R_xx_im[60]=-22.06;R_xx_re[61]=0.19; R_xx_im[61]=-30.79;R_xx_re[62]=-22.32; R_xx_im[62]=-21.38;R_xx_re[63]=-31.03; R_xx_im[63]=0.14;
R_xx_re[64]=-30.33; R_xx_im[64]=0.02;R_xx_re[65]=-21.36; R_xx_im[65]=21.42;R_xx_re[66]=0.07; R_xx_im[66]=30.31;R_xx_re[67]=21.51; R_xx_im[67]=21.37;R_xx_re[68]=30.69; R_xx_im[68]=0.00;R_xx_re[69]=21.29; R_xx_im[69]=-21.45;R_xx_re[70]=0.32; R_xx_im[70]=-29.98;R_xx_re[71]=-21.49; R_xx_im[71]=-21.50;R_xx_re[72]=-30.21; R_xx_im[72]=0.19;R_xx_re[73]=-21.53; R_xx_im[73]=21.40;R_xx_re[74]=0.13; R_xx_im[74]=30.16;R_xx_re[75]=21.18; R_xx_im[75]=21.86;R_xx_re[76]=30.72; R_xx_im[76]=-0.20;R_xx_re[77]=21.41; R_xx_im[77]=-21.35;R_xx_re[78]=-0.81; R_xx_im[78]=-30.48;R_xx_re[79]=-21.82; R_xx_im[79]=-21.32;
R_xx_re[80]=-21.63; R_xx_im[80]=-21.76;R_xx_re[81]=-30.62; R_xx_im[81]=-0.08;R_xx_re[82]=-21.68; R_xx_im[82]=21.65;R_xx_re[83]=-0.04; R_xx_im[83]=30.72;R_xx_re[84]=21.29; R_xx_im[84]=21.45;R_xx_re[85]=31.60; R_xx_im[85]=0.00;R_xx_re[86]=21.81; R_xx_im[86]=-21.25;R_xx_re[87]=0.14; R_xx_im[87]=-30.75;R_xx_re[88]=-21.73; R_xx_im[88]=-21.57;R_xx_re[89]=-30.75; R_xx_im[89]=-0.16;R_xx_re[90]=-21.68; R_xx_im[90]=21.62;R_xx_re[91]=-0.54; R_xx_im[91]=30.85;R_xx_re[92]=22.02; R_xx_im[92]=21.90;R_xx_re[93]=30.62; R_xx_im[93]=0.22;R_xx_re[94]=21.34; R_xx_im[94]=-22.15;R_xx_re[95]=-0.18; R_xx_im[95]=-30.93;
R_xx_re[96]=-0.30; R_xx_im[96]=-30.51;R_xx_re[97]=-21.76; R_xx_im[97]=-21.24;R_xx_re[98]=-30.41; R_xx_im[98]=0.38;R_xx_re[99]=-21.27; R_xx_im[99]=21.85;R_xx_re[100]=0.32; R_xx_im[100]=29.98;R_xx_re[101]=21.81; R_xx_im[101]=21.25;R_xx_re[102]=31.57; R_xx_im[102]=0.00;R_xx_re[103]=21.31; R_xx_im[103]=-21.82;R_xx_re[104]=-0.39; R_xx_im[104]=-30.36;R_xx_re[105]=-21.72; R_xx_im[105]=-21.33;R_xx_re[106]=-30.51; R_xx_im[106]=0.37;R_xx_re[107]=-21.75; R_xx_im[107]=21.40;R_xx_re[108]=0.47; R_xx_im[108]=30.76;R_xx_re[109]=21.84; R_xx_im[109]=21.32;R_xx_re[110]=30.55; R_xx_im[110]=-0.99;R_xx_re[111]=21.33; R_xx_im[111]=-22.04;
R_xx_re[112]=21.83; R_xx_im[112]=-21.79;R_xx_re[113]=-0.02; R_xx_im[113]=-30.74;R_xx_re[114]=-21.82; R_xx_im[114]=-21.75;R_xx_re[115]=-30.89; R_xx_im[115]=0.11;R_xx_re[116]=-21.49; R_xx_im[116]=21.50;R_xx_re[117]=0.14; R_xx_im[117]=30.75;R_xx_re[118]=21.31; R_xx_im[118]=21.82;R_xx_re[119]=32.22; R_xx_im[119]=0.00;R_xx_re[120]=21.67; R_xx_im[120]=-21.84;R_xx_re[121]=0.02; R_xx_im[121]=-30.86;R_xx_re[122]=-21.75; R_xx_im[122]=-21.70;R_xx_re[123]=-30.89; R_xx_im[123]=-0.49;R_xx_re[124]=-21.94; R_xx_im[124]=22.19;R_xx_re[125]=-0.03; R_xx_im[125]=30.74;R_xx_re[126]=22.37; R_xx_im[126]=21.38;R_xx_re[127]=31.04; R_xx_im[127]=-0.31;
R_xx_re[128]=30.68; R_xx_im[128]=0.16;R_xx_re[129]=21.71; R_xx_im[129]=-21.54;R_xx_re[130]=0.10; R_xx_im[130]=-30.67;R_xx_re[131]=-21.60; R_xx_im[131]=-21.77;R_xx_re[132]=-30.21; R_xx_im[132]=-0.19;R_xx_re[133]=-21.73; R_xx_im[133]=21.57;R_xx_re[134]=-0.39; R_xx_im[134]=30.36;R_xx_re[135]=21.67; R_xx_im[135]=21.84;R_xx_re[136]=32.20; R_xx_im[136]=0.00;R_xx_re[137]=21.79; R_xx_im[137]=-21.58;R_xx_re[138]=0.07; R_xx_im[138]=-30.57;R_xx_re[139]=-21.31; R_xx_im[139]=-22.26;R_xx_re[140]=-31.05; R_xx_im[140]=-0.00;R_xx_re[141]=-21.77; R_xx_im[141]=21.51;R_xx_re[142]=0.49; R_xx_im[142]=30.79;R_xx_re[143]=21.88; R_xx_im[143]=21.79;
R_xx_re[144]=21.81; R_xx_im[144]=21.75;R_xx_re[145]=30.73; R_xx_im[145]=-0.08;R_xx_re[146]=21.65; R_xx_im[146]=-21.83;R_xx_re[147]=-0.15; R_xx_im[147]=-30.99;R_xx_re[148]=-21.53; R_xx_im[148]=-21.40;R_xx_re[149]=-30.75; R_xx_im[149]=0.16;R_xx_re[150]=-21.72; R_xx_im[150]=21.33;R_xx_re[151]=0.02; R_xx_im[151]=30.86;R_xx_re[152]=21.79; R_xx_im[152]=21.58;R_xx_re[153]=32.72; R_xx_im[153]=0.00;R_xx_re[154]=21.63; R_xx_im[154]=-21.78;R_xx_re[155]=0.49; R_xx_im[155]=-30.98;R_xx_re[156]=-22.14; R_xx_im[156]=-21.79;R_xx_re[157]=-30.79; R_xx_im[157]=-0.09;R_xx_re[158]=-21.23; R_xx_im[158]=22.44;R_xx_re[159]=0.22; R_xx_im[159]=30.91;
R_xx_re[160]=-0.06; R_xx_im[160]=30.70;R_xx_re[161]=21.61; R_xx_im[161]=21.65;R_xx_re[162]=30.61; R_xx_im[162]=0.02;R_xx_re[163]=21.68; R_xx_im[163]=-21.79;R_xx_re[164]=0.13; R_xx_im[164]=-30.16;R_xx_re[165]=-21.68; R_xx_im[165]=-21.62;R_xx_re[166]=-30.51; R_xx_im[166]=-0.37;R_xx_re[167]=-21.75; R_xx_im[167]=21.70;R_xx_re[168]=0.07; R_xx_im[168]=30.57;R_xx_re[169]=21.63; R_xx_im[169]=21.78;R_xx_re[170]=32.60; R_xx_im[170]=0.00;R_xx_re[171]=22.01; R_xx_im[171]=-21.34;R_xx_re[172]=-0.19; R_xx_im[172]=-31.01;R_xx_re[173]=-21.65; R_xx_im[173]=-21.75;R_xx_re[174]=-30.63; R_xx_im[174]=0.60;R_xx_re[175]=-21.69; R_xx_im[175]=22.05;
R_xx_re[176]=-21.49; R_xx_im[176]=22.19;R_xx_re[177]=0.50; R_xx_im[177]=30.81;R_xx_re[178]=22.20; R_xx_im[178]=21.42;R_xx_re[179]=30.88; R_xx_im[179]=-0.65;R_xx_re[180]=21.18; R_xx_im[180]=-21.86;R_xx_re[181]=-0.54; R_xx_im[181]=-30.85;R_xx_re[182]=-21.75; R_xx_im[182]=-21.40;R_xx_re[183]=-30.89; R_xx_im[183]=0.49;R_xx_re[184]=-21.31; R_xx_im[184]=22.26;R_xx_re[185]=0.49; R_xx_im[185]=30.98;R_xx_re[186]=22.01; R_xx_im[186]=21.34;R_xx_re[187]=33.27; R_xx_im[187]=0.00;R_xx_re[188]=21.61; R_xx_im[188]=-22.55;R_xx_re[189]=-0.49; R_xx_im[189]=-30.68;R_xx_re[190]=-22.91; R_xx_im[190]=-21.01;R_xx_re[191]=-31.05; R_xx_im[191]=0.87;
R_xx_re[192]=-31.10; R_xx_im[192]=-0.12;R_xx_re[193]=-22.07; R_xx_im[193]=21.85;R_xx_re[194]=-0.08; R_xx_im[194]=31.10;R_xx_re[195]=21.98; R_xx_im[195]=22.06;R_xx_re[196]=30.72; R_xx_im[196]=0.20;R_xx_re[197]=22.02; R_xx_im[197]=-21.90;R_xx_re[198]=0.47; R_xx_im[198]=-30.76;R_xx_re[199]=-21.94; R_xx_im[199]=-22.19;R_xx_re[200]=-31.05; R_xx_im[200]=0.00;R_xx_re[201]=-22.14; R_xx_im[201]=21.79;R_xx_re[202]=-0.19; R_xx_im[202]=31.01;R_xx_re[203]=21.61; R_xx_im[203]=22.55;R_xx_re[204]=34.02; R_xx_im[204]=0.00;R_xx_re[205]=22.15; R_xx_im[205]=-21.83;R_xx_re[206]=-0.57; R_xx_im[206]=-31.22;R_xx_re[207]=-22.37; R_xx_im[207]=-21.93;
R_xx_re[208]=-21.77; R_xx_im[208]=-21.66;R_xx_re[209]=-30.63; R_xx_im[209]=0.10;R_xx_re[210]=-21.59; R_xx_im[210]=21.74;R_xx_re[211]=0.19; R_xx_im[211]=30.79;R_xx_re[212]=21.41; R_xx_im[212]=21.35;R_xx_re[213]=30.62; R_xx_im[213]=-0.22;R_xx_re[214]=21.84; R_xx_im[214]=-21.32;R_xx_re[215]=-0.03; R_xx_im[215]=-30.74;R_xx_re[216]=-21.77; R_xx_im[216]=-21.51;R_xx_re[217]=-30.79; R_xx_im[217]=0.09;R_xx_re[218]=-21.65; R_xx_im[218]=21.75;R_xx_re[219]=-0.49; R_xx_im[219]=30.68;R_xx_re[220]=22.15; R_xx_im[220]=21.83;R_xx_re[221]=33.20; R_xx_im[221]=0.00;R_xx_re[222]=21.18; R_xx_im[222]=-22.54;R_xx_re[223]=-0.50; R_xx_im[223]=-30.81;
R_xx_re[224]=0.72; R_xx_im[224]=-30.86;R_xx_re[225]=-21.25; R_xx_im[225]=-22.28;R_xx_re[226]=-30.77; R_xx_im[226]=-0.65;R_xx_re[227]=-22.32; R_xx_im[227]=21.38;R_xx_re[228]=-0.81; R_xx_im[228]=30.48;R_xx_re[229]=21.34; R_xx_im[229]=22.15;R_xx_re[230]=30.55; R_xx_im[230]=0.99;R_xx_re[231]=22.37; R_xx_im[231]=-21.38;R_xx_re[232]=0.49; R_xx_im[232]=-30.79;R_xx_re[233]=-21.23; R_xx_im[233]=-22.44;R_xx_re[234]=-30.63; R_xx_im[234]=-0.60;R_xx_re[235]=-22.91; R_xx_im[235]=21.01;R_xx_re[236]=-0.57; R_xx_im[236]=31.22;R_xx_re[237]=21.18; R_xx_im[237]=22.54;R_xx_re[238]=33.94; R_xx_im[238]=0.00;R_xx_re[239]=22.40; R_xx_im[239]=-21.59;
R_xx_re[240]=22.10; R_xx_im[240]=-21.77;R_xx_re[241]=0.20; R_xx_im[241]=-30.92;R_xx_re[242]=-21.74; R_xx_im[242]=-22.01;R_xx_re[243]=-31.03; R_xx_im[243]=-0.14;R_xx_re[244]=-21.82; R_xx_im[244]=21.32;R_xx_re[245]=-0.18; R_xx_im[245]=30.93;R_xx_re[246]=21.33; R_xx_im[246]=22.04;R_xx_re[247]=31.04; R_xx_im[247]=0.31;R_xx_re[248]=21.88; R_xx_im[248]=-21.79;R_xx_re[249]=0.22; R_xx_im[249]=-30.91;R_xx_re[250]=-21.69; R_xx_im[250]=-22.05;R_xx_re[251]=-31.05; R_xx_im[251]=-0.87;R_xx_re[252]=-22.37; R_xx_im[252]=21.93;R_xx_re[253]=-0.50; R_xx_im[253]=30.81;R_xx_re[254]=22.40; R_xx_im[254]=21.59;R_xx_re[255]=34.23; R_xx_im[255]=0.00;
    */
    /*30度
    R_xx_re[0]=7983.00000;R_xx_im[0]=+0.00000;     R_xx_re[1]=5.00000;R_xx_im[1]=-7975.00000;    R_xx_re[2]=-8011.00000;R_xx_im[2]=14.00000;   R_xx_re[3]=-5.00000;R_xx_im[3]=8004.00000;
    R_xx_re[4]=5.00000;R_xx_im[4]=975.00000;     R_xx_re[5]=8065.00000;R_xx_im[5]=0.00000;     R_xx_re[6]=-16.00000;R_xx_im[6]=-8037.00000;  R_xx_re[7]=-8017.00000;R_xx_im[7]=0.00000;
    R_xx_re[8]=-8011.00000;R_xx_im[8]=-14.00000;  R_xx_re[9]=-16.00000;R_xx_im[9]=8037.00000;   R_xx_re[10]=8191.00000;R_xx_im[10]=0.00000;     R_xx_re[11]=14.00000;R_xx_im[11]=-8051.00000;
    R_xx_re[12]=-5.00000;R_xx_im[12]=-8004.00000;   R_xx_re[13]=-8017.00000;R_xx_im[13]=-0.00000;   R_xx_re[14]=14.00000;R_xx_im[14]=8051.00000;    R_xx_re[15]=8216.00000;R_xx_im[15]=0.00000;
    //*/
    /*8x8矩陣 60度
    R_xx_re[0]=4164.0;R_xx_im[0]=0.0;       R_xx_re[1]=-3806.0;R_xx_im[1]=-1688.0;  R_xx_re[2]=2763.0;R_xx_im[2]=3092.0;    R_xx_re[3]=-1292.0;R_xx_im[3]=-3947.0;       R_xx_re[4]=-473.0;R_xx_im[4]=4134.0;       R_xx_re[5]=2086.0;R_xx_im[5]=-3574.0;   R_xx_re[6]=-3371.0;R_xx_im[6]=2434.0;   R_xx_re[7]=4087.0;R_xx_im[7]=-832.0;
    R_xx_re[8]=-3806.0;R_xx_im[8]=1688.0;   R_xx_re[9]=4190.0;R_xx_im[9]=0.0;       R_xx_re[10]=-3786.0;R_xx_im[10]=-1708.0;  R_xx_re[11]=2787.0;R_xx_im[11]=3091.0;         R_xx_re[12]=-1247.0;R_xx_im[12]=-3977.0;     R_xx_re[13]=-459.0;R_xx_im[13]=4120.0;    R_xx_re[14]=2099.0;R_xx_im[14]=-3596.0;   R_xx_re[15]=-3406.0;R_xx_im[15]=2419.0;
    R_xx_re[16]=2763.0;R_xx_im[16]=-3092.0;   R_xx_re[17]=-3786.0;R_xx_im[17]=1708.0;   R_xx_re[18]=4171.0;R_xx_im[18]=0.0;       R_xx_re[19]=-3796.0;R_xx_im[19]=-1664.0;       R_xx_re[20]=2764.0;R_xx_im[20]=3103.0;       R_xx_re[21]=-1271.0;R_xx_im[21]=-3926.0;  R_xx_re[22]=-431.0;R_xx_im[22]=4126.0;    R_xx_re[23]=2099.0;R_xx_im[23]=-3590.0;
    R_xx_re[24]=-1292.0;R_xx_im[24]=3947.0;   R_xx_re[25]=2787.0;R_xx_im[25]=-3091.0;   R_xx_re[26]=-3796.0;R_xx_im[26]=1664.0;   R_xx_re[27]=4196.0;R_xx_im[27]=0.0;            R_xx_re[28]=-3778.0;R_xx_im[28]=-1732.0;     R_xx_re[29]=2746.0;R_xx_im[29]=3088.0;    R_xx_re[30]=-1264.0;R_xx_im[30]=-3956.0;  R_xx_re[31]=-480.0;R_xx_im[31]=4139.0;
    R_xx_re[32]=-473.0;R_xx_im[32]=-4134.0;   R_xx_re[33]=-1247.0;R_xx_im[33]=3977.0;   R_xx_re[34]=2764.0;R_xx_im[34]=-3103.0;   R_xx_re[35]=-3778.0;R_xx_im[35]=1732.0;        R_xx_re[36]=4226.0;R_xx_im[36]=0.0;          R_xx_re[37]=-3793.0;R_xx_im[37]=-1666.0;  R_xx_re[38]=2805.0;R_xx_im[38]=3072.0;    R_xx_re[39]=-1287.0;R_xx_im[39]=-3966.0;
    R_xx_re[40]=2086.0;R_xx_im[40]=3574.0;    R_xx_re[41]=-459.0;R_xx_im[41]=-4120.0;   R_xx_re[42]=-1271.0;R_xx_im[42]=3926.0;   R_xx_re[43]=2746.0;R_xx_im[43]=-3088.0;        R_xx_re[44]=-3793.0;R_xx_im[44]=1666.0;      R_xx_re[45]=4189.0;R_xx_im[45]=0.0;       R_xx_re[46]=-3783.0;R_xx_im[46]=-1676.0;  R_xx_re[47]=2765.0;R_xx_im[47]=3094.0;
    R_xx_re[48]=-3371.0;R_xx_im[48]=-2434.0;  R_xx_re[49]=2099.0;R_xx_im[49]=3596.0;    R_xx_re[50]=-431.0;R_xx_im[50]=-4126.0;   R_xx_re[51]=-1264.0;R_xx_im[51]=3956.0;        R_xx_re[52]=2805.0;R_xx_im[52]=-3072.0;      R_xx_re[53]=-3783.0;R_xx_im[53]=1676.0;   R_xx_re[54]=4240.0;R_xx_im[54]=0.0;       R_xx_re[55]=-3802.0;R_xx_im[55]=-1717.0;
    R_xx_re[56]=4087.0;R_xx_im[56]=832.0;     R_xx_re[57]=-3406.0;R_xx_im[57]=-2419.0;  R_xx_re[58]=2099.0;R_xx_im[58]=3590.0;    R_xx_re[59]=-480.0;R_xx_im[59]=-4139.0;        R_xx_re[60]=-1287.0;R_xx_im[60]=3966.0;      R_xx_re[61]=2765.0;R_xx_im[61]=-3094.0;   R_xx_re[62]=-3802.0;R_xx_im[62]=1717.0;   R_xx_re[63]=4280.0;R_xx_im[63]=0.0;
    */
    /*
    R_xx_re[0]=15709; R_xx_im[0]=0;R_xx_re[1]=-14302; R_xx_im[1]=-6402;R_xx_re[2]=10491; R_xx_im[2]=11628;R_xx_re[3]=-4680; R_xx_im[3]=-14838;
    R_xx_re[4]=-14302; R_xx_im[4]=6402;R_xx_re[5]=15820; R_xx_im[5]=0;R_xx_re[6]=-14333; R_xx_im[6]=-6345;R_xx_re[7]=10346; R_xx_im[7]=11656;
    R_xx_re[8]=10491; R_xx_im[8]=-11628;R_xx_re[9]=-14333; R_xx_im[9]=6345;R_xx_re[10]=15914; R_xx_im[10]=0;R_xx_re[11]=-14162; R_xx_im[11]=-6474;
    R_xx_re[12]=-4680; R_xx_im[12]=14838;R_xx_re[13]=10346; R_xx_im[13]=-11656;R_xx_re[14]=-14162; R_xx_im[14]=6474;R_xx_re[15]=15808; R_xx_im[15]=0;
    */
    printf("\n");
    timeMusicre_start = clock();
    // compute eigenvector Ve (M, M) 
    //---------------------------------------------------------------
    float *Ve_re = (float *)malloc(M * M * sizeof(float));
    float *Ve_im = (float *)malloc(M * M * sizeof(float));
    float *De_re = (float *)malloc(M * M * sizeof(float));
    float *De_im = (float *)malloc(M * M * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Eigen_start, NULL);
    hybrid_eigen(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, M, M, qr_iter);
    gettimeofday(&time_Eigen_end, NULL);
    //print_complex_matrix_R_xx(R_xx_re, R_xx_im, M, M);
    //printf("----------Ve------------\n");
    //print_complex_matrix(Ve_re, Ve_im, M, M);
    //printf("----------De------------\n");
    //print_complex_matrix_i(De_re, De_im, M, M);
    /*
    //---------------------------------------------------------------
    int16_t *vet_noise_re_i = (int16_t *)malloc(M * (M - len_t_theta) * sizeof(int16_t));
    int16_t *vet_noise_im_i = (int16_t *)malloc(M * (M - len_t_theta) * sizeof(int16_t));
    //---------------------------------------------------------------

    for (int16_t i = 0; i < M; ++i)
    {
        for (int16_t j = len_t_theta; j < M; ++j)
        {
            vet_noise_re_i[i * (M - len_t_theta) + j - len_t_theta] = Ve_re[i * M + j];
            vet_noise_im_i[i * (M - len_t_theta) + j - len_t_theta] = Ve_im[i * M + j];
            //printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
        }
    }
    */
    //---------------------------------------------------------------
    float *vet_noise_re = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    float *vet_noise_im = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    //---------------------------------------------------------------

    for (int i = 0; i < M; ++i)
    {
        for (int j = len_t_theta; j < M; ++j)
        {
            vet_noise_re[i * (M - len_t_theta) + j - len_t_theta] = Ve_re[i * M + j];
            vet_noise_im[i * (M - len_t_theta) + j - len_t_theta] = Ve_im[i * M + j];
            //printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
        }
    }

    //---------------------------------------------------------------//calloc包含初始化
    //---------------------------------------------------------------
    float *Pn_re = (float *)calloc(M * M, sizeof(float));
    float *Pn_im = (float *)calloc(M * M, sizeof(float));
    //---------------------------------------------------------------
    int16_t *Pn_re_i = (int16_t *)calloc(M * M, sizeof(int16_t));
    int16_t *Pn_im_i = (int16_t *)calloc(M * M, sizeof(int16_t));
    //---------------------------------------------------------------
    compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    
    /*
    gettimeofday(&time_Pn_start, NULL);
    compute_Pn_i(Pn_re_i, Pn_im_i, vet_noise_re_i, vet_noise_im_i, M, len_t_theta);
    for(int w=0;w<M*M;w++){
        Pn_re[w]=Pn_re_i[w];
        Pn_im[w]=Pn_im_i[w];
    }
    gettimeofday(&time_Pn_end, NULL);
    */

    /*
    for (int i = 0; i < M * M; ++i)// Q_temp的100倍給vet_noise vet_noise再共軛轉置相乘=Pn變10000倍
    {
        Pn_re_f[i] = Pn_re_f[i]/10000;
        Pn_im_f[i] = Pn_im_f[i]/10000;
        //printf("(%f,%f) ", Pn_re_f[i], Pn_im_f[i]);
    }
    */
    // array pattern
    // parameter setting
    // const int len_dth = 401;
    
    const int len_dth = 1201;
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    //---------------------------------------------------------------
    float *a_vector_re = (float *)malloc(M * sizeof(float));
    float *a_vector_im = (float *)malloc(M * sizeof(float));
    float *S_MUSIC_re = (float *)malloc(len_dth * sizeof(float));
    float *S_MUSIC_im = (float *)malloc(len_dth * sizeof(float));
    float *S_MUSIC_dB = (float *)malloc(len_dth * sizeof(float));
    //---------------------------------------------------------------
    FILE *fp_excel = NULL;
    fp_excel = fopen("data/2D_MUSIC_dB.csv", "w");
    for (int i = 0; i < len_dth; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector_re, a_vector_im, Pn_re, Pn_im, M, &S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSIC_dB[i] = cpp_20log_abs(&S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("S_MUSIC_dB = %.4f\n", S_MUSIC_dB[i]);

        fprintf(fp_excel, "%.1f,%.4f\n", (-90 + 0.1 * i), S_MUSIC_dB[i]);
    }
    fclose(fp_excel);

    //---------------------------------------------------------------
    // find Max and position
    double max_temp = S_MUSIC_dB[0];
    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        if (S_MUSIC_dB[i] > max_temp)
        {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }
    }
    timeMusicre_end = clock();
    //-------------------------------------------------------------------
    // timersub function
    //-------------------------------------------------------------------
    float time_AWGN, time_Eigen, time_Pn;                           // create float parameter in order to convert (us) to (ms)
    timersub(&time_AWGN_end, &time_AWGN_start, &time_AWGN_diff);    // calculate AWGN
    timersub(&time_Eigen_end, &time_Eigen_start, &time_Eigen_diff); // calculate Eigen
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);          // calculate Pn
    time_AWGN = time_AWGN_diff.tv_usec;
    time_Eigen = time_Eigen_diff.tv_usec;
    time_Pn = time_Pn_diff.tv_usec;
    printf("Total AWGN time: \t%.3f(ms)\n", time_AWGN / 1000); 
    printf("Total Eigen time: \t%.3f(ms)\n", time_Eigen / 1000);
    printf("Total Pn time: \t\t%.3f(ms)\n", time_Pn / 1000);
    //-------------------------------------------------------------------
    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);

    printf(L_GREEN "Total MUSIC REAL time : \t%.3f(ms)\n" CLOSE, (timeMusicre_end - timeMusicre_start) / CLOCKS_PER_SEC * 1000);
}



int main()
{
    //-------------------------------------------------------------------
    // Parameter initialize
    float time_MUSIC = 0.0;
    float time_MVDR = 0.0;
    // struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // MUSIC time initial
    // struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;    // MVDR time initial

    float timeMusic_start, timeMusic_end; // Total MUSIC Algorithm time
    float timeMVDR_start, timeMVDR_end;   // Total MVDR Algorithm time
    //-------------------------------------------------------------------
    float angle[100] = {10, -47, -28, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 32;
    int snr = 10;
    int qr_iter =1;
    float result[3] = {0};
    // int angle = 50;
    //int iter = 10;

    //=================== MUSIC Algorithm =================================
    timeMusic_start = clock();
    // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
    MUSIC_DOA_2A_CPU_test(M, qr_iter, &angle[0],number_angle, result, snr);
    timeMusic_end = clock();
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MUSIC time : \t%.3f(ms)\n" CLOSE, (timeMusic_end - timeMusic_start) / CLOCKS_PER_SEC * 1000);
    printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time / 1000);
    total_multiply_time = 0;      // set to 0
    total_pre_transpose_time = 0; // set to 0
    //=====================================================================

}