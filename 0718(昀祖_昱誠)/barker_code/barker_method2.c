// 生成127bit碼，用sliding window確認有沒有對上，生成部分來源:barker/pn_generate.c
// MUSIC，方法2，對上那次&前4次、後4次做DOA
// 演算法來源:int32/hybrid_MQS_QR_prune2.c
// g++ -mavx512f -g -o barker_method2  barker_method2.c -Wall -Wextra -std=c++14 math_func.a
// ./barker_method2
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CODE_LENGTH 127

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
//-------------------------------------------
#define CODE_LENGTH 127
void print_code(int code[],int a){
    for(int i=0;i<a;i++){
        printf("%d",code[i]);
    }
    printf("\n");
}

// 檢查 pn code 是否與目標序列相同的函式
int checkpnCode(int code[], int target[],int* check_result) {
    int mul=0;
    
    //print_code(code,CODE_LENGTH);
    //print_code(target,CODE_LENGTH);
    for (int i = 0; i < CODE_LENGTH; i++) {
        mul=mul+code[i]*target[i];
    }
    //printf("mul=%d ",mul);
    if(mul==CODE_LENGTH){
        printf("matched\n");
        *check_result=1;
        //check_result=&x;
    }
    else{
        printf("not match\n");
        //check_result=&y;
        *check_result=0;
    }

}
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
            printf("\t%.5f", matA_re[i * colA + j]);
            printf("+%.5fi", matA_im[i * colA + j]);
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
//沒AVX的加法 時間:32ms
/*
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
*/
//AVX加法
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{   
    
    __m512 a_re,a_im,b_re,b_im,c_re,c_im;
    for (int i = 0; i <rowA * colA ; i += 16)
    {
        a_re = _mm512_loadu_ps(&matA_re[i]);
        a_im = _mm512_loadu_ps(&matA_im[i]);
        b_re = _mm512_loadu_ps(&matB_re[i]);
        b_im = _mm512_loadu_ps(&matB_im[i]);

        // Perform vectorized addition for real and imaginary parts
        c_re = _mm512_add_ps(a_re, b_re);
        c_im = _mm512_add_ps(a_im, b_im);

        // Store the results back to memory
        _mm512_storeu_ps(&matA_re[i],c_re);
        _mm512_storeu_ps(&matA_im[i],c_im);

        
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
//沒AVX的減法
/*
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
*/
//AVX減法
void complex_matrix_subtraction(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{   
    if(rowA<16){
        for (int i = 0; i < rowA; ++i)
        {
            for (int j = 0; j < colA; ++j)
            {
            matA_re[i * colA + j] -= matB_re[i * colA + j];
            matA_im[i * colA + j] -= matB_im[i * colA + j];
            }
        }
    }
    else{
        __m512 a_re,a_im,b_re,b_im,c_re,c_im;
        for (int i = 0; i <rowA * colA ; i += 16)
        {
            a_re = _mm512_loadu_ps(&matA_re[i]);
            a_im = _mm512_loadu_ps(&matA_im[i]);
            b_re = _mm512_loadu_ps(&matB_re[i]);
            b_im = _mm512_loadu_ps(&matB_im[i]);

            // Perform vectorized sub for real and imaginary parts
            c_re = _mm512_sub_ps(a_re, b_re);
            c_im = _mm512_sub_ps(a_im, b_im);

            // Store the results back to memory
            _mm512_storeu_ps(&matA_re[i],c_re);
            _mm512_storeu_ps(&matA_im[i],c_im);
        
        }
    }
    // Loop over blocks of 8 elements (AVX-512 can process 8 single-precision floats at once)
    

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
void complex_matrix_multiplication_noavx(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    memset(matC_re, 0, rowA * colB * sizeof(float));
    memset(matC_im, 0, rowA * colB * sizeof(float));

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
void complex_matrix_conjugate_transpose_multiplication_noavx(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    complex_matrix_conjugate_transpose(temp_re, temp_im, rowA, colA);
    complex_matrix_multiplication_noavx(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);

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
        complex_matrix_conjugate_transpose_multiplication_noavx(vet_noise_temp_re, vet_noise_temp_im, Pn_temp_re, Pn_temp_im, M, 1);
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
        int a3=8;
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
/*
void hybrid_qr(int *A_re, int *A_im, int *Q_re, int *Q_im, int *R_re, int *R_im, int row, int col)
{
    int X1 = 64;   //原始輸入放大X1倍，配合Rxx的放大記得改
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
    //print_complex_matrix_i(A_re, A_im, row, col ); //
    //printf(YELLOW"---------\n"CLOSE);
    //float
    for (int i = 0; i < 2; ++i)
    {
        //printf(YELLOW"-----i=(%d)----\n"CLOSE,i);
        //printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        //printf("一開始Q:\n");
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
        //

        
        //printf("歸一化後(調整過)q,(%d): X1倍 \n",i);
        //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
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
            //printf("v:Q_col_proj縮小前(%d):\n",i);
            //print_complex_matrix(Q_col_proj_re_f, Q_col_proj_im_f, row , col-(i+1));
            
            
            for(int w=0;w<row*size;w++){ //乘法 縮小X1
                Q_col_proj_re_f[w]=Q_col_proj_re_f[w]/X1;
                Q_col_proj_im_f[w]=Q_col_proj_im_f[w]/X1;
            }   
            
            //rintf("v:Q_col_proj(%d):\n",i);
            //print_complex_matrix(Q_col_proj_re_f, Q_col_proj_im_f, row , col-(i+1));
            complex_matrix_multiplication(Q_col_re_f, Q_col_im_f, Q_col_proj_re_f, Q_col_proj_im_f, proj_vector_re_f, proj_vector_im_f, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector 給R右上\n");
            //print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1));
            
            for (int j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re_f[j-(i+1)]; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im_f[j-(i+1)];
            }
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose( Q_col_re_f, Q_col_im_f, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("q(%d):\n",i);
            //print_complex_matrix(Q_col_re_f, Q_col_im_f, row, 1);
            
            
            //printf("q*v:r:proj_vector放大a2 :\n");
            //print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1)); 
            for(int w=0;w<col-(i+1);w++){ //乘法 縮小X
                proj_vector_re_f[w]=proj_vector_re_f[w]/X1;
                proj_vector_im_f[w]=proj_vector_im_f[w]/X1;
            }
            //printf("q*v:r:proj_vector\n");
            //print_complex_matrix(proj_vector_re_f, proj_vector_im_f, 1, col-(i+1)); 
            complex_matrix_multiplication( Q_col_re_f, Q_col_im_f, proj_vector_re_f, proj_vector_im_f, Q_sub_re_f, Q_sub_im_f, row, 1, col-(i+1));// row*col-(i+1)
            //printf("Q_sub=多個r*q = Q_col x proj_vector\n");
            //print_complex_matrix(Q_sub_re_f, Q_sub_im_f,  row, col-(i+1));
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {    
                //printf(RED"進for分別減,i=%d,j=%d\n"CLOSE,i,j);
                complex_matrix_get_columns(Q_re_f, Q_im_f, vector_cur_re_f, vector_cur_im_f, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns(Q_sub_re_f, Q_sub_im_f, Q_col_re_f, Q_col_im_f, row, col-(i+1), j-(i+1));
                
                
                //printf("調整過的vector_cur減Q_col\n");
                //printf("vector_cur:\n");
                //print_complex_matrix(vector_cur_re_f, vector_cur_im_f,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix(Q_col_re_f, Q_col_im_f,  row, 1);
                complex_matrix_subtraction(vector_cur_re_f, vector_cur_im_f, Q_col_re_f, Q_col_im_f, row, 1);
                
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
    //printf(BLUE"Q:\n"CLOSE);
    //print_complex_matrix_i(Q_re, Q_im, row, col );
    
    for (int i = 2; i < col; ++i)
    {
        //printf(YELLOW"-----i=(%hd)----\n"CLOSE,i);
        //printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        //printf("一開始Q:\n");
        //print_complex_matrix_i(Q_re, Q_im, row, col );
        //complex_matrix_get_columns_i(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i); //Q給Q_col相當於 v(i) : row*1
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
        
        int a3=8;
        //*把Q_col放大
        for(int w=0;w<col;w++){
            Q_col_re[w]=Q_col_re_f[w]*a3;
            Q_col_im[w]=Q_col_im_f[w]*a3;
        }
        //
        complex_matrix_conjugate_transpose_i32( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_re,Q_im(%hd):\n",i);
        //print_complex_matrix_i(Q_re, Q_im, row, col);
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
            //print_complex_matrix(Q_re, Q_im, row, col );   
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
            //print_complex_matrix_i(Q_re, Q_im, row, col);
            //printf("q:Q_col^H\n",i);
            //print_complex_matrix_i(Q_col_re, Q_col_im, 1, row);
            //printf("v:Q_col_proj縮小前\n");
            //print_complex_matrix_i(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            
            //* 讓Q_col_proj縮小別太小 total三塊程式要改(含此塊)[備註以免漏掉]
            int a=64;
            if(i>0){
                for(int w=0;w<row*size;w++){ //乘法 放大a
                    Q_col_proj_re[w]=Q_col_proj_re[w]*a;
                    Q_col_proj_im[w]=Q_col_proj_im[w]*a;
                }
            }
            //printf("if i>0才做,v:Q_col_proj先放大(xa):\n");
            //print_complex_matrix_i(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            //

            for(int w=0;w<row*size;w++){ //乘法 縮小X
                Q_col_proj_re[w]=Q_col_proj_re[w]/X1;
                Q_col_proj_im[w]=Q_col_proj_im[w]/X1;
            } 
            //printf("v:Q_col_proj:\n");
            

            //print_complex_matrix_i(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            complex_matrix_multiplication_iii32(Q_col_re, Q_col_im, Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector\n");
            //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));

            //配合讓Q_col_proj別太小 

            for(int w=0;w<size;w++){
                proj_vector_re[w]=proj_vector_re[w]/a;
                proj_vector_im[w]=proj_vector_im[w]/a;
            }
                //printf("q*v:r:Q_col x Q_col_proj = proj_vector ,給R右上,再縮小a的 \n");
                //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));
            
            
            //配合把Q_col放大
            for (int j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re[j-(i+1)]/a3; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im[j-(i+1)]/a3;
            }
            //
            
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose_i32( Q_col_re, Q_col_im, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("Q_col(%hd):\n",i);
            //print_complex_matrix_i(Q_col_re, Q_col_im, row, 1);
            
            int a2=16;
            //讓proj_vector縮小別太小 total四塊程式要改(含此塊)[備註以免漏掉]
            for(int w=0;w<col-(i+1);w++){ 
                proj_vector_re[w]=proj_vector_re[w]*a2;
                proj_vector_im[w]=proj_vector_im[w]*a2;
            }
            //
            //printf("q*v:r:proj_vector放大(*a2:%d)\n",a2);
            //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));
            for(int w=0;w<col-(i+1);w++){ //乘法 縮小X
                proj_vector_re[w]=proj_vector_re[w]/X1;
                proj_vector_im[w]=proj_vector_im[w]/X1;
            }
            //printf("q*v:r:proj_vector 最終:\n");


            //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));
            complex_matrix_multiplication_iii32( Q_col_re, Q_col_im, proj_vector_re, proj_vector_im, Q_sub_re, Q_sub_im, row, 1, col-(i+1));// row*col-(i+1)
            //printf("Q_sub=多個r*q = Q_col x proj_vector\n");
            //print_complex_matrix_i(Q_sub_re, Q_sub_im,  row, col-(i+1));
            int a33=a3*a3;
            //配合除法後把Q_col放大
            for(int w=0;w<row*size;w++){
                Q_sub_re[w]=Q_sub_re[w]/a33;
                Q_sub_im[w]=Q_sub_im[w]/a33;
            }
            //
            //printf("最終Q_sub:\n");
            //print_complex_matrix_i(Q_sub_re, Q_sub_im,  row, col-(i+1));
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {   
                //printf(RED"進for分別減,i=%hd,j=%hd\n"CLOSE,i,j);
                complex_matrix_get_columns_i32(Q_re, Q_im, vector_cur_re, vector_cur_im, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns_i32(Q_sub_re, Q_sub_im, Q_col_re, Q_col_im, row, col-(i+1), j-(i+1));
                //printf("vector_cur放大前:\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                // 配合讓proj_vector縮小別太小
                
                for(int m=0;m<col;m++){
                    vector_cur_re[m]=vector_cur_re[m]*a2;
                    vector_cur_im[m]=vector_cur_im[m]*a2;
                }
                //

                //printf("調整過\n");
                //printf("vector_cur:\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix_i(Q_col_re, Q_col_im,  row, 1);
                complex_matrix_subtraction_i32(vector_cur_re, vector_cur_im, Q_col_re, Q_col_im, row, 1);

                //printf("減完的vector_cur減完\n");
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
                // 配合讓proj_vector縮小別太小
                for(int m=0;m<col;m++){
                    vector_cur_re[m]=vector_cur_re[m]/a2;
                    vector_cur_im[m]=vector_cur_im[m]/a2;
                } 
                //

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
*/


void hybrid_qr(int *A_re, int *A_im, int *Q_re, int *Q_im, int *R_re, int *R_im, int row, int col)
{
    int X1 = 128;   //原始輸入放大X1倍，配合Rxx的放大記得改
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
    //print_complex_matrix_i(A_re, A_im, row, col ); //
    //printf(YELLOW"---------\n"CLOSE);
    //float
    //printf(BLUE"Q:\n"CLOSE);
    //print_complex_matrix_i(Q_re, Q_im, row, col );
    
    for (int i = 0; i < col; ++i)
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
        //printf("除法*X1後\n");
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
        
        int a3=64;
        //*把Q_col放大
        for(int w=0;w<col;w++){
            Q_col_re[w]=Q_col_re_f[w]*a3;
            Q_col_im[w]=Q_col_im_f[w]*a3;
        }
        //*/
        complex_matrix_conjugate_transpose_i32( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_re,Q_im(%hd):\n",i);
        //print_complex_matrix_i32(Q_col_re, Q_col_im, row, 1);
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
            //printf("v:Q_col_proj縮小前\n");
            //print_complex_matrix_i32(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            
            //* 讓Q_col_proj縮小別太小 total三塊程式要改(含此塊)[備註以免漏掉]
            int a=64;
            //if(i>0){
                for(int w=0;w<row*size;w++){ //乘法 放大a
                    Q_col_proj_re[w]=Q_col_proj_re[w]*a;
                    Q_col_proj_im[w]=Q_col_proj_im[w]*a;
                }
            //}
            //printf("v:Q_col_proj先放大(xa):\n");
            //print_complex_matrix_i32(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            //*/

            for(int w=0;w<row*size;w++){ //乘法 縮小X
                Q_col_proj_re[w]=Q_col_proj_re[w]/X1;
                Q_col_proj_im[w]=Q_col_proj_im[w]/X1;
            } 
            //printf("v:Q_col_proj:\n");
            

            //print_complex_matrix_i(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            complex_matrix_multiplication_iii32(Q_col_re, Q_col_im, Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector\n");
            //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));

            //*配合讓Q_col_proj別太小 

            for(int w=0;w<size;w++){
                proj_vector_re[w]=proj_vector_re[w]/a;
                proj_vector_im[w]=proj_vector_im[w]/a;
            }
                //printf("q*v:r:Q_col x Q_col_proj = proj_vector ,給R右上,再縮小a的 \n");
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
            
            int a2=4;
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


            //print_complex_matrix_i(proj_vector_re, proj_vector_im, 1, col-(i+1));
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
                //print_complex_matrix_i(vector_cur_re, vector_cur_im,  row, 1);
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
        A_re_i[w]=round(A_re[w]*128);
        A_im_i[w]=round(A_im[w]*128);
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



void MUSIC_DOA_2A_CPU_test(int M, int qr_iter,int hybrid_qr_iter, float *angle, int number_angle, float *result, int SNR, int *check_result)
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
    printf("check_result=%d\n",*check_result);
    const int fc = 2.5e+9;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d;
    if(*check_result == 1){
        d = lemda * 0.25;
    }
    else{
        d = 0;
    }
    //float d = lemda * 0.25;
    float kc = 2 * PI / lemda;
    const int nd = 512;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        printf("%.4f, ", angle[a]);
    }
    printf("\n");
    //---------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    float *A_theta4_re = (float *)malloc(4 * len_t_theta * sizeof(float));
    float *A_theta4_im = (float *)malloc(4 * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
           // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta4_re[i * len_t_theta + j], &A_theta4_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
           // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
        }
    }
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
    float *sig_co4_re = (float *)malloc(4 * nd * sizeof(float));
    float *sig_co4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    float *x_r4_re = (float *)malloc(4 * nd * sizeof(float));
    float *x_r4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    complex_matrix_multiplication(A_theta4_re, A_theta4_im, t_sig_re, t_sig_im, sig_co4_re, sig_co4_im, 4, len_t_theta, nd);
    //print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    //gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    cpp_awgn(sig_co4_re, sig_co4_im, x_r4_re, x_r4_im, SNR, 4, nd);
    //gettimeofday(&time_AWGN_end, NULL);
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    float *R_xx_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_im = (float *)malloc(M * M * sizeof(float));
    float M_re = M;
    float M_im = 0.0;
    float *M_ptr = &M_re;
    float *M_ptr_im = &M_im;
    //---------------------------------------------------------------
    float *R_xx4_re = (float *)malloc(4 * 4 * sizeof(float));
    float *R_xx4_im = (float *)malloc(4 * 4 * sizeof(float));
    float M4_re = 4;
    float M4_im = 0.0;
    float *M_ptr4 = &M4_re;
    float *M_ptr4_im = &M4_im;
    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int i = 0; i < M * M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r4_re, x_r4_im, R_xx4_re, R_xx4_im, 4, nd);
    for (int i = 0; i < 4 * 4; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx4_re[i], &R_xx4_im[i], M_ptr4, M_ptr4_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    // compute eigenvector Ve (M, M)
    //---------------------------------------------------------------
    //print_complex_matrix_R_xx(R_xx_re, R_xx_im, M, M);
    timeMusicre_start = clock();
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    float *Ve_re = (float *)malloc(M * M * sizeof(float));
    float *Ve_im = (float *)malloc(M * M * sizeof(float));
    float *De_re = (float *)malloc(M * M * sizeof(float));
    float *De_im = (float *)malloc(M * M * sizeof(float));
    //---------------------------------------------------------------
    float *Ve4_re = (float *)malloc(4 * 4 * sizeof(float));
    float *Ve4_im = (float *)malloc(4 * 4 * sizeof(float));
    float *De4_re = (float *)malloc(4 * 4 * sizeof(float));
    float *De4_im = (float *)malloc(4 * 4 * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Eigen_start, NULL);
    eigen(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, M, M, qr_iter);
    eigen(R_xx4_re, R_xx4_im, Ve4_re, Ve4_im, De4_re, De4_im, 4, 4, qr_iter);
    //gettimeofday(&time_Eigen_end, NULL);
    gettimeofday(&time_Eigen_end, NULL);
    //printf("----------Ve------------\n");
    //print_complex_matrix(Ve_re, Ve_im, M, M);
    //printf("----------De------------\n");
    //print_complex_matrix(De_re, De_im, M, M);
    //---------------------------------------------------------------
    float *vet_noise_re = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    float *vet_noise_im = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    float *vet_noise4_re = (float *)malloc(4 * (4 - len_t_theta) * sizeof(float));
    float *vet_noise4_im = (float *)malloc(4 * (4 - len_t_theta) * sizeof(float));
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
    for (int i = 0; i < 4; ++i)
    {
        for (int j = len_t_theta; j < 4; ++j)
        {
            vet_noise4_re[i * (4 - len_t_theta) + j - len_t_theta] = Ve4_re[i * 4 + j];
            vet_noise4_im[i * (4 - len_t_theta) + j - len_t_theta] = Ve4_im[i * 4 + j];
            //printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
        }
    }
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    float *Pn_re = (float *)calloc(M * M, sizeof(float));
    float *Pn_im = (float *)calloc(M * M, sizeof(float));
    //---------------------------------------------------------------
    float *Pn4_re = (float *)calloc(4 * 4, sizeof(float));
    float *Pn4_im = (float *)calloc(4 * 4, sizeof(float));
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    compute_Pn(Pn4_re, Pn4_im, vet_noise4_re, vet_noise4_im, 4, len_t_theta);
    //gettimeofday(&time_Pn_end, NULL);
    gettimeofday(&time_Pn_end, NULL);
    //---------------------------------------------------------------
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, M, M);
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    // parameter setting
    const int len_dth = 11; //1度
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    double max_temp ;
    int position = 0;
    const int len_dthA = 11;  //0.1度                                
    float *dthA = (float *)malloc(len_dthA * sizeof(float)); 
    float *drA = (float *)malloc(len_dthA * sizeof(float));
    double max_tempA ;
    int positionA = 0;
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -5 + 1 * i; 
        dr[i] = dth[i] * PI / 180;
    }
    //---------------------------------------------------------------
    float *a_vector4_re = (float *)malloc(4 * sizeof(float));
    float *a_vector4_im = (float *)malloc(4 * sizeof(float));
    float *S_MUSIC_re = (float *)malloc(len_dth * sizeof(float));
    float *S_MUSIC_im = (float *)malloc(len_dth * sizeof(float));
    float *S_MUSIC_dB = (float *)malloc(len_dth * sizeof(float));
    //---------------------------------------------------------------
    //FILE *fp_excel = NULL;
    //fp_excel = fopen("data/2D_MUSIC_dB.csv", "w");
    for (int i = 0; i < len_dth; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < 4; ++j)
        {
            cpp_exp2(&a_vector4_re[j], &a_vector4_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector4_re, a_vector4_im, Pn4_re, Pn4_im, 4, &S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSIC_dB[i] = cpp_20log_abs(&S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("S_MUSIC_dB = %.4f\n", S_MUSIC_dB[i]);

        //fprintf(fp_excel, "%.1f,%.4f\n", (-60 + 30 * i), S_MUSIC_dB[i]);
    }
    //fclose(fp_excel);
    //---------------------------------------------------------------
    // find Max and position
    max_temp = S_MUSIC_dB[0];
    for (int i = 0; i < len_dth; ++i)
    {
        //printf("S_MUSIC_dB[%d] = %.4f\n",i , S_MUSIC_dB[i]);
        if (S_MUSIC_dB[i] > max_temp)
        {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }

    }
    //---------------------------------------------------------------
    for (int i = 0; i < len_dthA; ++i)  
    {  
        dthA[i] = (dth[position]-0.5) + 0.1 * i;
            //printf("dth1:%f\n",dthA[i]);
        drA[i] = dthA[i] * PI / 180;
    }
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    float *a_vectorA_re = (float *)malloc(M * sizeof(float));
    float *a_vectorA_im = (float *)malloc(M * sizeof(float));
    float *S_MUSICA_re = (float *)malloc(len_dthA * sizeof(float));
    float *S_MUSICA_im = (float *)malloc(len_dthA * sizeof(float));
    float *S_MUSICA_dB = (float *)malloc(len_dthA * sizeof(float));
    //---------------------------------------------------------------
        for (int i = 0; i < len_dthA; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vectorA_re[j], &a_vectorA_im[j], drA, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vectorA_re, a_vectorA_im, Pn_re, Pn_im, M, &S_MUSICA_re[i], &S_MUSICA_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSICA_dB[i] = cpp_20log_abs(&S_MUSICA_re[i], &S_MUSICA_im[i]);
        //printf("S_MUSIC1_dB = %.4f\n", S_MUSIC1_dB[i]);
    }
    //printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    max_tempA = S_MUSICA_dB[0];
    for (int i = 0; i < len_dthA; ++i)
    {
        if (S_MUSICA_dB[i] > max_tempA)
        {
            max_tempA = S_MUSICA_dB[i];
            positionA = i;
        }
        //printf("max_temp1 = %.4f,(%d)\n", max_temp1,i);
    }
    timeMusicre_end = clock();
    // timersub function
    //-------------------------------------------------------------------
    float time_AWGN, time_Eigen, time_Pn;                           // create float parameter in order to convert (us) to (ms)
    timersub(&time_AWGN_end, &time_AWGN_start, &time_AWGN_diff);    // calculate AWGN
    timersub(&time_Eigen_end, &time_Eigen_start, &time_Eigen_diff); // calculate Eigen
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);          // calculate Pn
    time_AWGN = time_AWGN_diff.tv_usec;
    time_Eigen = time_Eigen_diff.tv_usec;
    time_Pn = time_Pn_diff.tv_usec;
    //printf("Total AWGN time: \t%.3f(ms)\n", time_AWGN / 1000);
    printf("Total Eigen time: \t%.3f(ms)\n", time_Eigen / 1000);
    printf("Total Pn time: \t\t%.3f(ms)\n", time_Pn / 1000);
    //-------------------------------------------------------------------
    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    printf("positionA : \t\t%d\n", positionA);
    printf(RED "Theta estimationA :\t%.4f (degree)\n" CLOSE, dthA[positionA]);
    printf("Max_theta :\t\t%f(dB)\n", max_tempA);
    


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
    float angle[100] = {2, 20.1, 50.6, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 16;
    int snr = -10;
    int qr_iter = 1;
    int hybrid_qr_iter = 1;
    float result[3] = {0};
    // int angle = 50;
    int iter = 1;

    //=================== MUSIC Algorithm =================================
    timeMusic_start = clock();
    // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
    //MUSIC_DOA_2A_CPU_test(M, qr_iter, hybrid_qr_iter, &angle[0],number_angle, result, snr, check_result);
    timeMusic_end = clock();
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MUSIC time : \t%.3f(ms)\n" CLOSE, (timeMusic_end - timeMusic_start) / CLOCKS_PER_SEC * 1000);
    printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time / 1000);
    total_multiply_time = 0;      // set to 0
    total_pre_transpose_time = 0; // set to 0
    //=====================================================================
    //*
    int target_sequence[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };
    int targetCode[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };
    int check_result=1;
    int temp ;
    //print_code(targetCode,CODE_LENGTH);
    //print_code(target_sequence,CODE_LENGTH);
    checkpnCode(targetCode, target_sequence,&check_result);
    MUSIC_DOA_2A_CPU_test(M, qr_iter, hybrid_qr_iter, &angle[0],number_angle, result, snr, &check_result);
    int search_point=4;
    //*
    printf("\n");
    for(int i=0;i<126-search_point;i++){
        printf("i=%d\n",i);
        temp = targetCode[i%127];
        //printf("%d\n",temp);
        for(int j=0;j<CODE_LENGTH-1;j++){
            target_sequence[j]=target_sequence[j+1];
        }
        target_sequence[126]=temp;
        //printf("sequence\n");
        //print_code(target_sequence,CODE_LENGTH);
        //printf("code\n");
        //print_code(targetCode,CODE_LENGTH);
        checkpnCode(targetCode, target_sequence,&check_result);
        //MUSIC_DOA_2A_CPU_test(M, qr_iter, hybrid_qr_iter, &angle[0],number_angle, result, snr, &check_result);
        
        //printf("\n");
    }
    for(int i=126-search_point;i<127+search_point;i++){
        printf("i=%d\n",i);
        temp = targetCode[i%127];
        //printf("%d\n",temp);
        for(int j=0;j<CODE_LENGTH-1;j++){
            target_sequence[j]=target_sequence[j+1];
        }
        target_sequence[126]=temp;
        //printf("sequence\n");
        //print_code(target_sequence,CODE_LENGTH);
        //printf("code\n");
        //print_code(targetCode,CODE_LENGTH);
        checkpnCode(targetCode, target_sequence,&check_result);
        MUSIC_DOA_2A_CPU_test(M, qr_iter, hybrid_qr_iter, &angle[0],number_angle, result, snr, &check_result);
        
        //printf("\n");
    }
    //*/

}