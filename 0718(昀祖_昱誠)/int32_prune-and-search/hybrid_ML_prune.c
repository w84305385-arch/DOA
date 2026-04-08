// ML 混合母版本//當M>8:30度和10度是用M=8 混合做[0.25*lemda] ; 3、1、0.1用M float做[0.25*lemda]
// 範圍-60~+60度，30->10->3->1->0.1 
//M:64 時間變成35ms;當AVX加減法，時間->33ms
// g++ -mavx512f -g -o hybrid_ML_prune hybrid_ML_prune.c -Wall -Wextra -std=c++14 math_func.a
// ./hybrid_ML_prune
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
#include <x86intrin.h>
#include <avx512vlbwintrin.h>
#include <avx512bwintrin.h>
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
#include <avx512fintrin.h>
//----------------------global variable---------------------------
static float total_test = 0;
static float total_multiply_time = 0;
static float total_pre_transpose_time = 0;
//----------------------------------------------------------------
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
//-------------------------------------------------------------------
__attribute__((aligned(32))) int matC_Re_sumi[30000000] = {0}; // Real
__attribute__((aligned(32))) int matC_Im_sumi[30000000] = {0}; // Imaginary
__attribute__((aligned(32))) int matC_Reali[30000000] = {0}; // re_C
__attribute__((aligned(32))) int matC_Imagi[30000000] = {0}; // im_C

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
void matrix_transpose_i(int *matA_re, int *matA_im, int rowA, int colA)
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
void print_complex_matrix_i(int *matA_re, int *matA_im, int rowA, int colA)
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



void complex_matrix_addition_i(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
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

void complex_matrix_subtraction_i(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
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

void complex_matrix_multiplication_iii(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int *matC_re, int *matC_im, int rowA, int rowB, int colB)
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
    matrix_transpose_i(matB_re, matB_im, rowB, colB);              // Matrix transpose
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
    matrix_transpose_i(matB_re, matB_im, rowB, colB); // Matrix transpse -> back to origin version
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
void complex_matrix_get_columns_i(int *matA_re, int *matA_im, int *matCol_re, int *matCol_im, int rowA, int colA, int colTarget)
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
void complex_matrix_get_rows_i(int *matA_re, int *matA_im, int *matRow_re, int *matRow_im, int rowA, int colA, int rowTarget)
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
void complex_matrix_conjugate_transpose_i(int *matA_re, int *matA_im, int rowA, int colA)
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
void complex_matrix_conjugate_transpose_multiplication_iii(int *matA_re, int *matA_im, int *matB_re, int *matB_im, int rowA, int colA)
{
    int *temp_re = (int *)malloc(colA * rowA * sizeof(int));
    int *temp_im = (int *)malloc(colA * rowA * sizeof(int));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int)));
    complex_matrix_conjugate_transpose_i(temp_re, temp_im, rowA, colA);
    //print_complex_matrix_i(matA_re, matA_im, rowA,colA );
    //print_complex_matrix_i(temp_re, temp_im, colA,rowA );
    complex_matrix_multiplication_iii(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);
    //print_complex_matrix_i(matB_re, matB_im, 1, 1);
    free(temp_re);
    free(temp_im);
}

void trace(float *theta_re, float *theta_im, float *S_ML_re, float *S_ML_im, int rowA, int colA, int i){
    float temp_re = 0.0 ;
    float temp_im = 0.0 ;
    for(int id=0; id < colA ;id++){
        temp_re += theta_re[id *colA+id];
        temp_im += theta_im[id *colA+id];
    }
    S_ML_re[i] = temp_re;
    S_ML_im[i] = temp_im;
    /*
    printf("---\n");
    printf("re=\t%.2f ", S_ML_re[i]);
    printf("im=\t%.2f ", S_ML_im[i]);
    printf("---\n");
    */
}
void trace_i(int *theta_re, int *theta_im, int *S_ML_re, int *S_ML_im, int rowA, int colA, int i){
    int temp_re = 0.0 ;
    int temp_im = 0.0 ;
    for(int id=0; id < colA ;id++){
        temp_re += theta_re[id *colA+id];
        temp_im += theta_im[id *colA+id];
    }
    S_ML_re[i] = temp_re;
    S_ML_im[i] = temp_im;

}

void matrix_inverse_LU(float *a_vector_re, float *a_vector_im, float *A_inv_re, float *A_inv_im, int rowA, int colA){
    float *L_re = (float *)malloc(colA * rowA * sizeof(float));
    float *U_re = (float *)malloc(colA * rowA * sizeof(float));
    float *L_im = (float *)malloc(colA * rowA * sizeof(float));
    float *U_im = (float *)malloc(colA * rowA * sizeof(float));

    float *L_Inverse_re = (float *)malloc(colA * rowA * sizeof(float));
    float *U_Inverse_re = (float *)malloc(colA * rowA * sizeof(float));
    float *L_Inverse_im = (float *)malloc(colA * rowA * sizeof(float));
    float *U_Inverse_im = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_re = (float *)malloc(sizeof(float));
    float *temp_im = (float *)malloc(sizeof(float));
    

    float *S_re = (float *)malloc( sizeof(float));
    float *S_im = (float *)malloc( sizeof(float));

    //對角元素
    for (int i = 0; i < rowA; ++i)
    {
        L_re[i * colA + i] = 1.0;
        L_im[i * colA + i] = 0.0;
    }
    /*
    printf("--L1--\n");
    print_complex_matrix(L_re, L_im, rowA, colA);
    printf("----\n");
    */
    for (int j = 0; j < colA; j++)
    {
        U_re[j] = a_vector_re[j];
        U_im[j] = a_vector_im[j];
    }
    /*
    printf("--U1--\n");
    print_complex_matrix(U_re, U_im, rowA, colA);
    printf("----\n");
    */
    for (int i = 1; i < colA; i++)
    {
        L_re[i * colA]= (a_vector_re[i*colA]*U_re[0] + a_vector_im[i*colA]*U_im[0]) / (U_re[0]*U_re[0] + U_im[0]*U_im[0]);
        L_im[i * colA]= (a_vector_im[i*colA]*U_re[0] - a_vector_re[i*colA]*U_im[0]) / (U_re[0]*U_re[0] + U_im[0]*U_im[0]);
    }
    /*
    printf("--L2--\n");
    print_complex_matrix(L_re, L_im, rowA, colA);
    printf("----\n");
    */
    for (int k = 1; k < rowA; k++)
    {
        for (int j = k; j < colA; j++)
        {
            S_re[0] = 0.0;
            S_im[0] = 0.0;
            for (int t = 0; t < k; t++) {
                
                temp_re[0] = (L_re[k * colA + t]*U_re[t * colA + j]) - (L_im[k * colA + t]*U_im[t * colA + j]);//temp=L*U (k,j,t)=(1,1,0),(1,2,0),(2,2,0),(2,2,1)
                //L(3)、U(1)//L(3)、U(2)//L(6)、U(2)//L(7)、U(5)//
                temp_im[0] = (L_re[k * colA + t]*U_im[t * colA + j]) + (L_im[k * colA + t]*U_re[t * colA + j]);
                S_re[0] = S_re[0]+temp_re[0];
                S_im[0] = S_im[0]+temp_im[0];
                printf("\tS_re=...=%.3f\n", S_re[0]);
                printf("\tS_im=...=%.3f\n", S_im[0]);
            }
            U_re[k * colA + j] = a_vector_re[k * colA + j] - S_re[0];
            U_im[k * colA + j] = a_vector_im[k * colA + j] - S_im[0];
            //printf("--U2--\n");
            //print_complex_matrix(U_re, U_im, rowA, colA);
            //printf("----\n");
        }
        
        for (int i = k; i < colA; i++)
        {
            S_re[0] = 0.0;
            S_im[0] = 0.0;
            for (int t = 0; t < k; t++)
            {
                temp_re[0] = L_re[i * colA + t]*U_re[t * colA + k]-L_im[i * colA + t]*U_im[t * colA + k];
                temp_im[0] = L_re[i * colA + t]*U_im[t * colA + k]+L_im[i * colA + t]*U_re[t * colA + k];
                S_re[0] = S_re[0]+temp_re[0];
                S_im[0] = S_im[0]+temp_im[0];
                //printf("\tS_re=%.3f\n", S_re[0]);
                //printf("\tS_im=%.3f\n", S_im[0]);
            }
            temp_re[0] = a_vector_re[i * colA + k] - S_re[0];
            temp_im[0] = a_vector_im[i * colA + k] - S_im[0];
            L_re[i * colA+ k]=(temp_re[0]*U_re[k * colA + k]+temp_im[0]*U_im[k * colA + k])/(U_re[k * colA + k]*U_re[k * colA + k]+U_im[k * colA + k]*U_im[k * colA + k]);
            L_im[i * colA+ k]=(temp_im[0]*U_re[k * colA + k]-temp_re[0]*U_im[k * colA + k])/(U_re[k * colA + k]*U_re[k * colA + k]+U_im[k * colA + k]*U_im[k * colA + k]);
        }
        
    }
    /*
    printf("--L3--\n");
    print_complex_matrix(L_re, L_im, rowA, colA);
    printf("----\n");
    
    
    printf("--U2--\n");
    print_complex_matrix(U_re, U_im, rowA, colA);
    printf("----\n");
    */
    for (int i = 0; i < rowA; i++)
    {
        L_Inverse_im[i * colA + i] = 0.0;
        L_Inverse_re[i * colA + i] = 1.0;
    }
    /*
    printf("--L_inv1--\n");
    print_complex_matrix(L_Inverse_re, L_Inverse_im, rowA, colA);
    printf("----\n");
    */
    for (int j = 0; j < colA; j++)
    {
        for (int i = j + 1; i < rowA; i++)
        {
            S_re[0] = 0.0;
            S_im[0] = 0.0;
            for (int k = j; k < i; k++) {
                temp_re[0] = L_re[i * colA + k]*L_Inverse_re[k * colA + j]-L_im[i * colA + k]*L_Inverse_im[k * colA + j];
                temp_im[0] = L_re[i * colA + k]*L_Inverse_im[k * colA + j]+L_im[i * colA + k]*L_Inverse_re[k * colA + j];
                S_re[0] = S_re[0]+temp_re[0];
                S_im[0] = S_im[0]+temp_im[0]; 
            }
            L_Inverse_re[i * colA + j] = (-1) * (L_Inverse_re[j * colA + j] * S_re[0]-L_Inverse_im[j * colA + j] * S_im[0]);
            L_Inverse_im[i * colA + j] = (-1) * (L_Inverse_re[j * colA + j] * S_im[0]+L_Inverse_im[j * colA + j] * S_re[0]);
        }
    }
    /*
    printf("--L_inv2--\n");
    print_complex_matrix(L_Inverse_re, L_Inverse_im, rowA, colA);
    printf("----\n");
    */
    S_re[0] = 1.0;
    S_im[0] = 0.0;
    
    for (int i = 0; i < colA; i++)                    //按列序，列内按照從下到上，計算u的逆矩陣
    {
        U_Inverse_re[i * colA + i] = (S_re[0]*U_re[i * colA + i]+S_im[0]*U_im[i * colA + i])/(U_re[i * colA + i]*U_re[i * colA + i]+U_im[i * colA + i]*U_im[i * colA + i]);
        U_Inverse_im[i * colA + i] = (S_im[0]*U_re[i * colA + i]-S_re[0]*U_im[i * colA + i])/(U_re[i * colA + i]*U_re[i * colA + i]+U_im[i * colA + i]*U_im[i * colA + i]);
    }
    /*
    printf("--U_inv2--\n");
    print_complex_matrix(U_Inverse_re, U_Inverse_im, rowA, colA);
    printf("----\n");
    */
    for (int j = 0; j < colA; j++)
    {
        for (int i = j - 1; i >= 0; i--)
        {
            S_im[0] = 0.0;
            S_re[0] = 0.0;
            for (int k = i + 1; k <= j; k++)
            {
                temp_re[0] = U_re[i * colA + k]*U_Inverse_re[k * colA + j]-U_im[i * colA + k]*U_Inverse_im[k * colA + j];
                temp_im[0] = U_re[i * colA + k]*U_Inverse_im[k * colA + j]+U_im[i * colA + k]*U_Inverse_re[k * colA + j];
                S_re[0] = S_re[0]+temp_re[0];
                S_im[0] = S_im[0]+temp_im[0];
            }
            S_im[0] = (-1)* S_im[0];
            S_re[0] = (-1)* S_re[0];;
            U_Inverse_re[i * colA + j] = (S_re[0]*U_re[i * colA + i]+S_im[0]*U_im[i * colA + i]) / (U_re[i * colA + i]*U_re[i * colA + i]+U_im[i * colA + i]*U_im[i * colA + i]);
            U_Inverse_im[i * colA + j] = (S_im[0]*U_re[i * colA + i]-S_re[0]*U_im[i * colA + i]) / (U_re[i * colA + i]*U_re[i * colA + i]+U_im[i * colA + i]*U_im[i * colA + i]);

        }
    }
    complex_matrix_multiplication(U_Inverse_re, U_Inverse_im, L_Inverse_re, L_Inverse_im, A_inv_re, A_inv_im, rowA, rowA, colA);
    free(L_re);
    free(U_re);
    free(L_im);
    free(U_im);
    free(L_Inverse_re);
    free(U_Inverse_re);
    free(L_Inverse_im);
    free(U_Inverse_im);
    free(temp_re);
    free(temp_im);
    free(S_re);
    free(S_im);
}

void ML_DOA_1D_CPU(int M, int qr_iter, float *angle, int number_angle, float *result, int SNR) {
    //-------------------------------------------------------------------
    struct timeval start_test, end_test, diff_test;    // multiplication variable
    // Parameter initialize
    struct timeval time_para_start, time_para_end, time_para_diff;          // time initial
    //-------------------------------------------------------------------
    struct timeval time_rec_start, time_rec_end, time_rec_diff;          // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_DML_start, time_DML_end, time_DML_diff;          // time initial
    //struct timeval time_LU_start, time_LU_end, time_LU_diff;          // time initial
    struct timeval time_exp1_start, time_exp1_end, time_exp1_diff;          // time initial
    struct timeval time_sig_start, time_sig_end, time_sig_diff;          // time initial
    //struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;          // time initial
    //struct timeval time_exp2_start, time_exp2_end, time_exp2_diff;          // time initial
    struct timeval time_findmax_start, time_findmax_end, time_findmax_diff;          // time initial
    struct timeval time_free_start, time_free_end, time_free_diff;          // time initial
    
    
    float time_MLre = 0.0;
    float timeMLre_start, timeMLre_end; // Total ML Algorithm time
    
    gettimeofday(&time_free_start, NULL);
    //gettimeofday(&time_sig_start, NULL);
    //-------------------------------------------------------------------
    printf("---------------\n");
    printf("--ML DOA--\n");
    printf("---------------\n");
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", SNR);
    //printf("QR iteration:\t\t%d\n", qr_iter);
    
    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    //-----------------------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------
    //gettimeofday(&time_para_start, NULL);
    const int fc = 2.5e+9;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.5;
    float kc =  2*PI / lemda;
    const int nd = 512;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    //gettimeofday(&time_para_end, NULL);
    //gettimeofday(&time_sigg_start, NULL);
    printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        printf("%.1f, \n", angle[a]);
    }
    // A_theta matrix (M, length of t_theta)
    //-----------------------------------------------------------------------------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    float *A_theta_re8 = (float *)malloc(8 * len_t_theta * sizeof(float));
    float *A_theta_im8 = (float *)malloc(8 * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    //gettimeofday(&time_exp1_start, NULL);
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
            // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
        }
    }
    if(M>8){
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < len_t_theta; ++j)
            {
                cpp_exp(&A_theta_re8[i * len_t_theta + j], &A_theta_im8[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
                // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
            }
        }
    }
    //gettimeofday(&time_exp1_end, NULL);
    // t_sig matrix (length of t_theta, nd)
    //---------------------------------------------------------------
    float *t_sig_re = (float *)malloc(nd * len_t_theta * sizeof(float));
    float *t_sig_im = (float *)malloc(nd * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < len_t_theta; ++i)
    {
        for (int j = 0; j < nd; ++j)
        {
            cpp_t_sig(&t_sig_re[i * nd + j], &t_sig_im[i * nd + j]);
            // printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
        }
    }
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    float *sig_co_re = (float *)malloc(M * nd * sizeof(float));
    float *sig_co_im = (float *)malloc(M * nd * sizeof(float)); 
    //---------------------------------------------------------------
    float *x_r_re = (float *)malloc(M * nd * sizeof(float));
    float *x_r_im = (float *)malloc(M * nd * sizeof(float));
    //---------------------------------------------------------------
    float *sig_co_re8 = (float *)malloc(8 * nd * sizeof(float));
    float *sig_co_im8 = (float *)malloc(8 * nd * sizeof(float));
    //---------------------------------------------------------------
    float *x_r_re8 = (float *)malloc(8 * nd * sizeof(float));
    float *x_r_im8 = (float *)malloc(8 * nd * sizeof(float));
    //---------------------------------------------------------------
    // compute sig_co
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    //gettimeofday(&time_sig_end, NULL);
    // print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    gettimeofday(&time_AWGN_end, NULL);
    //---------------------------------------------------------------
    if(M>8){
        complex_matrix_multiplication(A_theta_re8, A_theta_im8, t_sig_re, t_sig_im, sig_co_re8, sig_co_im8, 8, len_t_theta, nd);
        //print_complex_matrix(sig_co_re, sig_co_im, M, nd);
        //gettimeofday(&time_AWGN_start, NULL);
        cpp_awgn(sig_co_re8, sig_co_im8, x_r_re8, x_r_im8, SNR, 8, nd);
    }
    // ml algorithm
    // R_xx matrix (M, M)
    //gettimeofday(&time_rec_start, NULL);
    float *R_xx_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_im = (float *)malloc(M * M * sizeof(float));
    // matlab code:  (R_xx = (1 / M )* x_r * x_r')
    float M_re = M;
    float M_im = 0.0;
    float *M_ptr = &M_re;
    float *M_ptr_im = &M_im;
    //---------------------------------------------------------------
    //gettimeofday(&time_Rxx_start, NULL);
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int i = 0; i < M * M; ++i)
    {
        // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    //---------------------------------------------------------------
    float *R_xx_re8 = (float *)malloc(8 * 8 * sizeof(float));
    float *R_xx_im8 = (float *)malloc(8 * 8 * sizeof(float));
    int *R_xx_re8_i = (int *)malloc(8 * 8 * sizeof(int));
    int *R_xx_im8_i = (int *)malloc(8 * 8 * sizeof(int));
    float M_re8 = 8;
    float M_im8 = 0.0;
    float *M_ptr8 = &M_re8;
    float *M_ptr_im8 = &M_im8;
    //---------------------------------------------------------------
    if(M>8){
        complex_matrix_conjugate_transpose_multiplication(x_r_re8, x_r_im8, R_xx_re8, R_xx_im8, 8, nd);
        for (int i = 0; i < 8 * 8; ++i)
        {
            //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
            cpp_division(&R_xx_re8[i], &R_xx_im8[i], M_ptr8, M_ptr_im8);
            //printf("(%.2f,%.2f)\n", R_xx_re8[i], R_xx_im8[i]);
        }
        for(int w = 0; w < 8*8 ; w++){
            R_xx_re8_i[w]=round(R_xx_re8[w]/2);
            R_xx_im8_i[w]=round(R_xx_im8[w]/2);
        }
    }
    timeMLre_start = clock();
    //---------------------------------------------------------------
    // 取定點
    // parameter setting
    const int len_dth = 5; //30度
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    double max_temp ;
    int position = 0;
    const int len_dthA = 5;                                  //實際搜索      -60~-40 -50~-10 -20~20 10~50 40~60 overlap 10度
    float *dthA = (float *)malloc(len_dthA * sizeof(float)); //dth[position]  -60      -30     0     30    60 
    float *drA = (float *)malloc(len_dthA * sizeof(float));
    double max_tempA ;
    int positionA = 0;
    //-------------------------------------------------------//
if(M>8){
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 30 * i; //-60 -30 0 30 60
        dr[i] = dth[i] * PI / 180;
    }
    float *a_vector_re = (float *)malloc(M_re8 * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(M_re8 * sizeof(float));
    float *a_temp_re = (float *)malloc(M_re8 * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(M_re8 * sizeof(float));
    float *S_ML_re = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_im = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_dB = (float *)malloc(len_dth * sizeof(float));
    //float *theta_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *theta_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    float *AH_mulA_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *AH_mulA2_re = (float *)malloc(1 * M_re8 * sizeof(float));
    //float *AH_mulA2_im = (float *)malloc(1 * M_re8 * sizeof(float));
    //float *AH_mulA3_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *AH_mulA3_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    int *a_vector_re_i = (int *)malloc(M_re8 * sizeof(int));  
    int *a_vector_im_i = (int *)malloc(M_re8 * sizeof(int));
    int *a_temp_re_i = (int *)malloc(M_re8 * sizeof(int));  
    int *a_temp_im_i = (int *)malloc(M_re8 * sizeof(int));  
    int *AH_mulA_inv_re_i = (int *)malloc(1 * 1 * sizeof(int));  
    int *AH_mulA_inv_im_i = (int *)malloc(1 * 1 * sizeof(int));  
    int *AH_mulA2_re_i = (int *)malloc(1 * M_re8 * sizeof(int));
    int *AH_mulA2_im_i = (int *)malloc(1 * M_re8 * sizeof(int));
    int *AH_mulA3_re_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *AH_mulA3_im_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *theta_re_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *theta_im_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *S_ML_re_i = (int *)malloc(len_dth * sizeof(int));
    int *S_ML_im_i = (int *)malloc(len_dth * sizeof(int));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dth; ++i) { 
        for(int j = 0; j < M_re8; ++j) {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_re,a_vector_re,(M_re8 * 1 * sizeof(float)));
        memcpy(a_temp_im,a_vector_im,(M_re8 * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, M_re8);   
        //printf("(i=%d)a_temp:\n",i);
        //print_complex_matrix(a_temp_re, a_temp_im, 1, M_re8);
        //A^H*A =1*1  
        complex_matrix_multiplication(a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, M_re8, 1);
        //[inv(A^H*A)] =1*1  
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        for(int w = 0; w < M_re8 ; w++){
            a_vector_re_i[w]=round(a_vector_re[w]*16);
            a_vector_im_i[w]=round(a_vector_im[w]*16);
        }
        for(int w = 0; w < M_re8 ; w++){
            a_temp_re_i[w]=round(a_temp_re[w]*16);
            a_temp_im_i[w]=round(a_temp_im[w]*16);
        }
        AH_mulA_inv_re_i[0]=AH_mulA_inv_re[0]*16;
        AH_mulA_inv_im_i[0]=AH_mulA_inv_im[0]*16;
        //printf("(i=%d)a_temp:\n",i);
        //print_complex_matrix_i(a_temp_re_i, a_temp_im_i, 1, M_re8);
        //printf("(i=%d)AH_mulA_inv:\n",i);
        //print_complex_matrix_i(AH_mulA_inv_re_i, AH_mulA_inv_im_i, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication_iii(AH_mulA_inv_re_i, AH_mulA_inv_im_i, a_temp_re_i, a_temp_im_i, AH_mulA2_re_i, AH_mulA2_im_i, 1, 1, M_re8);
        for(int w = 0; w < 8; w++){
            AH_mulA2_re_i[w]=(AH_mulA2_re_i[w]/2);
            AH_mulA2_im_i[w]=(AH_mulA2_im_i[w]/2);
        }
        //printf("(i=%d)AH_mulA2:\n",i);
        //print_complex_matrix_i(AH_mulA2_re_i, AH_mulA2_im_i, 1, M_re8);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication_iii(a_vector_re_i, a_vector_im_i, AH_mulA2_re_i, AH_mulA2_im_i, AH_mulA3_re_i, AH_mulA3_im_i, M_re8, 1, M_re8);
        
        for(int w = 0; w < 8*8 ; w++){
            AH_mulA3_re_i[w]=(AH_mulA3_re_i[w]/16);
            AH_mulA3_im_i[w]=(AH_mulA3_im_i[w]/16);
        }
        //printf("(i=%d)AH_mulA3:\n",i);
        //print_complex_matrix_i(AH_mulA3_re_i, AH_mulA3_im_i, M_re8, M_re8);
        // P_A*R = M*M 
        complex_matrix_multiplication_iii(AH_mulA3_re_i, AH_mulA3_im_i, R_xx_re8_i, R_xx_im8_i, theta_re_i, theta_im_i, M_re8, M_re8, M_re8);
        //printf("(i=%d)theta:\n",i);
        //print_complex_matrix_i( theta_re_i, theta_im_i, M_re8, M_re8);
        // trace[P_A*R] 
        trace_i(theta_re_i, theta_im_i, S_ML_re_i, S_ML_im_i, M_re8, M_re8, i);
        //printf("(i=%d)trace完:\n",i);
        //print_complex_matrix_i(S_ML_re_i, S_ML_im_i, 1, len_dth);
        
        //printf(YELLOW"---\n"CLOSE);
    }
    free(a_temp_re);
    free(a_temp_im);

    for(int i = 0; i < len_dth; ++i) {
        S_ML_re[i]=S_ML_re_i[i];
        S_ML_im[i]=S_ML_im_i[i];
    }

    for(int i = 0; i < len_dth; ++i) {
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    // find Max and position
    max_temp = S_ML_dB[0];
    for(int i = 0; i < len_dth; ++i) {
        if(S_ML_dB[i] > max_temp) {
            max_temp = S_ML_dB[i];
            position =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
    // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
    //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
//*   
    //-------------------------------------------------------------------
    //10度10度找//30度內搜索時重疊10度並以10度為單位搜索
    if(position == 0)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthA-2; ++i)  
        {  
            dthA[i] = -60 + 10 * i;
            //printf("dthA:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    else if(position == 4)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthA-2; ++i)  
        {  
            dthA[i] = 40 + 10 * i;
            //printf("dth1:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    else
    {
        //printf("進else1\n");
        for (int i = 0; i < len_dthA; ++i)  
        {  
            dthA[i] = (dth[position]-20) + 10 * i;
            //printf("dth1:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    //-------------------------------------------------------------------
    float *a_vector_A_re = (float *)malloc(M_re8 * sizeof(float));  //a_vector = M*1
    float *a_vector_A_im = (float *)malloc(M_re8 * sizeof(float));
    float *a_temp_A_re = (float *)malloc(M_re8 * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_A_im = (float *)malloc(M_re8 * sizeof(float));
    float *S_ML_A_re = (float *)malloc(len_dthA * sizeof(float));
    float *S_ML_A_im = (float *)malloc(len_dthA * sizeof(float));
    float *S_ML_A_dB = (float *)malloc(len_dthA * sizeof(float));
    //float *theta_A_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *theta_A_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_A_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    float *AH_mulA_A_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *AH_mulA2_A_re = (float *)malloc(1 * M_re8 * sizeof(float));
    //float *AH_mulA2_A_im = (float *)malloc(1 * M_re8 * sizeof(float));
    //float *AH_mulA3_A_re = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    //float *AH_mulA3_A_im = (float *)malloc(M_re8 * M_re8 * sizeof(float));
    float *AH_mulA_inv_A_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_A_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    int *a_vector_A_re_i = (int *)malloc(M_re8 * sizeof(int));  
    int *a_vector_A_im_i = (int *)malloc(M_re8 * sizeof(int));
    int *a_temp_A_re_i = (int *)malloc(M_re8 * sizeof(int));  
    int *a_temp_A_im_i = (int *)malloc(M_re8 * sizeof(int));  
    int *AH_mulA_inv_A_re_i = (int *)malloc(1 * 1 * sizeof(int));  
    int *AH_mulA_inv_A_im_i = (int *)malloc(1 * 1 * sizeof(int));  
    int *AH_mulA2_A_re_i = (int *)malloc(1 * M_re8 * sizeof(int));
    int *AH_mulA2_A_im_i = (int *)malloc(1 * M_re8 * sizeof(int));
    int *AH_mulA3_A_re_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *AH_mulA3_A_im_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *theta_A_re_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *theta_A_im_i = (int *)malloc(M_re8 * M_re8 * sizeof(int));
    int *S_ML_A_re_i = (int *)malloc(len_dth * sizeof(int));
    int *S_ML_A_im_i = (int *)malloc(len_dth * sizeof(int));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dthA; ++i) { 
        for(int j = 0; j < M_re8; ++j) {
            cpp_exp2(&a_vector_A_re[j], &a_vector_A_im[j], drA, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_A_re,a_vector_A_re,(M_re8 * 1 * sizeof(float)));
        memcpy(a_temp_A_im,a_vector_A_im,(M_re8 * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_A_re, a_temp_A_im, 1, M_re8);   
        //A^H*A =1*1  
        complex_matrix_multiplication( a_temp_A_re, a_temp_A_im, a_vector_A_re, a_vector_A_im, AH_mulA_A_re, AH_mulA_A_im, 1, M_re8, 1);
        //[inv(A^H*A)] =1*1   
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_A_re, AH_mulA_A_im, AH_mulA_inv_A_re, AH_mulA_inv_A_im, 1, 1);
        for(int w = 0; w < M_re8 ; w++){
            a_vector_A_re_i[w]=round(a_vector_A_re[w]*16);
            a_vector_A_im_i[w]=round(a_vector_A_im[w]*16);
        }
        for(int w = 0; w < M_re8 ; w++){
            a_temp_A_re_i[w]=round(a_temp_A_re[w]*16);
            a_temp_A_im_i[w]=round(a_temp_A_im[w]*16);
        }
        AH_mulA_inv_A_re_i[0]=AH_mulA_inv_A_re[0]*16;
        AH_mulA_inv_A_im_i[0]=AH_mulA_inv_A_im[0]*16;
        //printf("(i=%d)a_temp:\n",i);
        //print_complex_matrix_i(a_temp_A_re_i, a_temp_A_im_i, 1, M_re8);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication_iii(AH_mulA_inv_A_re_i, AH_mulA_inv_A_im_i, a_temp_A_re_i, a_temp_A_im_i, AH_mulA2_A_re_i, AH_mulA2_A_im_i, 1, 1, M_re8);
        //printf("(i=%d)AH_mulA2:\n",i);
        //print_complex_matrix_i(AH_mulA2_A_re_i, AH_mulA2_A_im_i, 1, M_re8);
        for(int w = 0; w < 8; w++){
            AH_mulA2_A_re_i[w]=(AH_mulA2_A_re_i[w]/2);
            AH_mulA2_A_im_i[w]=(AH_mulA2_A_im_i[w]/2);
        }
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication_iii(a_vector_A_re_i, a_vector_A_im_i, AH_mulA2_A_re_i, AH_mulA2_A_im_i, AH_mulA3_A_re_i, AH_mulA3_A_im_i, M_re8, 1, M_re8);
        for(int w = 0; w < 8*8 ; w++){
            AH_mulA3_A_re_i[w]=(AH_mulA3_A_re_i[w]/16);
            AH_mulA3_A_im_i[w]=(AH_mulA3_A_im_i[w]/16);
        }
        //printf("(i=%d)AH_mulA3:\n",i);
        //print_complex_matrix_i(AH_mulA3_A_re_i, AH_mulA3_A_im_i, M_re8, M_re8);
        // P_A*R = M*M 
        complex_matrix_multiplication_iii(AH_mulA3_A_re_i, AH_mulA3_A_im_i, R_xx_re8_i, R_xx_im8_i, theta_A_re_i, theta_A_im_i, M_re8, M_re8, M_re8);
        //printf("(i=%d)theta_A:\n",i);
        //print_complex_matrix_i(theta_A_re_i, theta_A_im_i, M_re8, M_re8);
        // trace[P_A*R] 
        trace_i(theta_A_re_i, theta_A_im_i, S_ML_A_re_i, S_ML_A_im_i, M_re8, M_re8, i);
        //printf("(i=%d)trace完:\n",i);
        //print_complex_matrix_i(S_ML_A_re_i, S_ML_A_im_i, 1, len_dthA);
    }
    free(a_temp_A_re);
    free(a_temp_A_im);
    for(int i = 0; i < len_dth; ++i) {
        S_ML_A_re[i]=S_ML_A_re_i[i];
        S_ML_A_im[i]=S_ML_A_im_i[i];
    }
    for(int i = 0; i < len_dthA; ++i) {
        S_ML_A_dB[i] = cpp_20log_abs(&S_ML_A_re[i], &S_ML_A_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    //find Max and position
    //printf("\n");
    max_tempA = S_ML_A_dB[0];
    for(int i = 0; i < len_dthA; ++i) {
        if(S_ML_A_dB[i] > max_tempA) {
            max_tempA = S_ML_A_dB[i];
            positionA =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
       //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
}
else{
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 30 * i; //-60 -30 0 30 60
        dr[i] = dth[i] * PI / 180;
    }
    float *a_vector_re = (float *)malloc(M * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(M * sizeof(float));
    float *a_temp_re = (float *)malloc(M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(M * sizeof(float));
    float *S_ML_re = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_im = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_dB = (float *)malloc(len_dth * sizeof(float));
    float *theta_re = (float *)malloc(M * M * sizeof(float));
    float *theta_im = (float *)malloc(M * M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    for(int i = 0; i < len_dth; ++i) { 
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_re,a_vector_re,(M * 1 * sizeof(float)));
        memcpy(a_temp_im,a_vector_im,(M * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, M);   
        //A^H*A =1*1  
        complex_matrix_multiplication(a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, M, 1);
        //[inv(A^H*A)] =1*1  
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA2_re, AH_mulA2_im, 1, 1, M);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_re, a_vector_im, AH_mulA2_re, AH_mulA2_im,AH_mulA3_re, AH_mulA3_im, M, 1, M);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, R_xx_re, R_xx_im, theta_re, theta_im, M, M, M);
        // trace[P_A*R] 
        trace(theta_re, theta_im, S_ML_re, S_ML_im, M, M, i);
    }
    free(a_temp_re);
    free(a_temp_im);
    for(int i = 0; i < len_dth; ++i) {
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    // find Max and position
    max_temp = S_ML_dB[0];
    for(int i = 0; i < len_dth; ++i) {
        if(S_ML_dB[i] > max_temp) {
            max_temp = S_ML_dB[i];
            position =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
    // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
    //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
//*   
    //-------------------------------------------------------------------
    //10度10度找//30度內搜索時重疊10度並以10度為單位搜索
    if(position == 0)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthA-2; ++i)  
        {  
            dthA[i] = -60 + 10 * i;
            //printf("dthA:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    else if(position == 4)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthA-2; ++i)  
        {  
            dthA[i] = 40 + 10 * i;
            //printf("dth1:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    else
    {
        //printf("進else1\n");
        for (int i = 0; i < len_dthA; ++i)  
        {  
            dthA[i] = (dth[position]-20) + 10 * i;
            //printf("dth1:%f\n",dthA[i]);
            drA[i] = dthA[i] * PI / 180;
        }
    }
    //-------------------------------------------------------------------
    float *a_vector_A_re = (float *)malloc(M * sizeof(float));  //a_vector = M*1
    float *a_vector_A_im = (float *)malloc(M * sizeof(float));
    float *a_temp_A_re = (float *)malloc(M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_A_im = (float *)malloc(M * sizeof(float));
    float *S_ML_A_re = (float *)malloc(len_dthA * sizeof(float));
    float *S_ML_A_im = (float *)malloc(len_dthA * sizeof(float));
    float *S_ML_A_dB = (float *)malloc(len_dthA * sizeof(float));
    float *theta_A_re = (float *)malloc(M * M * sizeof(float));
    float *theta_A_im = (float *)malloc(M * M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_A_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_A_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_A_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_A_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_A_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_A_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_A_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_A_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dthA; ++i) { 
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_A_re[j], &a_vector_A_im[j], drA, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_A_re,a_vector_A_re,(M * 1 * sizeof(float)));
        memcpy(a_temp_A_im,a_vector_A_im,(M * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_A_re, a_temp_A_im, 1, M);   
        //A^H*A =1*1  
        complex_matrix_multiplication(a_temp_A_re, a_temp_A_im, a_vector_A_re, a_vector_A_im, AH_mulA_A_re, AH_mulA_A_im, 1, M, 1);
        //[inv(A^H*A)] =1*1   
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_A_re, AH_mulA_A_im, AH_mulA_inv_A_re, AH_mulA_inv_A_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_A_re, AH_mulA_inv_A_im, a_temp_A_re, a_temp_A_im, AH_mulA2_A_re, AH_mulA2_A_im, 1, 1, M);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_A_re, a_vector_A_im, AH_mulA2_A_re, AH_mulA2_A_im,AH_mulA3_A_re, AH_mulA3_A_im, M, 1, M);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_A_re, AH_mulA3_A_im, R_xx_re, R_xx_im, theta_A_re, theta_A_im, M, M, M);
        // trace[P_A*R] 
        trace(theta_A_re, theta_A_im, S_ML_A_re, S_ML_A_im, M, M, i);
    }
    free(a_temp_A_re);
    free(a_temp_A_im);
        for(int i = 0; i < len_dthA; ++i) {
        S_ML_A_dB[i] = cpp_20log_abs(&S_ML_A_re[i], &S_ML_A_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    //find Max and position
    //printf("\n");
    max_tempA = S_ML_A_dB[0];
    for(int i = 0; i < len_dthA; ++i) {
        if(S_ML_A_dB[i] > max_tempA) {
            max_tempA = S_ML_A_dB[i];
            positionA =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
       //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
}
    //-------------------------------------------------------------------
    //3度3度找//10度內搜索時重疊2度並以3度為單位搜索
    const int len_dthB = 5;  //-60~-53.5  -56.5~-43.5 -46.5~-33.5  -36.5~-23.5  -26.5~-13.5  -16.5~-3.5 -6.5~6.5 3.5~16.5 13.5~26.5 23.5~36.5 33.5~46.5 43.5~56.5 53.5~60 
    //dthA[positionA]             -60          -50         -40          -30           -20         -10        0      10        20         30       40        50       60
    //實際搜索                //-60~-54      -56~-44     -46~-34     -36~-24        -26~-14      -16~-4    -6~6    4~16      14~26     24~36    34~46     44~56    54~60 
    float *dthB = (float *)malloc(len_dthB * sizeof(float));
    float *drB = (float *)malloc(len_dthB * sizeof(float));
    if(position == 0 && positionA == 0)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthB-2; ++i)  
        {  
            dthB[i] = -60 + 3 * i;
            //printf("dthA:%f\n",dthA[i]);
            drB[i] = dthB[i] * PI / 180;
        }
    }
    else if(position == 4 && positionA == 2)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthB-2; ++i)  
        {  
            dthB[i] = 54 + 3 * i;
            //printf("dth1:%f\n",dthA[i]);
            drB[i] = dthB[i] * PI / 180;
        }
    }
    else
    {
        //printf("進else1\n");
        for (int i = 0; i < len_dthB; ++i)  
        {  
            dthB[i] = (dthA[positionA]-6) + 3 * i;
            //printf("dth1:%f\n",dthA[i]);
            drB[i] = dthB[i] * PI / 180;
        }
    }
    //-------------------------------------------------------------------
    float *a_vector_B_re = (float *)malloc(M * sizeof(float));  //a_vector = M*1
    float *a_vector_B_im = (float *)malloc(M * sizeof(float));
    float *a_temp_B_re = (float *)malloc(M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_B_im = (float *)malloc(M * sizeof(float));
    float *S_ML_B_re = (float *)malloc(len_dthB * sizeof(float));
    float *S_ML_B_im = (float *)malloc(len_dthB * sizeof(float));
    float *S_ML_B_dB = (float *)malloc(len_dthB * sizeof(float));
    float *theta_B_re = (float *)malloc(M * M * sizeof(float));
    float *theta_B_im = (float *)malloc(M * M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_B_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_B_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_B_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_B_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_B_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_B_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_B_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_B_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dthB; ++i) { 
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_B_re[j], &a_vector_B_im[j], drB, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_B_re,a_vector_B_re,(M * 1 * sizeof(float)));
        memcpy(a_temp_B_im,a_vector_B_im,(M * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_B_re, a_temp_B_im, 1, M);   
        //A^H*A =1*1  
        complex_matrix_multiplication( a_temp_B_re, a_temp_B_im, a_vector_B_re, a_vector_B_im, AH_mulA_B_re, AH_mulA_B_im, 1, M, 1);
        //[inv(A^H*A)] =1*1   
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_B_re, AH_mulA_B_im, AH_mulA_inv_B_re, AH_mulA_inv_B_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_B_re, AH_mulA_inv_B_im, a_temp_B_re, a_temp_B_im, AH_mulA2_B_re, AH_mulA2_B_im, 1, 1, M);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_B_re, a_vector_B_im, AH_mulA2_B_re, AH_mulA2_B_im,AH_mulA3_B_re, AH_mulA3_B_im, M, 1, M);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_B_re, AH_mulA3_B_im, R_xx_re, R_xx_im, theta_B_re, theta_B_im, M, M, M);
        // trace[P_A*R] 
        trace(theta_B_re, theta_B_im, S_ML_B_re, S_ML_B_im, M, M, i);
    }
    free(a_temp_B_re);
    free(a_temp_B_im);
        for(int i = 0; i < len_dthB; ++i) {
        S_ML_B_dB[i] = cpp_20log_abs(&S_ML_B_re[i], &S_ML_B_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    //find Max and position
    //printf("\n");
    double max_tempB = S_ML_B_dB[0];
    int positionB=0;
    for(int i = 0; i < len_dthB; ++i) {
        if(S_ML_B_dB[i] > max_tempB) {
            max_tempB = S_ML_B_dB[i];
            positionB =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
       //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
    //---------------------------------------------------------------
    //1度1度找//3度內以1度為單位搜索//1度搜索時有某些角度重疊2度
    const int len_dthC = 4;   
    //dthB[positionB]            -60        -57          -56         -54            -53            -50          -47          -46        -44 -43 -40 -37 -34
    //實際搜索                //-60~-58  -58.5~-55.5  -57.5~-54.5   -55.5~-52.5   -54.5~-51.5   -51.5~-48.5  -48.5~-45.5  -47.5~-44.5
    float *dthC = (float *)malloc(len_dthC * sizeof(float));
    float *drC = (float *)malloc(len_dthC * sizeof(float));
    if(position == 0 && positionA == 0 && positionB == 0)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthC-1; ++i)  
        {  
            dthC[i] = -60 +  i;
            //printf("dthA:%f\n",dthA[i]);
            drC[i] = dthC[i] * PI / 180;
        }
    }
    else if(position == 4 && positionA == 2 && positionB == 2)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthC-1; ++i)  
        {  
            dthC[i] = 58 +  i;
            //printf("dth1:%f\n",dthA[i]);
            drC[i] = dthC[i] * PI / 180;
        }
    }
    else
    {
        //printf("進else1\n");
        for (int i = 0; i < len_dthC; ++i)  
        {  
            dthC[i] = (dthB[positionB]-1.5) +  i;
            //printf("dth1:%f\n",dthA[i]);
            drC[i] = dthC[i] * PI / 180;
        }
    }
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    float *a_vector_C_re = (float *)malloc(M * sizeof(float));  //a_vector = M*1
    float *a_vector_C_im = (float *)malloc(M * sizeof(float));
    float *a_temp_C_re = (float *)malloc(M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_C_im = (float *)malloc(M * sizeof(float));
    float *S_ML_C_re = (float *)malloc(len_dthC * sizeof(float));
    float *S_ML_C_im = (float *)malloc(len_dthC * sizeof(float));
    float *S_ML_C_dB = (float *)malloc(len_dthC * sizeof(float));
    float *theta_C_re = (float *)malloc(M * M * sizeof(float));
    float *theta_C_im = (float *)malloc(M * M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_C_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_C_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_C_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_C_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_C_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_C_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_C_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_C_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dthC; ++i) { 
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_C_re[j], &a_vector_C_im[j], drC, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_C_re,a_vector_C_re,(M * 1 * sizeof(float)));
        memcpy(a_temp_C_im,a_vector_C_im,(M * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_C_re, a_temp_C_im, 1, M);   
        //A^H*A =1*1  
        complex_matrix_multiplication( a_temp_C_re, a_temp_C_im, a_vector_C_re, a_vector_C_im, AH_mulA_C_re, AH_mulA_C_im, 1, M, 1);
        //[inv(A^H*A)] =1*1   
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_C_re, AH_mulA_C_im, AH_mulA_inv_C_re, AH_mulA_inv_C_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_C_re, AH_mulA_inv_C_im, a_temp_C_re, a_temp_C_im, AH_mulA2_C_re, AH_mulA2_C_im, 1, 1, M);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_C_re, a_vector_C_im, AH_mulA2_C_re, AH_mulA2_C_im, AH_mulA3_C_re, AH_mulA3_C_im, M, 1, M);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_C_re, AH_mulA3_C_im, R_xx_re, R_xx_im, theta_C_re, theta_C_im, M, M, M);
        // trace[P_A*R] 
        trace(theta_C_re, theta_C_im, S_ML_C_re, S_ML_C_im, M, M, i);
    }
    free(a_temp_C_re);
    free(a_temp_C_im);
        for(int i = 0; i < len_dthC; ++i) {
        S_ML_C_dB[i] = cpp_20log_abs(&S_ML_C_re[i], &S_ML_C_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    //find Max and position
    //printf("\n");
    double max_tempC = S_ML_C_dB[0];
    int positionC = 0;
    for(int i = 0; i < len_dthC; ++i) {
        if(S_ML_C_dB[i] > max_tempC) {
            max_tempC = S_ML_C_dB[i];
            positionC =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
       //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
    //---------------------------------------------------------------
    //0.1度0.1度找//1度內以0.1度為單位搜索
    const int len_dthD = 11;   
    //dthC[positionC]           -60 -59 -58.5  -58 -57.5  -56.5  -55.5  -54.5  -53.5 -52.5 -51.5 -50.5
    //實際搜索                //
    float *dthD = (float *)malloc(len_dthD * sizeof(float));
    float *drD = (float *)malloc(len_dthD * sizeof(float));
    if(position == 0 && positionA == 0 && positionB == 0 && positionC == 0)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthD-5; ++i)  
        {  
            dthD[i] = -60 +  0.1*i;
            //printf("dthA:%f\n",dthA[i]);
            drD[i] = dthD[i] * PI / 180;
        }
    }
    else if(position == 4 && positionA == 2 && positionB == 2 && positionC == 2)
    {
        //printf("進if1\n");
        for (int i = 0; i < len_dthD-5; ++i)  
        {  
            dthD[i] = 59.5 +  0.1*i;
            //printf("dth1:%f\n",dthA[i]);
            drD[i] = dthD[i] * PI / 180;
        }
    }
    else
    {
        //printf("進else1\n");
        for (int i = 0; i < len_dthD; ++i)  
        {  
            dthD[i] = (dthC[positionC]-0.5) +  0.1*i;
            //printf("dth1:%f\n",dthA[i]);
            drD[i] = dthD[i] * PI / 180;
        }
    }
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    float *a_vector_D_re = (float *)malloc(M * sizeof(float));  //a_vector = M*1
    float *a_vector_D_im = (float *)malloc(M * sizeof(float));
    float *a_temp_D_re = (float *)malloc(M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_D_im = (float *)malloc(M * sizeof(float));
    float *S_ML_D_re = (float *)malloc(len_dthD * sizeof(float));
    float *S_ML_D_im = (float *)malloc(len_dthD * sizeof(float));
    float *S_ML_D_dB = (float *)malloc(len_dthD * sizeof(float));
    float *theta_D_re = (float *)malloc(M * M * sizeof(float));
    float *theta_D_im = (float *)malloc(M * M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_D_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_D_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_D_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_D_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_D_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_D_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_D_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_D_im = (float *)malloc(1 * 1 * sizeof(float));
    //-------------------------------------------------------------------
    for(int i = 0; i < len_dthD; ++i) { 
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_D_re[j], &a_vector_D_im[j], drD, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_D_re,a_vector_D_re,(M * 1 * sizeof(float)));
        memcpy(a_temp_D_im,a_vector_D_im,(M * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_D_re, a_temp_D_im, 1, M);   
        //A^H*A =1*1  
        complex_matrix_multiplication( a_temp_D_re, a_temp_D_im, a_vector_D_re, a_vector_D_im, AH_mulA_D_re, AH_mulA_D_im, 1, M, 1);
        //[inv(A^H*A)] =1*1   
        //gettimeofday(&time_findmax_start, NULL);
        gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_D_re, AH_mulA_D_im, AH_mulA_inv_D_re, AH_mulA_inv_D_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_D_re, AH_mulA_inv_D_im, a_temp_D_re, a_temp_D_im, AH_mulA2_D_re, AH_mulA2_D_im, 1, 1, M);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_D_re, a_vector_D_im, AH_mulA2_D_re, AH_mulA2_D_im, AH_mulA3_D_re, AH_mulA3_D_im, M, 1, M);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_D_re, AH_mulA3_D_im, R_xx_re, R_xx_im, theta_D_re, theta_D_im, M, M, M);
        // trace[P_A*R] 
        trace(theta_D_re, theta_D_im, S_ML_D_re, S_ML_D_im, M, M, i);
    }
    free(a_temp_D_re);
    free(a_temp_D_im);
        for(int i = 0; i < len_dthD; ++i) {
        S_ML_D_dB[i] = cpp_20log_abs(&S_ML_D_re[i], &S_ML_D_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    //find Max and position
    //printf("\n");
    double max_tempD = S_ML_D_dB[0];
    int positionD = 0;
    for(int i = 0; i < len_dthD; ++i) {
        if(S_ML_D_dB[i] > max_tempD) {
            max_tempD = S_ML_D_dB[i];
            positionD =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }
       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
       //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
    timeMLre_end = clock();
    




    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    //*
    printf("positionA : \t\t%d\n", positionA);
    printf(RED "Theta estimationA :\t%.3f (degree)\n" CLOSE, dthA[positionA]);
    printf("Max_thetaA :\t\t%f(dB)\n", max_tempA);
    
    
    printf("positionB : \t\t%d\n", positionB);
    printf(RED "Theta estimationB :\t%.3f (degree)\n" CLOSE, dthB[positionB]);
    printf("Max_thetaB :\t\t%f(dB)\n", max_tempB);
    
    printf("positionC : \t\t%d\n", positionC);
    printf(RED "Theta estimationC :\t%.3f (degree)\n" CLOSE, dthC[positionC]);
    printf("Max_thetaC :\t\t%f(dB)\n", max_tempC);

    printf("positionD : \t\t%d\n", positionD);
    printf(RED "Theta estimationD :\t%.3f (degree)\n" CLOSE, dthD[positionD]);
    printf("Max_thetaD :\t\t%f(dB)\n", max_tempD);
    
    printf(L_GREEN "Total ML REAL time :\t%.3f(ms)\n" CLOSE, (timeMLre_end - timeMLre_start) / CLOCKS_PER_SEC * 1000);
}

int main()
{
    //-------------------------------------------------------------------
    // Parameter initialize
    float time_ML = 0.0;
    float timeML_start, timeML_end; // Total ML Algorithm time
    //-------------------------------------------------------------------
    float angle[30] = {20.2, -10, 20, 60}; // angle of array
    int number_angle = 1;
    int M = 64;
    int snr = 10;
    int qr_iter = 1;
    float result[3] = {0};
    // int angle = 50;
    int iter = 1;
    //=================== ML Algorithm =================================
    
    timeML_start = clock();

    ML_DOA_1D_CPU(M, qr_iter, &angle[0], number_angle, result, snr);

    timeML_end = clock();
    
    printf("--------------------------------------\n");
    printf(L_GREEN "Total ML time : \t%.3f(ms)\n" CLOSE, (timeML_end - timeML_start) / CLOCKS_PER_SEC * 1000);
    
}
// g++ -mavx512f -g -o hybrid_ML_prune hybrid_ML_prune.c -Wall -Wextra -std=c++14 math_func.a
// ./hybrid_ML_prune
