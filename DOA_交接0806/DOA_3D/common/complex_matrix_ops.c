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
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
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


// ================================
// ========== matrix ops ==========
// ================================

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

void print_complex_matrix(float *matA_re, float *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%.5f ", matA_re[i * colA + j]);
            printf("+ %.5fi", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}

void multiply_matrix_avx(float *A, float multiplier, int row, int col) {
    int size = row*col;
    __m512 vec_multiplier = _mm512_set1_ps(multiplier);
    
    int i;
    for (i = 0; i <= size-AVX; i +=AVX) {
        __m512 vec_A = _mm512_loadu_ps(&A[i]);
        
        __m512 vec_result = _mm512_mul_ps(vec_A, vec_multiplier);
        
        _mm512_storeu_ps(&A[i], vec_result);
    }
    
    for (; i < size; i++) {
        A[i] *= multiplier;
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

__attribute__((aligned(64))) float matC_Re_sum[30000000] = {0.0}; // Real
__attribute__((aligned(64))) float matC_Im_sum[30000000] = {0.0}; // Imaginary
//-------------------------------------------------------------------
__attribute__((aligned(64))) float matC_Real[30000000] = {0.0}; // re_C
__attribute__((aligned(64))) float matC_Imag[30000000] = {0.0}; // im_C
__attribute__((aligned(64))) float buffer[10000000] = {0.0};

// remainder use storeu
void complex_matrix_multiplication_storeu(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------

    // Parameter initialize
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    gettimeofday(&start_multiply, NULL);                           // start

    //-------------------------------------------------------------------
    matrix_transpose(matB_re, matB_im, rowB, colB);              // Matrix transpose
    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    //memset(matC_Re_sum, 0, rowA * colB * sizeof(double));
    //memset(matC_Im_sum, 0, rowA * colB * sizeof(double));
    // memset(matA_re, 0, rowA * colB * sizeof(float));
    // memset(matA_im, 0, rowA * colB * sizeof(float));
    // memset(matB_re, 0, rowA * colB * sizeof(float));
    // memset(matB_im, 0, rowA * colB * sizeof(float));
    memset(matC_Real, 0, AVX * sizeof(float));
    memset(matC_Imag, 0, AVX * sizeof(float));
    //----------------------------------------------------------
    //----------------------------------------------------------
    __m512 re_A, re_B, re_C; // simd 256 for matrix real part
    __m512 im_A, im_B, im_C; // simd 256 for matrix Imaginary part

    memset(matC_re, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Real)
    memset(matC_im, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Imaginary)


    //----------------------------------------------------------
    for (int k = 0; k < ((rowB - 1) / AVX + 1); k++)
    {
        for (int i = 0; i < rowA; i++)
        {
            re_A = _mm512_loadu_ps(&matA_re[i * rowB + AVX * k]);
            im_A = _mm512_loadu_ps(&matA_im[i * rowB + AVX * k]);
            for (int j = 0; j < colB; j++)
            {
                re_B = _mm512_loadu_ps(&matB_re[j * rowB + AVX * k]);
                im_B = _mm512_loadu_ps(&matB_im[j * rowB + AVX * k]);
                
                //-------------------------------------------------------
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B)); // Re{ac-bd}
                //-------------------------------------------------------
                //-------------------------------------------------------
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B)); // Im{ad+bc}
                //-------------------------------------------------------
                if(k<((rowB ) / AVX)){
                    matC_re[i*colB + j] +=_mm512_reduce_add_ps(re_C);
                    matC_im[i*colB + j] +=_mm512_reduce_add_ps(im_C);
                }
                else{
                    _mm512_storeu_ps(&matC_Real[0], re_C); // store Re value
                    _mm512_storeu_ps(&matC_Imag[0], im_C); // store Im value
                    for(int m = 0; m < rowB%AVX; m++)
                    {
                        matC_re[i*colB + j] += matC_Real[m];
                        matC_im[i*colB + j] += matC_Imag[m];
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------

    //-------------------------------------------------------------------
    matrix_transpose(matB_re, matB_im, colB, rowB); // Matrix transpse -> back to origin version
    //-------------------------------------------------------------------

    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}

// remainder use mask
void complex_matrix_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    gettimeofday(&start_multiply, NULL);                           // start

    //-------------------------------------------------------------------
    matrix_transpose(matB_re, matB_im, rowB, colB);              // Matrix transpose
    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    __m512 re_A, re_B, re_C; // simd 256 for matrix real part
    __m512 im_A, im_B, im_C; // simd 256 for matrix Imaginary part
    memset(matC_re, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Real)
    memset(matC_im, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Imaginary)
    //----------------------------------------------------------
    int full_chunks = rowB / AVX;
    int remainder   = rowB % AVX;
    
    for (int k = 0; k < full_chunks; k++) {
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_loadu_ps(&matA_re[i * rowB + k * AVX]);
            im_A = _mm512_loadu_ps(&matA_im[i * rowB + k * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_loadu_ps(&matB_re[j * rowB + k * AVX]);
                im_B = _mm512_loadu_ps(&matB_im[j * rowB + k * AVX]);
    
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B));
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B));
    
                matC_re[i * colB + j] += _mm512_reduce_add_ps(re_C);
                matC_im[i * colB + j] += _mm512_reduce_add_ps(im_C);
            }
        }
    }
    __mmask16 mask;
    if (remainder > 0) {
        mask = (1 << remainder) - 1;
    
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_maskz_loadu_ps(mask, &matA_re[i * rowB + full_chunks * AVX]);
            im_A = _mm512_maskz_loadu_ps(mask, &matA_im[i * rowB + full_chunks * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_maskz_loadu_ps(mask, &matB_re[j * rowB + full_chunks * AVX]);
                im_B = _mm512_maskz_loadu_ps(mask, &matB_im[j * rowB + full_chunks * AVX]);
    
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B));
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B));
    
                matC_re[i * colB + j] += _mm512_reduce_add_ps(re_C);
                matC_im[i * colB + j] += _mm512_reduce_add_ps(im_C);
            }
        }
    }
    //-------------------------------------------------------------------
    matrix_transpose(matB_re, matB_im, colB, rowB); // Matrix transpse -> back to origin version
    //------------------------------------------------------------
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}


void complex_matrix_multiplication_ijk(float *A_re,
                         float *A_im,
                         float *B_re,
                         float *B_im,
                         float       *C_re,
                         float       *C_im,
                         int rowA, int rowB, int colB)
{
    matrix_transpose(B_re, B_im, rowB, colB);
    memset(C_re, 0, (size_t)rowA * colB * sizeof(float));
    memset(C_im, 0, (size_t)rowA * colB * sizeof(float));

    const int chunks   = rowB / AVX;
    const int rem      = rowB % AVX;
    const __mmask16 m  = (1u << rem) - 1u;
    __m512 acc_re, acc_im, a_re, a_im, b_re, b_im, p_re, p_im;
    for (int i = 0; i < rowA; ++i) {

        const float *ai_re = A_re + (size_t)i * rowB;
        const float *ai_im = A_im + (size_t)i * rowB;

        for (int j = 0; j < colB; ++j) {

            const float *bj_re = B_re + (size_t)j * rowB;
            const float *bj_im = B_im + (size_t)j * rowB;

            acc_re = _mm512_setzero_ps();
            acc_im = _mm512_setzero_ps();

            for (int k = 0; k < chunks; ++k) {

                a_re = _mm512_loadu_ps(ai_re + k * AVX);
                a_im = _mm512_loadu_ps(ai_im + k * AVX);
                b_re = _mm512_loadu_ps(bj_re + k * AVX);
                b_im = _mm512_loadu_ps(bj_im + k * AVX);

                p_re = _mm512_sub_ps(
                                _mm512_mul_ps(a_re, b_re),
                                _mm512_mul_ps(a_im, b_im));
                p_im = _mm512_add_ps(
                                _mm512_mul_ps(a_re, b_im),
                                _mm512_mul_ps(a_im, b_re));


                acc_re = _mm512_add_ps(acc_re, p_re);
                acc_im = _mm512_add_ps(acc_im, p_im);
            }

            if (rem) {
                a_re = _mm512_maskz_loadu_ps(m, ai_re + chunks * AVX);
                a_im = _mm512_maskz_loadu_ps(m, ai_im + chunks * AVX);
                b_re = _mm512_maskz_loadu_ps(m, bj_re + chunks * AVX);
                b_im = _mm512_maskz_loadu_ps(m, bj_im + chunks * AVX);

                p_re = _mm512_sub_ps(
                                _mm512_mul_ps(a_re, b_re),
                                _mm512_mul_ps(a_im, b_im));
                p_im = _mm512_add_ps(
                                _mm512_mul_ps(a_re, b_im),
                                _mm512_mul_ps(a_im, b_re));

                acc_re = _mm512_add_ps(acc_re, p_re);
                acc_im = _mm512_add_ps(acc_im, p_im);
            }

            C_re[i * colB + j] = _mm512_reduce_add_ps(acc_re);
            C_im[i * colB + j] = _mm512_reduce_add_ps(acc_im);
        }
    }
    matrix_transpose(B_re, B_im, colB, rowB);
}

void complex_matrix_multiplication_scalar(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
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

// get complex matrix by column
void complex_matrix_get_columns(float *matA_re, float *matA_im, float *matCol_re, float *matCol_im, int rowA, int colA, int colTarget)
{
    for (int i = 0; i < rowA; ++i)
    {
        matCol_re[i] = matA_re[i * colA + colTarget];
        matCol_im[i] = matA_im[i * colA + colTarget];
    }
}

// get complex matrix by row
void complex_matrix_get_rows(float *matA_re, float *matA_im, float *matRow_re, float *matRow_im, int colA, int rowTarget)
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

void complex_matrix_conjugate_transpose_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float) + AVX*sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float) + AVX*sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    complex_matrix_conjugate_transpose(temp_re, temp_im, rowA, colA);
    complex_matrix_multiplication(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);

    free(temp_re);
    free(temp_im);
}