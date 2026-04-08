// ML_DOA原始版//不動//約1850 ms//有改過trace
// 全float//用LU分解求inverse，0.1度(1200次)搜-60~+60
// g++ -mavx512f -g -o ML_DOA ML_DOA.c -Wall -Wextra -std=c++14 math_func.a
// ./ML_DOA
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
#include <avx512fintrin.h>
//----------------------global variable---------------------------
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

void print_complex_matrix(float *matA_re, float *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%.2f ", matA_re[i * colA + j]);
            printf("+ %.2fi", matA_im[i * colA + j]);
        }
        printf("\n");
    }
}

// complex matrix addition
/*
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{   
    // Loop over blocks of 8 elements (AVX-512 can process 8 single-precision floats at once)
    float *temp_re = (float *)malloc(rowA * colA * sizeof(float));
    float *temp_im = (float *)malloc(rowA * colA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    float *temp_re2 = (float *)malloc(rowA * colA * sizeof(float));
    float *temp_im2 = (float *)malloc(rowA * colA * sizeof(float));
    memcpy(temp_re2, matB_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im2, matB_im, (rowA * colA * sizeof(float)));
    for (int i = 0; i <rowA * colA ; i += 16)
    {
        __m512 a_re = _mm512_loadu_ps(&temp_re[i]);
        __m512 a_im = _mm512_loadu_ps(&temp_im[i]);
        __m512 b_re = _mm512_loadu_ps(&temp_re2[i]);
        __m512 b_im = _mm512_loadu_ps(&temp_im2[i]);

        // Perform vectorized addition for real and imaginary parts
        a_re = _mm512_add_ps(a_re, b_re);
        a_im = _mm512_add_ps(a_im, b_im);

        // Store the results back to memory
        _mm512_storeu_ps(&matA_re[i], a_re);
        _mm512_storeu_ps(&matA_im[i], a_im);
    }
    free(temp_re);
    free(temp_im);
    free(temp_re2);
    free(temp_im2);
}
*/
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

__attribute__((aligned(32))) float matC_Re_sum[30000000] = {0.0}; // Real
__attribute__((aligned(32))) float matC_Im_sum[30000000] = {0.0}; // Imaginary
//__attribute__((aligned(32))) float matA_re[1000000] = {0.0};     // re_A
//__attribute__((aligned(32))) float matA_im[1000000] = {0.0};     // im_A
//-------------------------------------------------------------------
//__attribute__((aligned(32))) float matB_re[1000000] = {0.0}; // re_B
//__attribute__((aligned(32))) float matB_im[1000000] = {0.0}; // im_B
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
                re_A = _mm512_loadu_ps(&matA_re[i * rowB + AVX * k]);  //20x20:[0],[16],[20],[36],[40],[56]
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
// get complex matrix by column
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
void complex_matrix_get_columns(float *matA_re, float *matA_im, float *matCol_re, float *matCol_im, int rowA, int colA, int colTarget)
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
    float *temp_re = (float *)malloc(colA * rowA * sizeof(float));
    float *temp_im = (float *)malloc(colA * rowA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    complex_matrix_conjugate_transpose(temp_re, temp_im, rowA, colA);
    complex_matrix_multiplication(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA);

    free(temp_re);
    free(temp_im);
}

/*
void trace(float *theta_re, float *theta_im, float *S_ML_re, float *S_ML_im, int rowA, int colA, int i){
    float temp_re[0] ;
    float temp_im[0] ;
    for(int i=0; i < rowA ;i++){
        temp_re[0] = theta_re[i * colA + i]+temp_re[0];
        temp_im[0] = theta_im[i * colA + i]+temp_im[0];
    }
    S_ML_re[i] = temp_re[0];
    S_ML_im[i] = temp_im[0];
}
*/
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


//*
void ML_DOA_2A_CPU(int M, int qr_iter, float *angle, int number_angle, float *result, int SNR) {
    //-------------------------------------------------------------------
    float time_MLre = 0.0;
    float timeMLre_start, timeMLre_end; // Total ML Algorithm time
    // Parameter initialize
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_ML_start, time_ML_end, time_ML_diff;          // time initial
    struct timeval time_findmax_start, time_findmax_end, time_findmax_diff;          // time initial
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
    //---------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------------------------------------
    const int fc = 2.5e+9;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.25;
    float kc =  2*PI / lemda;
    const int nd = 512;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    gettimeofday(&time_findmax_start, NULL);
    printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        printf("%.1f, ", angle[a]);
    }
    printf("\n");
    // A_theta matrix (M, length of t_theta)
    //-----------------------------------------------------------------------------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
            //printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
        }
    }
    // t_sig matrix (length of t_theta, nd)
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
    //---------------------------------------------------------------
    float *sig_co_re = (float *)malloc(M * nd * sizeof(float));
    float *sig_co_im = (float *)malloc(M * nd * sizeof(float)); 
    //---------------------------------------------------------------
    float *x_r_re = (float *)malloc(M * nd * sizeof(float));
    float *x_r_im = (float *)malloc(M * nd * sizeof(float));
    //---------------------------------------------------------------
    // compute sig_co
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    // print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    gettimeofday(&time_AWGN_end, NULL);

// receiver
    // x_r matrix (M, nd)
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    //----------------------------------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------------------------------
// ml algorithm
    // R_xx matrix (M, M)
    float *R_xx_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_im = (float *)malloc(M * M * sizeof(float));
    // matlab code:  (R_xx = (1 / M )* x_r * x_r')
    float M_re = M;
    float M_im = 0.0;
    float *M_ptr = &M_re;
    float *M_ptr_im = &M_im;
    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int i = 0; i < M * M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    timeMLre_start = clock();
    //----------------------------------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------
    // array pattern
    // parameter setting
    const int len_dth = 1201;
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -60 + 0.1*i;
        dr[i] = dth[i] * PI / 180;
    }
    gettimeofday(&time_findmax_end, NULL);
    // compute S_ML_dB
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
    //float *eye_re = (float *)malloc(M * M * sizeof(float));
    //float *eye_im = (float *)malloc(M * M * sizeof(float));
    
    gettimeofday(&time_ML_start, NULL);
    //compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    
    
    //a_vector = M * 1
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
        complex_matrix_multiplication( a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, M, 1);

        //[inv(A^H*A)] =1*1  
        //print_complex_matrix(AH_mulA_re, AH_mulA_im, 1,1);
        matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        //cpp_division2(1,0,AH_mulA_re,AH_mulA_im,AH_mulA_inv_re,AH_mulA_inv_im);

        //print_complex_matrix(AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA2_re, AH_mulA2_im, 1, 1, M);

        // P_A = A*[inv(A^H*A)]*A^H = M*M
        complex_matrix_multiplication(a_vector_re, a_vector_im, AH_mulA2_re, AH_mulA2_im,AH_mulA3_re, AH_mulA3_im, M, 1, M);

        
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, R_xx_re, R_xx_im, theta_re, theta_im, M, M, M);

        // trace{P_A*R} 
        
        trace(theta_re, theta_im, S_ML_re, S_ML_im, M, M, i);
        //printf("re=\t%.2f\n ", S_ML_re[i]);
        //printf("im=\t%.2f ", S_ML_im[i]);
        //S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
		
        // compute S_ML_dB
    }
    free(a_temp_re);
    free(a_temp_im);
    
    
    for(int i = 0; i < len_dth; i++) {
        //S_ML_dB[i] = S_ML_re[i];
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
        //printf(" (%d):\t%.2f \n", i, S_ML_dB[i]);
        //printf(" \t%.2f \n", S_ML_dB[i]);
        //output_dB=
    }

    timeMLre_end = clock();
    printf(RED "ML REAL TIME:%.5f" CLOSE, (timeMLre_end - timeMLre_start) / CLOCKS_PER_SEC * 1000);
    gettimeofday(&time_ML_end, NULL);
    //--------------------------------------------------------------------------------------------------------------------------------
    // find Max and position
    
    printf("\n");
    double max_temp = S_ML_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        //printf(" %.4f, " , S_ML_dB[i]);
        if(S_ML_dB[i] > max_temp) {
            max_temp = S_ML_dB[i];
            position =  i ; // position = i;   //+的 position =  600+(4+i)/2 //-的 position =  i/2
        }

       // printf("i= %d, max_dB=\t%.2f \n", i, max_temp);
        //printf("i= %d, max0_dB=\t%.2f \n", i,S_ML_dB[i]);
    }
    
    
    

    float time_ML;  
    timersub(&time_ML_end, &time_ML_start, &time_ML_diff);
    time_ML = time_ML_diff.tv_usec;
    //printf("Total calculate ML time: \t\t%.3f(ms)\n", time_ML / 1000);

    float time_findmax;  
    timersub(&time_findmax_end, &time_findmax_start, &time_findmax_diff);
    time_findmax = time_findmax_diff.tv_usec;
    //printf("Total calculate findmax time: \t\t%.3f(ms)\n", time_findmax / 1000);

    float time_AWGN;
    timersub(&time_AWGN_end, &time_AWGN_start, &time_AWGN_diff);
    time_AWGN = time_AWGN_diff.tv_usec;
    printf("Total AWGN time: \t%.3f(ms)\n", time_AWGN / 1000);
    //-------------------------------------------------------------------
    // timersub function
    //-------------------------------------------------------------------
    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    //printf("Min_theta :\t\t%f(dB)\n", min_temp);
    /*
    free(t_theta);
    free(A_theta_re);
    free(A_theta_im);
    free(t_sig_re);
    free(t_sig_im);
    free(sig_co_re);
    free(sig_co_im); 
    free(x_r_re);
    free(x_r_im);
    free(R_xx_re);
    free(R_xx_im);
    free(dth);
    free(dr); 
    free(a_vector_re);
    free(a_vector_im);
    free(S_ML_re);
    free(S_ML_im); 
    free(S_ML_dB);
    
    
    free(theta_re);
    free(theta_im);
    free(AH_mulA_re);
    free(AH_mulA_im);
    free(AH_mulA2_re);
    free(AH_mulA2_im);
    free(AH_mulA3_re);
    free(AH_mulA3_im);
    free(AH_mulA_inv_re);
    free(AH_mulA_inv_im);
    */
    //free(eye_re);
    //free(eye_im);
   
}
//*/

int main()
{
    //-------------------------------------------------------------------
    // Parameter initialize
    float time_ML = 0.0;
    float timeML_start, timeML_end; // Total MUSIC Algorithm time
    //-------------------------------------------------------------------
    float angle[100] = {20.2, 30, 20, 60}; // angle of array
    int number_angle = 1;
    int M =  64;
    int snr = 10;
    int qr_iter = 10;
    float result[3] = {0};
    // int angle = 50;
    int iter = 1;

    //=================== ML Algorithm =================================
    timeML_start = clock();
    ML_DOA_2A_CPU(M, qr_iter, &angle[0], number_angle, result, snr);
    timeML_end = clock();
    //printf("--------------------------------------\n");
    //printf(L_GREEN "Total ML time : \t%.3f(ms)\n" CLOSE, (timeML_end - timeML_start) / CLOCKS_PER_SEC * 1000);
    
}


    //
    /* 
    float *AH_mulA_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * M * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(M * M * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(M * M * sizeof(float));
    Complex_F A[M * M];
    Complex_F A_inv[M * M];
    gettimeofday(&time_findmax_end, NULL);
    gettimeofday(&time_ML_start, NULL);
    //compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    / a_vector = 1*M
    for(int i = 0; i < len_dth; ++i) { // can be paralleled to compute S_ML_dB
        for(int j = 0; j < M; ++j) {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            cpp_exp2(&a_temp_re[j], &a_temp_im[j], dr, d, kc, i, j);
        } 
		///不能用compute_S_MUSIC，要自己寫
        //a_vector_conjugate_transpose A^H = M*1
        complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, M);  
        //A^H*A = M* M
        complex_matrix_multiplication( a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, M, 1, M);
        //[inv(A^H*A)] =M*M  
        for (int k = 0; k < M * M; k++)
        {
            A[k].real = AH_mulA_re[k];
            A[k].imag = AH_mulA_im[k];
        }
        Inverse_LU(A, A_inv, M , M);
        for (int k = 0; k < M * M; k++)
        {
            AH_mulA_inv_re[k] = A_inv[k].real;
            AH_mulA_inv_im[k] = A_inv[k].imag;
        }
        // [inv(A^H*A)]*A^H = M*1
        complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA_re, AH_mulA_im, M, M, 1);
        // P_A = A*[inv(A^H*A)]*A^H = 1*1
        complex_matrix_multiplication(a_vector_re, a_vector_im, a_temp_re, a_temp_im,AH_mulA_re, AH_mulA_im, 1, M, 1);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA_re, AH_mulA_im,R_xx_re, R_xx_im, theta_re, theta_im, M, M, M);
        for(int j=0 ; j<M*M ; j++ ){
            theta_re[j] = AH_mulA_re[0]*R_xx_re[j];
            theta_im[j] = AH_mulA_im[0]*R_xx_im[j];
        }
        // trace{P_A*R} 
        trace(theta_re, theta_im, &S_ML_re[i], &S_ML_im[i], M, M,i);
		printf("---\n");
        printf("re=\t%.2f ", S_ML_re[i]);
        printf("im=\t%.2f ", S_ML_im[i]);
        printf("---\n");
        //*
        // compute S_ML_dB
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
    }
    */