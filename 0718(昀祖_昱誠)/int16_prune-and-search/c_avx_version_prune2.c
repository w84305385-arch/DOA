// 都是float //範圍-60~+60度，30->10->3->1->0.1 //1度搜索時有某些角度重疊2度
// g++ -mavx512f -g -o c_avx_version_prune2  c_avx_version_prune2.c -Wall -Wextra -std=c++14 math_func.a
// ./c_avx_version_prune2
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
            printf("\t%.5f ", matA_re[i * colA + j]);
            printf("+ %.5fi", matA_im[i * colA + j]);
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
    float *Q_col_re = (float *)malloc(row * 1 * sizeof(float));
    float *Q_col_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *vector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *vector_cur_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *Qvector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *Qvector_cur_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *power_cur_re = (float *)malloc(sizeof(float));
    float *power_cur_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *power_val_re = (float *)malloc(sizeof(float));
    float *power_val_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *proj_val_re = (float *)malloc(sizeof(float));
    float *proj_val_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *proj_Qvector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *proj_Qvector_cur_im = (float *)malloc(row * 1 * sizeof(float));
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
        //---------------------------------------------------------------
        //---------------------------------------------------------------
        complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col_re, Q_col_im, row, 1); // A Col -> A row [a,a,a,a,a,a,]
        memset(power_cur_re, 0, sizeof(float));
        memset(power_cur_im, 0, sizeof(float));

        complex_matrix_conjugate_transpose_multiplication(Q_col_re, Q_col_im, power_cur_re, power_cur_im, 1, row);
        //---------------------------------------------------------------
        // printf(L_BLUE "re = %.2f, im = %.2f, power_cur[0] = %.2f\n" CLOSE, power_cur_re[0], power_cur_im[0], power_cur_re[0]);
        cpp_sqrt(&power_cur_re[0], &power_cur_im[0]);
        // printf(YELLOW "power_cur[0] = %.2f\n" CLOSE, power_cur_re[0]);
        //---------------------------------------------------------------
        //   calculate q2 vector
        if (i > 0)
        {
            complex_matrix_get_columns(A_re, A_im, vector_cur_re, vector_cur_im, row, col, i);
            float *Q_col_proj_re = (float *)malloc(row * i * sizeof(float));
            float *Q_col_proj_im = (float *)malloc(row * i * sizeof(float));
            memset(Q_col_proj_re, 0, row * i * sizeof(float));
            //---------------------------------------------------------------
            float *proj_vector_re = (float *)malloc(i * sizeof(float));
            float *proj_vector_im = (float *)malloc(i * sizeof(float));
            //---------------------------------------------------------------
            memset(proj_vector_re, 0, i * sizeof(float));
            memset(proj_vector_im, 0, i * sizeof(float));
            //---------------------------------------------------------------
            // printf("Q_col_proj_re = ");
            // for (int i = 0; i < 20; i++)
            // {
            //     printf(PURPLE "%.0f, " CLOSE, Q_col_proj_re[i]);
            // }
            // printf("\n");
            //---------------------------------------------------------------
            // printf(RED "Q_col_proj_re = " CLOSE);
            for (int j = 0; j < i; ++j)
            {
                for (int m = 0; m < row; ++m)
                {
                    Q_col_proj_re[m * i + j] = Q_re[m * col + j];
                    Q_col_proj_im[m * i + j] = Q_im[m * col + j];
                    // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }
            // printf("\n");
            //---------------------------------------------------------------
            complex_matrix_conjugate_transpose(Q_col_proj_re, Q_col_proj_im, row, i);
            complex_matrix_multiplication(Q_col_proj_re, Q_col_proj_im, vector_cur_re, vector_cur_im, proj_vector_re, proj_vector_im, i, row, 1);
            complex_matrix_conjugate_transpose(Q_col_proj_re, Q_col_proj_im, i, row);

            //---------------------------------------------------------------
            memset(Q_col_re, 0, row * 1 * sizeof(float));
            memset(Q_col_im, 0, row * 1 * sizeof(float));
            //---------------------------------------------------------------
            complex_matrix_multiplication(Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, Q_col_re, Q_col_im, row, i, 1);
            complex_matrix_subtraction(vector_cur_re, vector_cur_im, Q_col_re, Q_col_im, row, 1);

            for (int m = 0; m < row; m++)
            {
                Q_re[m * col + i] = vector_cur_re[m];
                Q_im[m * col + i] = vector_cur_im[m];
            }
            // A=QR
            for (int j = 0; j < i; ++j)
            {
                R_re[i + col * j] = proj_vector_re[j];
                R_im[i + col * j] = proj_vector_im[j];
            }
            free(Q_col_proj_re);
            free(Q_col_proj_im);
            free(proj_vector_re);
            free(proj_vector_im);
        }
        complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col_re, Q_col_im, row, 1);
        //---------------------------------------------------------------
        memset(power_val_re, 0, sizeof(float));
        memset(power_val_im, 0, sizeof(float));
        //---------------------------------------------------------------
        complex_matrix_conjugate_transpose_multiplication(Q_col_re, Q_col_im, power_val_re, power_val_im, 1, row);
        cpp_sqrt(&power_val_re[0], &power_val_im[0]);

        // 1e-4 = 0.0001
        if (power_val_re[0] / power_cur_re[0] < 1e-4)
        {
            R_re[i * row + i] = 0;
            R_im[i * row + i] = 0;
            // span again
            for (int m = 0; m < row; ++m)
            {
                Q_re[m * col + i] = 0;
                Q_im[m * col + i] = 0;
            }
            Q_re[i * row + i] = 1;
            complex_matrix_get_columns(Q_re, Q_im, vector_cur_re, vector_cur_im, row, col, i);
            for (int j = 0; j < i; j++)
            {
                complex_matrix_get_columns(Q_re, Q_im, Qvector_cur_re, Qvector_cur_im, row, col, j);
                //---------------------------------------------------------------
                memset(power_val_re, 0, sizeof(float));
                memset(power_val_im, 0, sizeof(float));
                //---------------------------------------------------------------
                complex_matrix_conjugate_transpose(Qvector_cur_re, Qvector_cur_im, row, 1);
                complex_matrix_multiplication(Qvector_cur_re, Qvector_cur_im, vector_cur_re, vector_cur_im, proj_val_re, proj_val_im, 1, row, 1);
                complex_matrix_conjugate_transpose(Qvector_cur_re, Qvector_cur_im, 1, row);
                complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
                //---------------------------------------------------------------
                memset(proj_Qvector_cur_re, 0, row * 1 * sizeof(float));
                memset(proj_Qvector_cur_re, 0, row * 1 * sizeof(float));
                //---------------------------------------------------------------
                complex_matrix_multiplication(Qvector_cur_re, Qvector_cur_im, proj_val_re, proj_val_im, proj_Qvector_cur_re, proj_Qvector_cur_im, row, 1, 1);
                complex_matrix_subtraction(Q_col_re, Q_col_im, proj_Qvector_cur_re, proj_Qvector_cur_im, row, 1);
                for (int m = 0; m < row; ++m)
                {
                    Q_re[m * col + i] = Q_col_re[m];
                    Q_im[m * col + i] = Q_col_im[m];
                }
            }
            complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
            complex_matrix_conjugate_transpose(Q_col_re, Q_col_im, row, 1);
            //---------------------------------------------------------------
            memset(power_val_re, 0, sizeof(float));
            memset(power_val_re, 0, sizeof(float));
            //---------------------------------------------------------------
            complex_matrix_conjugate_transpose_multiplication(Q_col_re, Q_col_im, power_val_re, power_val_im, 1, row);
            cpp_sqrt(&power_val_re[0], &power_val_im[0]);

            complex_matrix_conjugate_transpose(Q_col_re, Q_col_im, 1, row);
            for (int m = 0; m < row; m++)
            {
                // Q_re[m * col + i] /= power_val_re[0];
                // Q_im[m * col + i] /= power_val_im[0];
                cpp_division(&Q_re[m * col + i], &Q_im[m * col + i], &power_val_re[0], &power_val_im[0]);
            }
        }
        else
        {
            R_re[i * row + i] = power_val_re[0];
            R_im[i * row + i] = power_val_im[0];
            for (int m = 0; m < row; ++m)
            {
                // printf(YELLOW "cpp_division(%.2f,%.2f) /= power_val(%.2f,%.2f)", Q_re[m * col + i], Q_im[m * col + i], power_val_re[0], power_val_im[0]);
                cpp_division(&Q_re[m * col + i], &Q_im[m * col + i], &power_val_re[0], &power_val_im[0]);
                // printf(" = Q[%d](%.2f,%.2f)\n" CLOSE, m * col + i, Q_re[m * col + i], Q_im[m * col + i]);
            }
            // printf("\n");
        }
    }
    free(Q_col_re);
    free(Q_col_im);
    free(vector_cur_re);
    free(vector_cur_im);
    free(Qvector_cur_re);
    free(Qvector_cur_im);
    free(power_cur_re);
    free(power_cur_im);
    free(power_val_re);
    free(power_val_im);
    free(proj_val_re);
    free(proj_val_im);
    free(proj_Qvector_cur_re);
    free(proj_Qvector_cur_im);
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

void MUSIC_DOA_2A_CPU_test(int M, int qr_iter, float *angle, int number_angle, float *result, int SNR)
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
           // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
    for (int i = 0; i < M * M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    timeMusicre_start = clock();
    // compute eigenvector Ve (M, M)
    //---------------------------------------------------------------
    float *Ve_re = (float *)malloc(M * M * sizeof(float));
    float *Ve_im = (float *)malloc(M * M * sizeof(float));
    float *De_re = (float *)malloc(M * M * sizeof(float));
    float *De_im = (float *)malloc(M * M * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Eigen_start, NULL);
    eigen(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, M, M, qr_iter);
    gettimeofday(&time_Eigen_end, NULL);
    //printf("----------Ve------------\n");
    //print_complex_matrix(Ve_re, Ve_im, M, M);
    //printf("----------De------------\n");
    //print_complex_matrix(De_re, De_im, M, M);
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
    //---------------------------------------------------------------
    float *Pn_re = (float *)calloc(M * M, sizeof(float));
    float *Pn_im = (float *)calloc(M * M, sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    gettimeofday(&time_Pn_end, NULL);
    //---------------------------------------------------------------
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, M, M);
    //---------------------------------------------------------------
    // 取定點
    // parameter setting
    const int len_dth = 5; //30度
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 30 * i; //-60 -30 0 30 60
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
        /*
        for(int j = 1; j < M; j+=4)
        {
            a_vector_re[j] = 0;
            a_vector_im[j] = 0;
        }
        for (int j = 0; j < M; ++j)
        {
            printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        */
        compute_S_MUSIC(a_vector_re, a_vector_im, Pn_re, Pn_im, M, &S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSIC_dB[i] = cpp_20log_abs(&S_MUSIC_re[i], &S_MUSIC_im[i]);
        //printf("S_MUSIC_dB = %.4f\n", S_MUSIC_dB[i]);

        fprintf(fp_excel, "%.1f,%.4f\n", (-60 + 30 * i), S_MUSIC_dB[i]);
    }
    fclose(fp_excel);
    //---------------------------------------------------------------
    // find Max and position
    double max_temp = S_MUSIC_dB[0];
    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        printf("S_MUSIC_dB[%d] = %.4f\n",i , S_MUSIC_dB[i]);
        if (S_MUSIC_dB[i] > max_temp)
        {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }

    }
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //10度10度找//30度內搜索時重疊10度並以10度為單位搜索
    const int len_dthA = 5;                                  //實際搜索      -60~-40 -50~-10 -20~20 10~50 40~60 overlap 10度
    float *dthA = (float *)malloc(len_dthA * sizeof(float)); //dth[position]  -60      -30     0     30    60 
    float *drA = (float *)malloc(len_dthA * sizeof(float));
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
    printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    double max_tempA = S_MUSICA_dB[0];
    int positionA = 0;
    for (int i = 0; i < len_dthA; ++i)
    {
        if (S_MUSICA_dB[i] > max_tempA)
        {
            max_tempA = S_MUSICA_dB[i];
            positionA = i;
        }
        //printf("max_temp1 = %.4f,(%d)\n", max_temp1,i);
    }

    //-------------------------------------------------------------------
    printf("---\n");
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
    float *a_vectorB_re = (float *)malloc(M * sizeof(float));
    float *a_vectorB_im = (float *)malloc(M * sizeof(float));
    float *S_MUSICB_re = (float *)malloc(len_dthB * sizeof(float));
    float *S_MUSICB_im = (float *)malloc(len_dthB * sizeof(float));
    float *S_MUSICB_dB = (float *)malloc(len_dthB * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < len_dthB; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vectorB_re[j], &a_vectorB_im[j], drB, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vectorB_re, a_vectorB_im, Pn_re, Pn_im, M, &S_MUSICB_re[i], &S_MUSICB_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSICB_dB[i] = cpp_20log_abs(&S_MUSICB_re[i], &S_MUSICB_im[i]);
        //printf("S_MUSIC1_dB = %.4f\n", S_MUSIC1_dB[i]);
    }
    printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    double max_tempB = S_MUSICB_dB[0];
    int positionB = 0;
    for (int i = 0; i < len_dthB; ++i)
    {
        if (S_MUSICB_dB[i] > max_tempB)
        {
            max_tempB = S_MUSICB_dB[i];
            //printf("max_temp1 = %.4f\n", max_temp1);
            positionB = i;
        }
    }
    //---------------------------------------------------------------
    //1度1度找//3度內以1度為單位搜索
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
    float *a_vectorC_re = (float *)malloc(M * sizeof(float));
    float *a_vectorC_im = (float *)malloc(M * sizeof(float));
    float *S_MUSICC_re = (float *)malloc(len_dthC * sizeof(float));
    float *S_MUSICC_im = (float *)malloc(len_dthC * sizeof(float));
    float *S_MUSICC_dB = (float *)malloc(len_dthC * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < len_dthC; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vectorC_re[j], &a_vectorC_im[j], drC, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vectorC_re, a_vectorC_im, Pn_re, Pn_im, M, &S_MUSICC_re[i], &S_MUSICC_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSICC_dB[i] = cpp_20log_abs(&S_MUSICC_re[i], &S_MUSICC_im[i]);
        //printf("S_MUSIC1_dB = %.4f\n", S_MUSIC1_dB[i]);
    }
    printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    double max_tempC = S_MUSICC_dB[0];
    int positionC = 0;
    for (int i = 0; i < len_dthC; ++i)
    {
        if (S_MUSICC_dB[i] > max_tempC)
        {
            max_tempC = S_MUSICC_dB[i];
            //printf("max_temp1 = %.4f\n", max_temp1);
            positionC = i;
        }
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
    float *a_vectorD_re = (float *)malloc(M * sizeof(float));
    float *a_vectorD_im = (float *)malloc(M * sizeof(float));
    float *S_MUSICD_re = (float *)malloc(len_dthD * sizeof(float));
    float *S_MUSICD_im = (float *)malloc(len_dthD * sizeof(float));
    float *S_MUSICD_dB = (float *)malloc(len_dthD * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < len_dthD; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vectorD_re[j], &a_vectorD_im[j], drD, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vectorD_re, a_vectorD_im, Pn_re, Pn_im, M, &S_MUSICD_re[i], &S_MUSICD_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSICD_dB[i] = cpp_20log_abs(&S_MUSICD_re[i], &S_MUSICD_im[i]);
        //printf("S_MUSIC1_dB = %.4f\n", S_MUSIC1_dB[i]);
    }
    printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    double max_tempD = S_MUSICD_dB[0];
    int positionD = 0;
    for (int i = 0; i < len_dthD; ++i)
    {
        if (S_MUSICD_dB[i] > max_tempD)
        {
            max_tempD = S_MUSICD_dB[i];
            //printf("max_temp1 = %.4f\n", max_temp1);
            positionD = i;
        }
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
    printf("Total AWGN time: \t%.3f(ms)\n", time_AWGN / 1000);
    printf("Total Eigen time: \t%.3f(ms)\n", time_Eigen / 1000);
    printf("Total Pn time: \t\t%.3f(ms)\n", time_Pn / 1000);
    //-------------------------------------------------------------------
    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    
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

    printf(L_GREEN "Total MUSIC REAL time : \t%.3f(ms)\n" CLOSE, (timeMusicre_end - timeMusicre_start) / CLOCKS_PER_SEC * 1000);
}

void MVDR_DOA_2A_CPU_test(int M, int qr_iter, float *angle, int number_angle, float *result, int SNR)
{
    float time_MVDRre = 0.0;
    float timeMVDRre_start, timeMVDRre_end; // Total MUSIC Algorithm time
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    //-------------------------------------------------------------------
    printf("---------------\n");
    printf("--MVDR DOA--\n");
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
            // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
            // printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
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
    // print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    gettimeofday(&time_AWGN_end, NULL);
    // for (int a = 0; a < M * nd; a++)
    // {
    //     printf("\t(%f,%f)\n", x_r_re[a], x_r_im[a]);
    // }
    //---------------------------------------------------------------
    float *R_xx_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_im = (float *)malloc(M * M * sizeof(float));
    float M_re = M;
    float M_im = 0.0;
    float *M_ptr = &M_re;
    float *M_ptr_im = &M_im;

    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd); // notice!
    for (int i = 0; i < M * M; ++i)
    {
        // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    timeMVDRre_start = clock();
    // compute eigenvector Ve (M, M)
    //---------------------------------------------------------------
    float *Ve_re = (float *)malloc(M * M * sizeof(float));
    float *Ve_im = (float *)malloc(M * M * sizeof(float));
    float *De_re = (float *)malloc(M * M * sizeof(float));
    float *De_im = (float *)malloc(M * M * sizeof(float));
    float *De_result = (float *)malloc(M * M * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Eigen_start, NULL);
    eigen(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, M, M, qr_iter);
    gettimeofday(&time_Eigen_end, NULL);
    // printf("----------Ve------------\n");
    // print_complex_matrix(Ve_re, Ve_im, M, M);
    // printf("----------De------------\n");
    // print_complex_matrix(De_re, De_im, M, M);
    //---------------------------------------------------------------
    float *vet_noise_re = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    float *vet_noise_im = (float *)malloc(M * (M - len_t_theta) * sizeof(float));
    //---------------------------------------------------------------

    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = len_t_theta; j < M; ++j)
    //     {
    //         vet_noise_re[i * (M - len_t_theta) + j - len_t_theta] = Ve_re[i * M + j];
    //         vet_noise_im[i * (M - len_t_theta) + j - len_t_theta] = Ve_im[i * M + j];
    //         // printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
    //     }
    // }
    // //---------------------------------------------------------------
    float *R_xx_inv_1_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_inv_1_im = (float *)malloc(M * M * sizeof(float));
    float *Pn_re = (float *)calloc(M * M, sizeof(float));
    float *Pn_im = (float *)calloc(M * M, sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    //compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    gettimeofday(&time_Pn_end, NULL);
    float temp_real = 1;
    float temp_imag = 0;
    for (int i = 0; i < M * M; i += (M + 1))
    {
        cpp_abs(&De_re[i], &De_im[i], &De_result[i]);
        if (abs(De_result[i]) < 0.00000001)
        {
            De_re[i] = 1000000;
            De_im[i] = 0;
        }
        else
        {
            cpp_division3(&temp_real, &temp_imag, &De_re[i], &De_im[i]);
        }
    }

    complex_matrix_multiplication(Ve_re, Ve_im, De_re, De_im, R_xx_inv_1_re, R_xx_inv_1_im, M, M, M);
    complex_matrix_conjugate_transpose(Ve_re, Ve_im, M, M);
    complex_matrix_multiplication(R_xx_inv_1_re, R_xx_inv_1_im, Ve_re, Ve_im, Pn_re, Pn_im, M, M, M);
    // printf("----------R_xx------------\n");
    // print_complex_matrix(R_xx,M,M);
    // printf("----------R_xx_inv_1------------\n");
    // print_complex_matrix(R_xx_inv_1,M,M);
    //---------------------------------------------------------------
    // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, M, M);

    // array pattern
    // parameter setting
    // const int len_dth = 401;
    const int len_dth = 121;
    float *dth = (float *)malloc(len_dth * sizeof(float));
    float *dr = (float *)malloc(len_dth * sizeof(float));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    //---------------------------------------------------------------
    float *a_vector_re = (float *)malloc(M * sizeof(float));
    float *a_vector_im = (float *)malloc(M * sizeof(float));
    float *S_MVDR_re = (float *)malloc(len_dth * sizeof(float));
    float *S_MVDR_im = (float *)malloc(len_dth * sizeof(float));
    float *S_MVDR_dB = (float *)malloc(len_dth * sizeof(float));
    //---------------------------------------------------------------

    
    for (int i = 0; i < len_dth; ++i)
    {
        // can be paralleled to compute S_MVDR_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            // printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector_re, a_vector_im, Pn_re, Pn_im, M, &S_MVDR_re[i], &S_MVDR_im[i]);
        // printf("\tS_MVDR(%f,%f), ", S_MVDR_re[i], S_MVDR_im[i]);
        S_MVDR_dB[i] = cpp_20log_abs(&S_MVDR_re[i], &S_MVDR_im[i]);
        // printf("S_MVDR_dB = %.4f\n", S_MVDR_dB[i]);


    }
    
   
    //---------------------------------------------------------------
    // find Max and position
    double max_temp = S_MVDR_dB[0];
    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        if (S_MVDR_dB[i] > max_temp)
        {
            max_temp = S_MVDR_dB[i];
            position = i;
        }
    }
    const int len_dth1 = 21; //前後10點
    float *dth1 = (float *)malloc(len_dth1 * sizeof(float));
    float *dr1 = (float *)malloc(len_dth1 * sizeof(float));
    for (int i = 0; i < len_dth1; ++i)  
    { // do only one time, no need to be paralleled
        dth1[i] = (-60+(position-1)) + 0.1 * i;
        //printf("%f\n",dth1[i]);
        dr1[i] = dth1[i] * PI / 180;
    }
    //-------------------------------------------------------------------
    float *a_vector1_re = (float *)malloc(M * sizeof(float));
    float *a_vector1_im = (float *)malloc(M * sizeof(float));
    float *S_MUSIC1_re = (float *)malloc(len_dth1 * sizeof(float));
    float *S_MUSIC1_im = (float *)malloc(len_dth1 * sizeof(float));
    float *S_MUSIC1_dB = (float *)malloc(len_dth1 * sizeof(float));
    //---------------------------------------------------------------
    
    for (int i = 0; i < len_dth1; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vector1_re[j], &a_vector1_im[j], dr1, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector1_re, a_vector1_im, Pn_re, Pn_im, M, &S_MUSIC1_re[i], &S_MUSIC1_im[i]);
        //printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSIC1_dB[i] = cpp_20log_abs(&S_MUSIC1_re[i], &S_MUSIC1_im[i]);
        //printf("S_MUSIC1_dB = %.4f\n", S_MUSIC1_dB[i]);

    }
    printf("---\n");
    //---------------------------------------------------------------
    // find Max and position
    double max_temp1 = S_MUSIC1_dB[0];
    int position1 = 0;
    for (int i = 0; i < len_dth1; ++i)
    {
        if (S_MUSIC1_dB[i] > max_temp1)
        {
            max_temp1 = S_MUSIC1_dB[i];
            //printf("max_temp1 = %.4f\n", max_temp1);
            position1 = i;
        }
    }
    timeMVDRre_end = clock();
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
    printf("position1 : \t\t%d\n", position1);
    printf(RED "Theta estimation1 :\t%.3f (degree)\n" CLOSE, dth1[position1]);
    printf("Max_theta1 :\t\t%f(dB)\n", max_temp1);    
    printf(L_GREEN "Total MVDR REAL time : \t%.3f(ms)\n" CLOSE, (timeMVDRre_end - timeMVDRre_start) / CLOCKS_PER_SEC * 1000);
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
    float angle[100] = {16, 20.1, 50.6, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 8;
    int snr = 40;
    int qr_iter = 10;
    float result[3] = {0};
    // int angle = 50;
    int iter = 1;

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

    //=================== MVDR Algorithm ==================================
    //timeMVDR_start = clock();
    //MVDR_DOA_2A_CPU_test(M, qr_iter, &angle[0], number_angle, result, snr);
    //timeMVDR_end = clock();
    //printf("--------------------------------------\n");
    //printf(L_GREEN "Total MVDR time : \t%.3f(ms)\n" CLOSE, (timeMVDR_end - timeMVDR_start) / CLOCKS_PER_SEC * 1000);
    //printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time / 1000);
    //total_multiply_time = 0;      // set to 0
    //total_pre_transpose_time = 0; // set to 0
    //=====================================================================
    //printf("Size of int: %zu bytes\n", sizeof(int));
}