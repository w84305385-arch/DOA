// AVX512
// g++ -mavx512f -g -o c_avx_thread c_avx_thread.c -Wall -Wextra -std=c++14 math_func.a -lpthread
// ./c_avx_thread -t4
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
#include <sys/syscall.h> // gettid()

// C
#include <unistd.h> // getopt()
#include <pthread.h>
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
static float sum_thread_syscall = 0.0;
static float total_multiply_time[8] = {0};
static float total_pre_transpose_time[8] = {0};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int flag_ind = 0;
int basic_val = 0;
//---------------------
extern char *optarg;
extern int optind;
extern int opterr;
extern int optopt;
//---------------------
typedef struct
{
    int M;
    int qr_iter;
    float *angle;
    int number_angle;
    float *result;
    int SNR;
    int index;
} MUSIC_VAR, MVDR_VAR;
//----------------------------------------------------------------

// pthread id function
pid_t gettid()
{
    return syscall(SYS_gettid);
}

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
            // // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
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
            // printf("\t%.2f ", matA_re[i * colA + j]);
            // printf("+ %.2fi", matA_im[i * colA + j]);
        }
        // printf("\n");
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

//__attribute__((aligned(32))) float matA_re[100000] = {0.0};     // re_A
//__attribute__((aligned(32))) float matA_im[100000] = {0.0};     // im_A
//-------------------------------------------------------------------
//__attribute__((aligned(32))) float matB_re[100000] = {0.0}; // re_B
//__attribute__((aligned(32))) float matB_im[100000] = {0.0}; // im_B
//-------------------------------------------------------------------
__attribute__((aligned(32))) float matC_Re_sum[4][30000000] = {0.0}; // Real
__attribute__((aligned(32))) float matC_Im_sum[4][30000000] = {0.0}; // Imaginary
__attribute__((aligned(32))) float matC_Real[4][30000000] = {0.0};   // re_C
__attribute__((aligned(32))) float matC_Imag[4][30000000] = {0.0};   // im_C

void complex_matrix_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timespec start, end;
    struct timeval time_data_start, time_data_end, time_data_diff; // time initial
    struct timeval sys_start, sys_end, sys_diff;                   // time initial
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    struct timeval start_transpose, end_transpose, diff_transpose; // transpose variable

    int temp_index = gettid() - basic_val;
    // printf("temp_index = %d\n", temp_index);
    gettimeofday(&start_multiply, NULL); // start

    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);                                     // start
    matrix_transpose(matB_re, matB_im, rowB, colB);                           // Matrix transpose
    gettimeofday(&end_transpose, NULL);                                       // end
    timersub(&end_transpose, &start_transpose, &diff_transpose);              // calculate total transpose time
    total_pre_transpose_time[gettid() - basic_val] += diff_transpose.tv_usec; // global variable can store transpose time

    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    memset(matC_Re_sum[temp_index], 0, rowA * colB * sizeof(float));
    memset(matC_Im_sum[temp_index], 0, rowA * colB * sizeof(float));
    // memset(matA_re, 0, rowA * colB * sizeof(float));
    // memset(matA_im, 0, rowA * colB * sizeof(float));
    // memset(matB_re, 0, rowA * colB * sizeof(float));
    // memset(matB_im, 0, rowA * colB * sizeof(float));
    memset(matC_Real[temp_index], 0, rowA * colB * sizeof(float));
    memset(matC_Imag[temp_index], 0, rowA * colB * sizeof(float));
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
    // // printf("matA_re:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     // printf("%.2f, ", matA_re[a]);
    // }
    // // printf("\nmatA_im:\n");
    // for (int a = 0; a < rowA * colA; a++)
    // {
    //     // printf("%.2f, ", matA_im[a]);
    // }
    // // printf("\n");
    // // printf("matB_re:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     // printf("%.2f, ", matB_re[a]);
    // }
    // // printf("\nmatB_im:\n");
    // for (int a = 0; a < rowB * colB; a++)
    // {
    //     // printf("%.2f, ", matB_im[a]);
    // }
    // // printf("\n");
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

                _mm512_storeu_ps(&matC_Real[temp_index][(i * colB * rowB + j * rowB) + AVX * k], re_C); // store Re value
                _mm512_storeu_ps(&matC_Imag[temp_index][(i * colB * rowB + j * rowB) + AVX * k], im_C); // store Im value
            }
        }
    }
    // gettimeofday(&end_multiply, NULL);
    // timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate
    // //printf(L_PURPLE "\nElapsed AVX512 multiply time: %ld(us)\n" CLOSE, (long int)diff_multiply.tv_usec);

    //-------------------------------------------------------------------

    for (int i = 0; i < rowA * colB; i++)
    {
        // // printf("※ matC_Real[%d] = %.2f, matC_Imag[%d] = %.2f\n", i * rowB, matC_Real[i * rowB], i * rowB, matC_Imag[i * rowB]);
        for (int a = 0; a < rowB; a++)
        {
            matC_Re_sum[temp_index][i] += matC_Real[temp_index][i * rowB + a];
            matC_Im_sum[temp_index][i] += matC_Imag[temp_index][i * rowB + a];
            // // printf("matC_Re_sum[%d] = %.2f, ", i, matC_Re_sum[i]);
            //    // printf("matC_Im_sum[%d] = %.0f, ", i * colB + a, matC_Im_sum[i]);
        }
        // // printf(" \n");
    }
    // // printf("\n");

    for (int i = 0; i < rowA * colB; i++)
    {
        // matC[i] = {matC_Real[i], matC_Imag[i]};
        //  matC[i] = matC_Imag[i];
        matC_re[i] = matC_Re_sum[temp_index][i];
        matC_im[i] = matC_Im_sum[temp_index][i];
    }
    //-------------------------------------------------------------------
    gettimeofday(&start_transpose, NULL);           // start
    matrix_transpose(matB_re, matB_im, rowB, colB); // Matrix transpose -> back to origin version
    gettimeofday(&end_transpose, NULL);             // end
    //-------------------------------------------------------------------
    timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    //***************
    gettimeofday(&sys_start, NULL);
    total_pre_transpose_time[gettid() - basic_val] += diff_transpose.tv_usec; // global variable can store transpose time
    //------------------------------------------------------------
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply);           // calculate total multiply time
    total_multiply_time[gettid() - basic_val] += diff_multiply.tv_usec; // global variable can store multiply time
    gettimeofday(&sys_end, NULL);
    timersub(&sys_end, &sys_start, &sys_diff);
    sum_thread_syscall += sys_diff.tv_usec;
    //***************
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
            // // printf("(%.0f + %.0fi), ", matA[j * rowA + i].real(), matA[j * rowA + i].imag());
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
    // // printf("debug vet_noise_re\n");
    // for (int i = 0; i < M * (M - len_t_theta); i++)
    // {
    //     // printf("\t(%f,%f)\n", vet_noise_re[i], vet_noise_im[i]);
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
    // // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    complex_matrix_multiplication(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector_re, a_vector_im, M, 1);
    complex_matrix_multiplication(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, S_MUSIC_temp_re, S_MUSIC_temp_im, 1, M, 1);
    cpp_division2(1, 0, &S_MUSIC_temp_re[0], &S_MUSIC_temp_im[0], music_Real, music_Imag);

    // // printf("music = (%f,%f)\n", *music_Real, *music_Real);
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
        // // printf(L_BLUE "re = %.2f, im = %.2f, power_cur[0] = %.2f\n" CLOSE, power_cur_re[0], power_cur_im[0], power_cur_re[0]);
        cpp_sqrt(&power_cur_re[0], &power_cur_im[0]);
        // // printf(YELLOW "power_cur[0] = %.2f\n" CLOSE, power_cur_re[0]);
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
            // // printf("Q_col_proj_re = ");
            // for (int i = 0; i < 20; i++)
            // {
            //     // printf(PURPLE "%.0f, " CLOSE, Q_col_proj_re[i]);
            // }
            // // printf("\n");
            //---------------------------------------------------------------
            // // printf(RED "Q_col_proj_re = " CLOSE);
            for (int j = 0; j < i; ++j)
            {
                for (int m = 0; m < row; ++m)
                {
                    Q_col_proj_re[m * i + j] = Q_re[m * col + j];
                    Q_col_proj_im[m * i + j] = Q_im[m * col + j];
                    // // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }
            // // printf("\n");
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
                // // printf(YELLOW "cpp_division(%.2f,%.2f) /= power_val(%.2f,%.2f)", Q_re[m * col + i], Q_im[m * col + i], power_val_re[0], power_val_im[0]);
                cpp_division(&Q_re[m * col + i], &Q_im[m * col + i], &power_val_re[0], &power_val_im[0]);
                // // printf(" = Q[%d](%.2f,%.2f)\n" CLOSE, m * col + i, Q_re[m * col + i], Q_im[m * col + i]);
            }
            // // printf("\n");
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
                // // printf(PURPLE "eigenvalue[%d] = %.2f\n" CLOSE, i * col + j, eigenvalue[i * col + j]);
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
            // // printf(L_BLUE "eigenvector[%d] = %.2f\n" CLOSE, m * col + i, eigenvector_re[m * col + i]);
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
        // // printf(YELLOW "\n----------------Before QR-------------------\n" CLOSE);
        // // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        // // printf("Q = \t\n");
        // print_complex_matrix(Q_re, Q_im, row, col);
        // // printf("R = \t\n");
        // print_complex_matrix(R_re, R_im, row, col);
        //-------------------------------After QR--------------------- --------------------

        qr(A_re, A_im, Q_re, Q_im, R_re, R_im, row, col);

        // // printf(YELLOW "\n----------------After QR-------------------\n" CLOSE);
        // // printf("A = \t\n");
        // print_complex_matrix(A_re, A_im, row, col);
        // // printf("Q = \t\n");
        // print_complex_matrix(Q_re, Q_im, row, col);
        // // printf("R = \t\n");
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

// compute the MUSIC DOA in one dimension on CPU
void MUSIC_DOA_1D_CPU(void)
{
}

void *MUSIC_DOA_2A_CPU_test(void *struct_var)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    pthread_mutex_lock(&mutex);
    MUSIC_VAR *music_param = (MUSIC_VAR *)struct_var;
    int thread_id = 0;
    //-------------------------------------------------------------------
    // Thread index
    if (flag_ind == 0)
    {
        basic_val = gettid();
        flag_ind++;
    }
    thread_id = gettid() - basic_val;
    printf("Thread ID = %ld\n", gettid() - basic_val);
   
    //-------------------------------------------------------------------
    // assign struct's parameter to stack variable

    int M = music_param->M;
    int qr_iter = music_param->qr_iter;
    float *angle = music_param->angle;
    int number_angle = music_param->number_angle;
    float *result = music_param->result;
    int SNR = music_param->SNR;
    int index = music_param->index; // thread index number

    //-------------------------------------------------------------------
    // printf("---------------\n");
    // printf("--MUSIC DOA--\n");
    // printf("---------------\n");
    // printf("--Parameter--\n");
    // printf("Antenna count:\t\t%d\n", M);
    // printf("SNR:\t\t\t%d\n", SNR);
    // printf("QR iteration:\t\t%d\n", qr_iter);

    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.5;
    float kc = 2 * PI / lemda;
    const int nd = 500;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    // printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        // printf("%.0f, ", angle[a]);
    }
    // printf("\n");
    pthread_mutex_unlock(&mutex);
    //---------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
            // // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
            // // printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
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
    //     // printf("\t(%f,%f)\n", x_r_re[a], x_r_im[a]);
    // }
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
        // // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }

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
    // // printf("----------Ve------------\n");
    // print_complex_matrix(Ve_re, Ve_im, M, M);
    // // printf("----------De------------\n");
    // print_complex_matrix(De_re, De_im, M, M);
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
            // // printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
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
    // // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, M, M);

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
    pthread_mutex_lock(&mutex);
    FILE *fp_excel = NULL;
    fp_excel = fopen("data/2D_MUSIC_dB.csv", "w");
    for (int i = 0; i < len_dth; ++i)
    {
        // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            // // printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector_re, a_vector_im, Pn_re, Pn_im, M, &S_MUSIC_re[i], &S_MUSIC_im[i]);
        // // printf("\tS_MUSIC(%f,%f), ", S_MUSIC_re[i], S_MUSIC_im[i]);
        S_MUSIC_dB[i] = cpp_20log_abs(&S_MUSIC_re[i], &S_MUSIC_im[i]);
        // // printf("S_MUSIC_dB = %.4f\n", S_MUSIC_dB[i]);

        fprintf(fp_excel, "%.1f,%.4f\n", (-60 + 0.1 * i), S_MUSIC_dB[i]);
    }
    fclose(fp_excel);
    pthread_mutex_unlock(&mutex);
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
    //-------------------------------------------------------------------
    // printf("sum_thread_syscall : \t%.3f(ms)\n", sum_thread_syscall / 1000);
}

void *MVDR_DOA_2A_CPU_test(void *struct_var)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
                                                                      //-------------------------------------------------------------------

    pthread_mutex_lock(&mutex);
    MVDR_VAR *mvdr_param = (MVDR_VAR *)struct_var;
    int thread_id = 0;
    //-------------------------------------------------------------------
    // Thread index
    if (flag_ind == 0)
    {
        basic_val = gettid();
        flag_ind++;
    }
    thread_id = gettid() - basic_val;
    printf("Thread ID = %ld\n", gettid() - basic_val);
    pthread_mutex_unlock(&mutex);

    int M = mvdr_param->M;
    int qr_iter = mvdr_param->qr_iter;
    float *angle = mvdr_param->angle;
    int number_angle = mvdr_param->number_angle;
    float *result = mvdr_param->result;
    int SNR = mvdr_param->SNR;
    int index = mvdr_param->index; // thread index number
    //-------------------------------------------------------------------
    // printf("---------------\n");
    // printf("--MVDR DOA--\n");
    // printf("---------------\n");
    // printf("--Parameter--\n");
    // printf("Antenna count:\t\t%d\n", M);
    // printf("SNR:\t\t\t%d\n", SNR);
    // printf("QR iteration:\t\t%d\n", qr_iter);

    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.5;
    float kc = 2 * PI / lemda;
    const int nd = 500;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    // printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        // printf("%.0f, ", angle[a]);
    }
    // printf("\n");
    //---------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
            // // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
            // // printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
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
    //     // printf("\t(%f,%f)\n", x_r_re[a], x_r_im[a]);
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
        // // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }

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
    // // printf("----------Ve------------\n");
    // print_complex_matrix(Ve_re, Ve_im, M, M);
    // // printf("----------De------------\n");
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
    //         // // printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
    //     }
    // }
    // //---------------------------------------------------------------
    float *R_xx_inv_1_re = (float *)malloc(M * M * sizeof(float));
    float *R_xx_inv_1_im = (float *)malloc(M * M * sizeof(float));
    float *Pn_re = (float *)calloc(M * M, sizeof(float));
    float *Pn_im = (float *)calloc(M * M, sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    // compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
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
    // // printf("----------R_xx------------\n");
    // print_complex_matrix(R_xx,M,M);
    // // printf("----------R_xx_inv_1------------\n");
    // print_complex_matrix(R_xx_inv_1,M,M);
    //---------------------------------------------------------------
    // // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, M, M);

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
    float *S_MVDR_re = (float *)malloc(len_dth * sizeof(float));
    float *S_MVDR_im = (float *)malloc(len_dth * sizeof(float));
    float *S_MVDR_dB = (float *)malloc(len_dth * sizeof(float));
    //---------------------------------------------------------------
    FILE *fp_excel = NULL;
    fp_excel = fopen("data/2D_MVDR_dB.csv", "w");
    for (int i = 0; i < len_dth; ++i)
    {
        // can be paralleled to compute S_MVDR_dB
        for (int j = 0; j < M; ++j)
        {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            // // printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        }
        compute_S_MUSIC(a_vector_re, a_vector_im, Pn_re, Pn_im, M, &S_MVDR_re[i], &S_MVDR_im[i]);
        // // printf("\tS_MVDR(%f,%f), ", S_MVDR_re[i], S_MVDR_im[i]);
        S_MVDR_dB[i] = cpp_20log_abs(&S_MVDR_re[i], &S_MVDR_im[i]);
        // // printf("S_MVDR_dB = %.4f\n", S_MVDR_dB[i]);

        fprintf(fp_excel, "%.1f,%.4f\n", (-60 + 0.1 * i), S_MVDR_dB[i]);
    }
    fclose(fp_excel);
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
}

void MUSIC_DOA_1D_CPU_test(int M, int qr_iter, int angle, float *result, int SNR)
{
    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    //-------------------------------------------------------------------
    // printf("---------------\n");
    // printf("--MUSIC DOA--\n");
    // printf("---------------\n");
    // printf("--Parameter--\n");
    // printf("Antenna count:\t\t%d\n", M);
    // printf("SNR:\t\t\t%d\n", SNR);
    // printf("QR iteration:\t\t%d\n", qr_iter);
    // printf(RED "Input angle :\t\t%d (degree)\n" CLOSE, angle);

    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const float lemda = (float)c / (float)fc;
    float d = lemda * 0.5;
    float kc = 2 * PI / lemda;
    const int nd = 500;
    const int len_t_theta = 1;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    t_theta[0] = angle;
    //---------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta_re[i * len_t_theta + j], &A_theta_im[i * len_t_theta + j], t_theta, d, kc, i, j);
            // // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
            // // printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
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
    //     // printf("\t(%f,%f)\n", x_r_re[a], x_r_im[a]);
    // }
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
        // // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }

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
    // // printf("----------Ve------------\n");
    // print_complex_matrix(Ve_re, Ve_im, M, M);
    // // printf("----------De------------\n");
    // print_complex_matrix(De_re, De_im, M, M);
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
            // // printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
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
    // // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, M, M);

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
    // printf("Total AWGN time: \t%.3f(ms)\n", time_AWGN / 1000);
    // printf("Total Eigen time: \t%.3f(ms)\n", time_Eigen / 1000);
    // printf("Total Pn time: \t\t%.3f(ms)\n", time_Pn / 1000);
    //-------------------------------------------------------------------
    // printf("position : \t\t%d\n", position);
    // printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    // printf("Max_theta :\t\t%f(dB)\n", max_temp);
}

int main(int argc, char **argv)
{
    //-------------------------------------------------------------------
    int opt;
    int thread_num;
    pthread_t th[8];

    while ((opt = getopt(argc, argv, "t:")) != -1)
    {
        // printf("handling optarg %c\n", opt);
        switch (opt)
        {
        case 't':

            thread_num = atoi(optarg);
            // printf("thread_num = %d\n", thread_num);
            break;

        default:
            break;
        }
    }
    // Parameter initialize
    float time_MUSIC = 0.0;
    float time_MVDR = 0.0;
    // struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // MUSIC time initial
    // struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;    // MVDR time initial

    // Total MUSIC Algorithm time
    float timeMusic_start[8] = {0.0};
    float timeMusic_end[8] = {0.0};

    // Total MVDR Algorithm time
    float timeMVDR_start[8] = {0.0};
    float timeMVDR_end[8] = {0.0};
    //-------------------------------------------------------------------
    float angle[100] = {-50, -47, -28, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 64;
    int snr = 40;
    int qr_iter = 10;
    float result[8] = {0};
    // int angle = 50;
    int iter = 1;

    //-------------------------------------------------------------------
    MUSIC_VAR *music_param = (MUSIC_VAR *)malloc(sizeof(MUSIC_VAR));
    // assign struct's parameter to stack variable
    music_param->M = M;
    music_param->qr_iter = qr_iter;
    music_param->angle = angle;
    music_param->number_angle = number_angle;
    music_param->result = result;
    music_param->SNR = snr;
    //-------------------------------------------------------------------
    //=================== MUSIC Algorithm =================================
    // printf("MUSIC Algorithm\n");
    for (int i = 0; i < thread_num; i++)
    {
        timeMusic_start[i] = clock();
        // printf("pthread_create[%d]\n", i);
        if (pthread_create(&th[i], NULL, &MUSIC_DOA_2A_CPU_test, music_param) != 0)
        {
            perror("Failed to create thread\n");
            return 1;
        }
        // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
        // MUSIC_DOA_2A_CPU_test(M, qr_iter, &angle[0], number_angle, result, snr);
    }

    for (int i = 0; i < thread_num; i++)
    {
        if (pthread_join(th[i], NULL) != 0)
        {
            perror("Failed join thread");
            return 1;
        }
        timeMusic_end[i] = clock();

        pthread_mutex_lock(&mutex);
        printf("--------------------------------------\n");
        printf(L_GREEN "Total MUSIC time : \t%.3f(ms)\n" CLOSE, (timeMusic_end[i] - timeMusic_start[i]) / CLOCKS_PER_SEC * 1000);
        printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time[i] / 1000);
        printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time[i] / 1000);
        total_multiply_time[i] = 0;      // set to 0
        total_pre_transpose_time[i] = 0; // set to 0
        pthread_mutex_unlock(&mutex);
    }
    basic_val = 0;
    flag_ind = 0;
    //=====================================================================
    
    //=================== MVDR Algorithm ==================================
    //-------------------------------------------------------------------
    MVDR_VAR *mvdr_param = (MVDR_VAR *)malloc(sizeof(MVDR_VAR));
    // assign struct's parameter to stack variable
    mvdr_param->M = M;
    mvdr_param->qr_iter = qr_iter;
    mvdr_param->angle = angle;
    mvdr_param->number_angle = number_angle;
    mvdr_param->result = result;
    mvdr_param->SNR = snr;
    //-------------------------------------------------------------------
    for (int i = 0; i < thread_num; i++)
    {
        timeMVDR_start[i] = clock();
        // printf("pthread_create[%d]\n", i);
        if (pthread_create(&th[i], NULL, &MVDR_DOA_2A_CPU_test, mvdr_param) != 0)
        {
            perror("Failed to create thread\n");
            return 1;
        }
    }
    // MVDR_DOA_2A_CPU_test(M, qr_iter, &angle[0], number_angle, result, snr);
    for (int i = 0; i < thread_num; i++)
    {
        if (pthread_join(th[i], NULL) != 0)
        {
            perror("Failed join thread");
            return 1;
        }
        timeMVDR_end[i] = clock();

        pthread_mutex_lock(&mutex);
        printf("--------------------------------------\n");
        printf(L_GREEN "Total MVDR time : \t%.3f(ms)\n" CLOSE, (timeMVDR_end[i] - timeMVDR_start[i]) / CLOCKS_PER_SEC * 1000);
        printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time[i] / 1000);
        printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time[i] / 1000);
        total_multiply_time[i] = 0;      // set to 0
        total_pre_transpose_time[i] = 0; // set to 0
        pthread_mutex_unlock(&mutex);
    }
    
}