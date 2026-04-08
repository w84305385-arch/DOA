//先4根再16根 multi-thread
// g++ -mavx512f -mavx512bw -g -o m_barker_method2_ML m_barker_method2_ML.c -Wall -Wextra -std=c++14 math_func.a -lpthread
// ./m_barker_method2_ML -t4
// #define DATA_CSV_MODE 1
#define PI acos(-1)
#define BLOCK_SIZE 16
#define PRINT_RESULT 1
#define PLOT_RESULT 0
//--------------------
#define AVX 16
#define M_Antenna 64
#define ND 512
#define CODE_LENGTH 127
//--------------------
#include <immintrin.h>
#include "math_func.h"
#include <sys/syscall.h> // gettid()
// C
#include <pthread.h>
#include <complex.h>
#include <assert.h>
#include "color.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>// getopt()
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
//----------------------------------------------------------------
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
    int *check_result;
} ML_VAR;
//----------------------------------------------------------------
// pthread id function
pid_t gettid()
{
    return syscall(SYS_gettid);
}
//-----
__attribute__((aligned(32))) float matC_Re_sum[4][30000000] = {0.0}; // Real
__attribute__((aligned(32))) float matC_Im_sum[4][30000000] = {0.0}; // Imaginary
__attribute__((aligned(32))) float matC_Real[4][30000000] = {0.0};   // re_C
__attribute__((aligned(32))) float matC_Imag[4][30000000] = {0.0};   // im_C
//-----
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
//-----

void matrix_transpose(float *matA_re, float *matA_im, int rowA, int colA)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matC_Real[temp_index][i * colA + j] = matA_re[i * colA + j];
            matC_Imag[temp_index][i * colA + j] = matA_im[i * colA + j];
        }
    }
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
        matA_re[j * rowA + i] = matC_Real[temp_index][i * colA + j];
        matA_im[j * rowA + i] = matC_Imag[temp_index][i * colA + j];
        }
    }
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


/*
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matC_Real[temp_index][i * colA + j] = matA_re[i * colA + j]+matB_re[i * colA + j];
            matC_Imag[temp_index][i * colA + j] = matA_im[i * colA + j]+matB_im[i * colA + j];
        }
    }
    for (int i = 0; i < rowA * colA; i++)
    {
        matA_re[i] = matC_Real[temp_index][i];
        matA_im[i] = matC_Imag[temp_index][i];
    }
}
*/

/*
void complex_matrix_subtraction(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matC_Real[temp_index][i * colA + j] = matA_re[i * colA + j]-matB_re[i * colA + j];
            matC_Imag[temp_index][i * colA + j] = matA_im[i * colA + j]-matB_im[i * colA + j];
        }
    }
    for (int i = 0; i < rowA * colA; i++)
    {
        matA_re[i] = matC_Real[temp_index][i];
        matA_im[i] = matC_Imag[temp_index][i];
    }
}
*/ 

//AVX加減
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{   
    struct timespec start, end;
    struct timeval start_plus_o, end_plus_o, diff_plus_o;  
    int temp_index = gettid() - basic_val;
    //float*temp_re= (float *)aligned_alloc(sizeof(__m512), sizeof(float) * rowA * colA);
    //float*temp_im= (float *)aligned_alloc(sizeof(__m512), sizeof(float) * rowA * colA);
    //memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
   // memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    memset(matC_Re_sum[temp_index], 0, rowA * colA * sizeof(float));
    memset(matC_Im_sum[temp_index], 0, rowA * colA * sizeof(float));
    // Loop over blocks of 8 elements (AVX-512 can process 8 single-precision floats at once)
    
    gettimeofday(&start_plus_o, NULL);
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
        _mm512_storeu_ps(&matC_Re_sum[temp_index][i],c_re);
        _mm512_storeu_ps(&matC_Im_sum[temp_index][i],c_im); 
    }
    for (int i = 0; i < rowA * colA; i++)
    {
        // matC[i] = {matC_Real[i], matC_Imag[i]};
        //  matC[i] = matC_Imag[i];
        matA_re[i] = matC_Re_sum[temp_index][i];
        matA_im[i] = matC_Im_sum[temp_index][i];
    }
}
void complex_matrix_subtraction(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{   
    struct timespec start, end;
    struct timeval start_plus_o, end_plus_o, diff_plus_o;  
    int temp_index = gettid() - basic_val;
    //float*temp_re= (float *)aligned_alloc(sizeof(__m512), sizeof(float) * rowA * colA);
    //float*temp_im= (float *)aligned_alloc(sizeof(__m512), sizeof(float) * rowA * colA);
    //memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
   // memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    memset(matC_Re_sum[temp_index], 0, rowA * colA * sizeof(float));
    memset(matC_Im_sum[temp_index], 0, rowA * colA * sizeof(float));
    // Loop over blocks of 8 elements (AVX-512 can process 8 single-precision floats at once)
    
    gettimeofday(&start_plus_o, NULL);
    __m512 a_re,a_im,b_re,b_im,c_re,c_im;
    for (int i = 0; i <rowA * colA ; i += 16)
    {
        a_re = _mm512_loadu_ps(&matA_re[i]);
        a_im = _mm512_loadu_ps(&matA_im[i]);
        b_re = _mm512_loadu_ps(&matB_re[i]);
        b_im = _mm512_loadu_ps(&matB_im[i]);

        // Perform vectorized addition for real and imaginary parts
        c_re = _mm512_sub_ps(a_re, b_re);
        c_im = _mm512_sub_ps(a_im, b_im);

        // Store the results back to memory
        _mm512_storeu_ps(&matC_Re_sum[temp_index][i],c_re);
        _mm512_storeu_ps(&matC_Im_sum[temp_index][i],c_im); 
    }
    for (int i = 0; i < rowA * colA; i++)
    {
        // matC[i] = {matC_Real[i], matC_Imag[i]};
        //  matC[i] = matC_Imag[i];
        matA_re[i] = matC_Re_sum[temp_index][i];
        matA_im[i] = matC_Im_sum[temp_index][i];
    }
}

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
    //gettimeofday(&start_multiply, NULL); // start

    //-------------------------------------------------------------------
    //gettimeofday(&start_transpose, NULL);                                     // start
    matrix_transpose(matB_re, matB_im, rowB, colB);                           // Matrix transpose
    //gettimeofday(&end_transpose, NULL);                                       // end
    //timersub(&end_transpose, &start_transpose, &diff_transpose);              // calculate total transpose time
    //total_pre_transpose_time[gettid() - basic_val] += diff_transpose.tv_usec; // global variable can store transpose time

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
    //gettimeofday(&start_transpose, NULL);           // start
    matrix_transpose(matB_re, matB_im, rowB, colB); // Matrix transpose -> back to origin version
    //gettimeofday(&end_transpose, NULL);             // end
    //-------------------------------------------------------------------
    //timersub(&end_transpose, &start_transpose, &diff_transpose); // calculate total transpose time
    //***************
    //gettimeofday(&sys_start, NULL);
    //total_pre_transpose_time[gettid() - basic_val] += diff_transpose.tv_usec; // global variable can store transpose time
    //------------------------------------------------------------
    //gettimeofday(&end_multiply, NULL);
    //timersub(&end_multiply, &start_multiply, &diff_multiply);           // calculate total multiply time
    //total_multiply_time[gettid() - basic_val] += diff_multiply.tv_usec; // global variable can store multiply time
    //gettimeofday(&sys_end, NULL);
    //timersub(&sys_end, &sys_start, &sys_diff);
    //sum_thread_syscall += sys_diff.tv_usec;
    //***************
}

// get complex matrix by column
void complex_matrix_get_columns(float *matA_re, float *matA_im, float *matCol_re, float *matCol_im, int rowA, int colA, int colTarget)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < rowA; ++i)
    {
        matC_Real[temp_index][i] = matA_re[i * colA + colTarget];
        matC_Imag[temp_index][i] = matA_im[i * colA + colTarget];
    }
    for (int i = 0; i < rowA; ++i)
    {
        matCol_re[i] = matC_Real[temp_index][i];
        matCol_im[i] = matC_Imag[temp_index][i];
    }
}


// get complex matrix by row
void complex_matrix_get_rows(float *matA_re, float *matA_im, float *matRow_re, float *matRow_im, int rowA, int colA, int rowTarget)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < colA; ++i)
    {
        matC_Real[temp_index][i] = matA_re[rowTarget * colA + i];
        matC_Imag[temp_index][i] = matA_im[rowTarget * colA + i];
    }
    for (int i = 0; i < colA; ++i)
    {
        matRow_re[i] = matC_Real[temp_index][i];
        matRow_im[i] = matC_Imag[temp_index][i];
    }
}


void complex_matrix_conjugate_transpose(float *matA_re, float *matA_im, int rowA, int colA)
{
    int temp_index = gettid() - basic_val;
    memset(matC_Real[temp_index], 0, rowA * colA * sizeof(int));
    memset(matC_Imag[temp_index], 0, rowA * colA * sizeof(int));
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matC_Real[temp_index][i * colA + j] = matA_re[i * colA + j];
            matC_Imag[temp_index][i * colA + j] = -matA_im[i * colA + j];
        }
    }
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
        matA_re[j * rowA + i] = matC_Real[temp_index][i * colA + j];
        matA_im[j * rowA + i] = matC_Imag[temp_index][i * colA + j];
        }
    }
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

void *ML_DOA_1D_CPU_test(void *struct_var) 
{
    float time_MLre = 0.0;
    float timeMLre_start, timeMLre_end; 
    float timeML_test_start[4] = {0.0};
    float timeML_test_end[4] = {0.0};    

    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_ML_start, time_ML_end, time_ML_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_Eigen_start_8, time_Eigen_end_8, time_Eigen_diff_8; // time initial
    struct timeval time_vet_start, time_vet_end, time_vet_diff;    // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    struct timeval time_Pn_start_8, time_Pn_end_8, time_Pn_diff_8;          // time initial
    struct timeval time_search_start, time_search_end, time_search_diff;          // time initial
    struct timeval time_search_start_8, time_search_end_8, time_search_diff_8;          // time initial
    struct timeval time_small_search_start, time_small_search_end, time_small_search_diff;          // time initial
    //-------------------------------------------------------------------
    pthread_mutex_lock(&mutex);
    ML_VAR *ml_param = (ML_VAR *)struct_var;
    int thread_id = 0;
    //-------------------------------------------------------------------
    // Thread index
    if (flag_ind == 0)
    {
        basic_val = gettid();
        flag_ind++;
    }
    thread_id = gettid() - basic_val;
    //printf("gettid() = %ld\n", gettid());
    //printf("basic_val = %ld\n", basic_val);
    printf("Thread ID = %ld\n", gettid() - basic_val);
    //-------------------------------------------------------------------
    int M = ml_param->M;
    int qr_iter = ml_param->qr_iter;
    float *angle = ml_param->angle;
    int number_angle = ml_param->number_angle;
    float *result = ml_param->result;
    int SNR = ml_param->SNR;
    int index = ml_param->index; // thread index number
    int *check_result = ml_param->check_result;
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
    float d;
    if(*check_result == 1){
        d = lemda * 0.25;
    }
    else{
        d = 0;
    }
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
    pthread_mutex_unlock(&mutex);
    // A_theta matrix (M, length of t_theta)
    //-----------------------------------------------------------------------------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
     //---------------------------------------------------------------
    float *A_theta4_re = (float *)malloc(4 * len_t_theta * sizeof(float));
    float *A_theta4_im = (float *)malloc(4 * len_t_theta * sizeof(float));
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
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta4_re[i * len_t_theta + j], &A_theta4_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
           // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
    float *sig_co4_re = (float *)malloc(4 * nd * sizeof(float));
    float *sig_co4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    float *x_r4_re = (float *)malloc(4 * nd * sizeof(float));
    float *x_r4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    // compute sig_co
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    //gettimeofday(&time_sig_end, NULL);
    // print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    //gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    complex_matrix_multiplication(A_theta4_re, A_theta4_im, t_sig_re, t_sig_im, sig_co4_re, sig_co4_im, 4, len_t_theta, nd);
    //print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    //gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co4_re, sig_co4_im, x_r4_re, x_r4_im, SNR, 4, nd);
    //gettimeofday(&time_AWGN_end, NULL);
    //---------------------------------------------------------------
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
    float *R_xx4_re = (float *)malloc(4 * 4 * sizeof(float));
    float *R_xx4_im = (float *)malloc(4 * 4 * sizeof(float));
    float M4_re = 4;
    float M4_im = 0.0;
    float *M_ptr4 = &M4_re;
    float *M_ptr4_im = &M4_im;
    //---------------------------------------------------------------
    //gettimeofday(&time_Rxx_start, NULL);
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int i = 0; i < M * M; ++i)
    {
        // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r4_re, x_r4_im, R_xx4_re, R_xx4_im, 4, nd);
    for (int i = 0; i < 4 * 4; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx4_re[i], &R_xx4_im[i], M_ptr4, M_ptr4_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    //---------------------------------------------------------------
    //timeMLre_start = clock();
    //---------------------------------------------------------------
    // 取定點
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
    //-------------------------------------------------------//
    pthread_mutex_lock(&mutex);
    gettimeofday(&time_search_start, NULL);
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -5 + 1 * i; //-60 -30 0 30 60
        dr[i] = dth[i] * PI / 180;
    }
    //---------------------------------------------------------------
    float *a_vector_re = (float *)malloc(4 * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(4 * sizeof(float));
    float *a_temp_re = (float *)malloc(4 * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(4 * sizeof(float));
    float *S_ML_re = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_im = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_dB = (float *)malloc(len_dth * sizeof(float));
    float *theta_re = (float *)malloc(4 * 4 * sizeof(float));
    float *theta_im = (float *)malloc(4 * 4 * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA_im = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * 4 * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * 4 * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    //---------------------------------------------------------------
     for(int i = 0; i < len_dth; ++i) { 
        for(int j = 0; j < 4; ++j) {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_re,a_vector_re,(4 * 1 * sizeof(float)));
        memcpy(a_temp_im,a_vector_im,(4 * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, 4);   
        //A^H*A =1*1  
        complex_matrix_multiplication(a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, 4, 1);
        //[inv(A^H*A)] =1*1  
        //gettimeofday(&time_findmax_start, NULL);
        //gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA2_re, AH_mulA2_im, 1, 1, 4);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_re, a_vector_im, AH_mulA2_re, AH_mulA2_im,AH_mulA3_re, AH_mulA3_im, 4, 1, 4);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, R_xx4_re, R_xx4_im, theta_re, theta_im, 4, 4, 4);
        // trace[P_A*R] 
        trace(theta_re, theta_im, S_ML_re, S_ML_im, 4, 4, i);
    }
    free(a_temp_re);
    free(a_temp_im);
    for(int i = 0; i < len_dth; ++i) {
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    pthread_mutex_unlock(&mutex);
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
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dthA[i] = dth[position]-0.5 + 0.1 * i; //-60 -30 0 30 60
        drA[i] = dthA[i] * PI / 180;
    }
    
    pthread_mutex_lock(&mutex);

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
        //gettimeofday(&start_test, NULL);
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
    gettimeofday(&time_search_end, NULL);
    pthread_mutex_unlock(&mutex);
    
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
    
    //find Max and position
    //printf("\n");
    //timeMLre_end = clock();
    
    //-------------------------------------------------------------------
    // timersub function
    //-------------------------------------------------------------------
    float time_search;                           // create float parameter in order to convert (us) to (ms)
    timersub(&time_search_end, &time_search_start, &time_search_diff);
    time_search = time_search_diff.tv_usec;

    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    printf(RED "Theta estimationA :\t%.3f (degree)\n" CLOSE, dthA[positionA]);
    printf("Max_thetaA :\t\t%f(dB)\n", max_tempA);


    
    //printf("Total search time: \t\t%.3f(ms)\n", time_search / 1000);

    float total_time = time_search; 
    printf(BLUE"Total time: \t%.3f(ms)\n"CLOSE, total_time / 1000);


    //printf(L_GREEN "Total ML REAL time :\t%.3f(ms)\n" CLOSE, (timeMLre_end - timeMLre_start) / CLOCKS_PER_SEC * 1000);
        

}

void ML_thread_4(int argc, char **argv, int *check_result)
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
    float time_ML = 0.0;
    // Total ML Algorithm time
    float timeML_start[8] = {0.0};
    float timeML_end[8] = {0.0};
    //-------------------------------------------------------------------
    float angle[100] = {2, 20.1, 50.6, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 16;
    int snr = -10;
    int qr_iter = 1;
    float result[8] = {0};
    // int angle = 50;
    int iter = 1;
    //-------------------------------------------------------------------
    ML_VAR *ml_param = (ML_VAR *)malloc(sizeof(ML_VAR));
    // assign struct's parameter to stack variable
    ml_param->M = M;
    ml_param->qr_iter = qr_iter;
    ml_param->angle = angle;
    ml_param->number_angle = number_angle;
    ml_param->result = result;
    ml_param->SNR = snr;
    ml_param->check_result=check_result;
    //-------------------------------------------------------------------
    //=================== MUSIC Algorithm =================================
    // printf("MUSIC Algorithm\n");
    for (int i = 0; i < thread_num; i++)
    {
        //timeML_start[i] = clock();
        // printf("pthread_create[%d]\n", i);
        if (pthread_create(&th[i], NULL, &ML_DOA_1D_CPU_test, ml_param) != 0)
        {
            perror("Failed to create thread\n");
            //return 1;
        }
        // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
        // MUSIC_DOA_2A_CPU_test(M, qr_iter, &angle[0], number_angle, result, snr);
    }

    for (int i = 0; i < thread_num; i++)
    {
        if (pthread_join(th[i], NULL) != 0)
        {
            perror("Failed join thread");
            //return 1;
        }
        //timeML_end[i] = clock();

        //pthread_mutex_lock(&mutex);
        //printf("--------------------------------------\n");
        //printf(L_GREEN "Total ML time : \t%.3f(ms)\n" CLOSE, (timeML_end[i] - timeML_start[i]) / CLOCKS_PER_SEC * 1000);
        //printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time[i] / 1000);
        //printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time[i] / 1000);
        //total_multiply_time[i] = 0;      // set to 0
        //total_pre_transpose_time[i] = 0; // set to 0
        //pthread_mutex_unlock(&mutex);
        
    }    
    basic_val = 0;
    flag_ind = 0;
    


}

void *ML_DOA_1D_CPU_test1(void *struct_var) 
{
    float time_MLre = 0.0;
    float timeMLre_start, timeMLre_end; 
    float timeML_test_start[4] = {0.0};
    float timeML_test_end[4] = {0.0};    

    //-------------------------------------------------------------------
    // Parameter initialize
    struct timeval time_ML_start, time_ML_end, time_ML_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_Eigen_start_8, time_Eigen_end_8, time_Eigen_diff_8; // time initial
    struct timeval time_vet_start, time_vet_end, time_vet_diff;    // time initial
    struct timeval time_AWGN_start, time_AWGN_end, time_AWGN_diff;    // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial
    struct timeval time_Pn_start_8, time_Pn_end_8, time_Pn_diff_8;          // time initial
    struct timeval time_search_start, time_search_end, time_search_diff;          // time initial
    struct timeval time_search_start_8, time_search_end_8, time_search_diff_8;          // time initial
    struct timeval time_small_search_start, time_small_search_end, time_small_search_diff;          // time initial
    //-------------------------------------------------------------------
    pthread_mutex_lock(&mutex);
    ML_VAR *ml_param = (ML_VAR *)struct_var;
    int thread_id = 0;
    //-------------------------------------------------------------------
    // Thread index
    if (flag_ind == 0)
    {
        basic_val = gettid();
        flag_ind++;
    }
    thread_id = gettid() - basic_val;
    //printf("gettid() = %ld\n", gettid());
    //printf("basic_val = %ld\n", basic_val);
    printf("Thread ID = %ld\n", gettid() - basic_val);
    //-------------------------------------------------------------------
    int M = ml_param->M;
    int qr_iter = ml_param->qr_iter;
    float *angle = ml_param->angle;
    int number_angle = ml_param->number_angle;
    float *result = ml_param->result;
    int SNR = ml_param->SNR;
    int index = ml_param->index; // thread index number
    int *check_result = ml_param->check_result;
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
    float d;
    if(*check_result == 1){
        d = lemda * 0.25;
    }
    else{
        d = 0;
    }
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
    pthread_mutex_unlock(&mutex);
    // A_theta matrix (M, length of t_theta)
    //-----------------------------------------------------------------------------------------------------------------------------------
    float *A_theta_re = (float *)malloc(M * len_t_theta * sizeof(float));
    float *A_theta_im = (float *)malloc(M * len_t_theta * sizeof(float));
     //---------------------------------------------------------------
    float *A_theta4_re = (float *)malloc(4 * len_t_theta * sizeof(float));
    float *A_theta4_im = (float *)malloc(4 * len_t_theta * sizeof(float));
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
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            cpp_exp(&A_theta4_re[i * len_t_theta + j], &A_theta4_im[i * len_t_theta + j], &t_theta[j], d, kc, i, j);
           // printf("\t(%f,%f)\n", A_theta_re[i * len_t_theta + j], A_theta_im[i * len_t_theta + j]);
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
    float *sig_co4_re = (float *)malloc(4 * nd * sizeof(float));
    float *sig_co4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    float *x_r4_re = (float *)malloc(4 * nd * sizeof(float));
    float *x_r4_im = (float *)malloc(4 * nd * sizeof(float));
    //---------------------------------------------------------------
    //---------------------------------------------------------------
    // compute sig_co
    complex_matrix_multiplication(A_theta_re, A_theta_im, t_sig_re, t_sig_im, sig_co_re, sig_co_im, M, len_t_theta, nd);
    //gettimeofday(&time_sig_end, NULL);
    // print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    //gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co_re, sig_co_im, x_r_re, x_r_im, SNR, M, nd);
    complex_matrix_multiplication(A_theta4_re, A_theta4_im, t_sig_re, t_sig_im, sig_co4_re, sig_co4_im, 4, len_t_theta, nd);
    //print_complex_matrix(sig_co_re, sig_co_im, M, nd);
    //gettimeofday(&time_AWGN_start, NULL);
    cpp_awgn(sig_co4_re, sig_co4_im, x_r4_re, x_r4_im, SNR, 4, nd);
    //gettimeofday(&time_AWGN_end, NULL);
    //---------------------------------------------------------------
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
    float *R_xx4_re = (float *)malloc(4 * 4 * sizeof(float));
    float *R_xx4_im = (float *)malloc(4 * 4 * sizeof(float));
    float M4_re = 4;
    float M4_im = 0.0;
    float *M_ptr4 = &M4_re;
    float *M_ptr4_im = &M4_im;
    //---------------------------------------------------------------
    //gettimeofday(&time_Rxx_start, NULL);
    complex_matrix_conjugate_transpose_multiplication(x_r_re, x_r_im, R_xx_re, R_xx_im, M, nd);
    for (int i = 0; i < M * M; ++i)
    {
        // printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        // printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r4_re, x_r4_im, R_xx4_re, R_xx4_im, 4, nd);
    for (int i = 0; i < 4 * 4; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx4_re[i], &R_xx4_im[i], M_ptr4, M_ptr4_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    //---------------------------------------------------------------
    //timeMLre_start = clock();
    //---------------------------------------------------------------
    // 取定點
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
    //-------------------------------------------------------//
    pthread_mutex_lock(&mutex);
    gettimeofday(&time_search_start, NULL);
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -5 + 1 * i; //-60 -30 0 30 60
        dr[i] = dth[i] * PI / 180;
    }
    //---------------------------------------------------------------
    float *a_vector_re = (float *)malloc(4 * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(4 * sizeof(float));
    float *a_temp_re = (float *)malloc(4 * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(4 * sizeof(float));
    float *S_ML_re = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_im = (float *)malloc(len_dth * sizeof(float));
    float *S_ML_dB = (float *)malloc(len_dth * sizeof(float));
    float *theta_re = (float *)malloc(4 * 4 * sizeof(float));
    float *theta_im = (float *)malloc(4 * 4 * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA_im = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * 4 * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * 4 * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(4 * 4 * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    //---------------------------------------------------------------
     for(int i = 0; i < len_dth; ++i) { 
        for(int j = 0; j < 4; ++j) {
            cpp_exp2(&a_vector_re[j], &a_vector_im[j], dr, d, kc, i, j);
            //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
        } 
        memcpy(a_temp_re,a_vector_re,(4 * 1 * sizeof(float)));
        memcpy(a_temp_im,a_vector_im,(4 * 1 * sizeof(float)));
        //a_vector_conjugate_transpose A^H
        complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, 4);   
        //A^H*A =1*1  
        complex_matrix_multiplication(a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, 4, 1);
        //[inv(A^H*A)] =1*1  
        //gettimeofday(&time_findmax_start, NULL);
        //gettimeofday(&start_test, NULL);
        matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1, 1);
        // [inv(A^H*A)]*A^H = 1*M
        complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA2_re, AH_mulA2_im, 1, 1, 4);
        // P_A = A*[inv(A^H*A)]*A^H = M*M  
        complex_matrix_multiplication(a_vector_re, a_vector_im, AH_mulA2_re, AH_mulA2_im,AH_mulA3_re, AH_mulA3_im, 4, 1, 4);
        // P_A*R = M*M 
        complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, R_xx4_re, R_xx4_im, theta_re, theta_im, 4, 4, 4);
        // trace[P_A*R] 
        trace(theta_re, theta_im, S_ML_re, S_ML_im, 4, 4, i);
    }
    free(a_temp_re);
    free(a_temp_im);
    for(int i = 0; i < len_dth; ++i) {
        S_ML_dB[i] = cpp_20log_abs(&S_ML_re[i], &S_ML_im[i]);
        //printf(" output_dB=\t%.2f \n",S_ML_dB[i]);
    }
    pthread_mutex_unlock(&mutex);
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
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dthA[i] = dth[position]-0.5 + 0.1 * i; //-60 -30 0 30 60
        drA[i] = dthA[i] * PI / 180;
    }
    
    pthread_mutex_lock(&mutex);

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
        //gettimeofday(&start_test, NULL);
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
    gettimeofday(&time_search_end, NULL);
    pthread_mutex_unlock(&mutex);
    
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
    
    //find Max and position
    //printf("\n");
    //timeMLre_end = clock();
    
    //-------------------------------------------------------------------
    // timersub function
    //-------------------------------------------------------------------
    float time_search;                           // create float parameter in order to convert (us) to (ms)
    timersub(&time_search_end, &time_search_start, &time_search_diff);
    time_search = time_search_diff.tv_usec;

    printf("position : \t\t%d\n", position);
    printf(RED "Theta estimation :\t%.3f (degree)\n" CLOSE, dth[position]);
    printf("Max_theta :\t\t%f(dB)\n", max_temp);
    printf(RED "Theta estimationA :\t%.3f (degree)\n" CLOSE, dthA[positionA]);
    printf("Max_thetaA :\t\t%f(dB)\n", max_tempA);


    
    //printf("Total search time: \t\t%.3f(ms)\n", time_search / 1000);

    float total_time = time_search; 
    printf(BLUE"Total time: \t%.3f(ms)\n"CLOSE, total_time / 1000);


    //printf(L_GREEN "Total ML REAL time :\t%.3f(ms)\n" CLOSE, (timeMLre_end - timeMLre_start) / CLOCKS_PER_SEC * 1000);
        

}

void ML_thread_1(int argc, char **argv, int *check_result)
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
    float time_ML = 0.0;
    // Total ML Algorithm time
    float timeML_start[8] = {0.0};
    float timeML_end[8] = {0.0};
    //-------------------------------------------------------------------
    float angle[100] = {2, 20.1, 50.6, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 16;
    int snr = -10;
    int qr_iter = 1;
    float result[8] = {0};
    // int angle = 50;
    int iter = 1;
    //-------------------------------------------------------------------
    ML_VAR *ml_param = (ML_VAR *)malloc(sizeof(ML_VAR));
    // assign struct's parameter to stack variable
    ml_param->M = M;
    ml_param->qr_iter = qr_iter;
    ml_param->angle = angle;
    ml_param->number_angle = number_angle;
    ml_param->result = result;
    ml_param->SNR = snr;
    ml_param->check_result=check_result;
    //-------------------------------------------------------------------
    //=================== MUSIC Algorithm =================================
    // printf("MUSIC Algorithm\n");
    for (int i = 0; i < thread_num-3; i++)
    {
        //timeML_start[i] = clock();
        // printf("pthread_create[%d]\n", i);
        if (pthread_create(&th[i], NULL, &ML_DOA_1D_CPU_test1, ml_param) != 0)
        {
            perror("Failed to create thread\n");
            //return 1;
        }
        // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
        // MUSIC_DOA_2A_CPU_test(M, qr_iter, &angle[0], number_angle, result, snr);
    }

    for (int i = 0; i < thread_num-3; i++)
    {
        if (pthread_join(th[i], NULL) != 0)
        {
            perror("Failed join thread");
            //return 1;
        }
        //timeML_end[i] = clock();

        //pthread_mutex_lock(&mutex);
        //printf("--------------------------------------\n");
        //printf(L_GREEN "Total ML time : \t%.3f(ms)\n" CLOSE, (timeML_end[i] - timeML_start[i]) / CLOCKS_PER_SEC * 1000);
        //printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time[i] / 1000);
        //printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time[i] / 1000);
        //total_multiply_time[i] = 0;      // set to 0
        //total_pre_transpose_time[i] = 0; // set to 0
        //pthread_mutex_unlock(&mutex);
        
    }    
    basic_val = 0;
    flag_ind = 0;
    


}

int main(int argc, char **argv)
{
    int target_sequence[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };
    int targetCode[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };
    int check_result=1;
    int temp ;
    //print_code(targetCode,CODE_LENGTH);
    //print_code(target_sequence,CODE_LENGTH);
    checkpnCode(targetCode, target_sequence,&check_result);
    int search_point=4;
    //-------------------------------------------
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
        //MVDR_DOA_2A_CPU_test(M, qr_iter, hybrid_qr_iter, &angle[0],number_angle, result, snr, &check_result);
        
        //printf("\n");
    }
    for(int i=126-search_point;i<126;i++){
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
    }   
    ML_thread_4(argc, argv, &check_result);//前四次
    
    for(int i=126;i<127;i++){
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
        ML_thread_1(argc, argv, &check_result);
    }

    for(int i=127;i<127+search_point;i++){
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
    } 
    ML_thread_4(argc, argv, &check_result);//後四次
    //*/
}

// g++ -mavx512f -mavx512bw -g -o m_barker_method2_ML m_barker_method2_ML.c -Wall -Wextra -std=c++14 math_func.a -lpthread
// ./m_barker_method2_ML -t4
