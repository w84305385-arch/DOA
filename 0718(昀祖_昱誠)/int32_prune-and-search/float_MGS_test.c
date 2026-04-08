// 原始輸入放大64，因應power_value縮小128
// 每次輸入固定
// g++ -mavx512f -g -o float_MGS_test  float_MGS_test.c -Wall -Wextra -std=c++14 math_func.a
// ./float_MGS_test 
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
            printf("\t%.1f", matA_re[i * colA + j]);
            printf("+%.1fi", matA_im[i * colA + j]);
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

    //printf("music = (%f,%f) ", *music_Real, *music_Real);
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
    //int X2 = 128;  //為了使power_val不超過32768需要Q_col除以X2倍
    
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
        //printf("除完的Q_re,Q_im(%d):\n",i);
        //print_complex_matrix(Q_re, Q_im, row, col);
        int a3=1;
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
        
        //printf("除完的Q_col*a3,(a3:%d) \n",a3);
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        complex_matrix_conjugate_transpose( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        
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
            //printf("v:Q_col_proj(%d):\n",i);
            //print_complex_matrix(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            
            //* 讓Q_col_proj縮小別太小 total三塊程式要改(含此塊)[備註以免漏掉]
            int a=1;
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
            
            int a2=1;
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
            /*
            for(int w=0;w<row*size;w++){ ///16
                Q_sub_re[w]=Q_sub_re[w]/16;
                Q_sub_im[w]=Q_sub_im[w]/16;
            }
            */
            int a33=a3*a3;
            //*配合除法後把Q_col放大
            for(int w=0;w<row*size;w++){
                Q_sub_re[w]=Q_sub_re[w]/a33;
                Q_sub_im[w]=Q_sub_im[w]/a33;
            }
            //printf("Q_sub 除a33後\n");
            //print_complex_matrix(Q_sub_re, Q_sub_im, row, col-(i+1));
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
    float *QxR_re = (float *)malloc(row * col * sizeof(float));
    float *QxR_im = (float *)malloc(row * col * sizeof(float));
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
    for(int w=0;w<row*col;w++){
        A_re[w]=round(A_re[w]*64);
        A_im[w]=round(A_im[w]*64);
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
         //printf("A = \t\n");
         //print_complex_matrix(A_re, A_im, row, col);
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
    float d = lemda * 0.125;
    float kc = 2 * PI / lemda;
    const int nd = 512;
    const int len_t_theta = number_angle;
    float *t_theta = (float *)malloc(len_t_theta * sizeof(float));
    printf("Input angle: \t\t");
    for (int a = 0; a < len_t_theta; a++)
    {
        t_theta[a] = angle[a];
        printf("%.0f, ", angle[a]);
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
    for (int i = 0; i < M * M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        cpp_division(&R_xx_re[i], &R_xx_im[i], M_ptr, M_ptr_im);
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
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
    /* 30度
    R_xx_re[0]=7983.00000;R_xx_im[0]=+0.00000;     R_xx_re[1]=5.00000;R_xx_im[1]=-7975.00000;    R_xx_re[2]=-8011.00000;R_xx_im[2]=14.00000;   R_xx_re[3]=-5.00000;R_xx_im[3]=8004.00000;
    R_xx_re[4]=5.00000;R_xx_im[4]=975.00000;     R_xx_re[5]=8065.00000;R_xx_im[5]=0.00000;     R_xx_re[6]=-16.00000;R_xx_im[6]=-8037.00000;  R_xx_re[7]=-8017.00000;R_xx_im[7]=0.00000;
    R_xx_re[8]=-8011.00000;R_xx_im[8]=-14.00000;  R_xx_re[9]=-16.00000;R_xx_im[9]=8037.00000;   R_xx_re[10]=8191.00000;R_xx_im[10]=0.00000;     R_xx_re[11]=14.00000;R_xx_im[11]=-8051.00000;
    R_xx_re[12]=-5.00000;R_xx_im[12]=-8004.00000;   R_xx_re[13]=-8017.00000;R_xx_im[13]=-0.00000;   R_xx_re[14]=14.00000;R_xx_im[14]=8051.00000;    R_xx_re[15]=8216.00000;R_xx_im[15]=0.00000;
    */
    //*0度
    R_xx_re[0]=8290.00000;R_xx_im[0]=0.00000;     R_xx_re[1]=8274.00000;R_xx_im[1]=-15.00000;   R_xx_re[2]=8319.00000;R_xx_im[2]=-17.00000;   R_xx_re[3]=8278.00000;R_xx_im[3]=0.00000;
    R_xx_re[4]=8274.00000;R_xx_im[4]=15.00000;    R_xx_re[5]=8365.00000;R_xx_im[5]=0.00000;     R_xx_re[6]=8328.00000;R_xx_im[6]=8.00000;     R_xx_re[7]=8293.00000;R_xx_im[7]=15.00000;
    R_xx_re[8]=8319.00000;R_xx_im[8]=17.00000;    R_xx_re[9]=8328.00000;R_xx_im[9]=-8.00000;    R_xx_re[10]=8516.00000;R_xx_im[10]=0.00000;     R_xx_re[11]=8351.00000;R_xx_im[11]=12.00000;
    R_xx_re[12]=8278.00000;R_xx_im[12]=-0.00000;    R_xx_re[13]=8293.00000;R_xx_im[13]=-15.00000;   R_xx_re[14]=8351.00000;R_xx_im[14]=-12.00000;   R_xx_re[15]=8476.00000;R_xx_im[15]=0.00000;
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
    /*
    for (int16_t i = 0; i < M * M; ++i)
    {
        R_xx_re[i] = round(R_xx_re[i]*64);
        R_xx_im[i] = round(R_xx_im[i]*64);
        //printf("(%hd,%hd) ", R_xx_re_i[i], R_xx_im_i[i]); //M=4時，R_xx:120~-120，M=16時，R_xx:30~-30，
    }
    */
        /*
    H:
0 -100 55 -54 /557 127 0 97    
100 0 127 53 /-54 53 -97 0     
1600 0 0 0 /-12 -4 -4 244      
97 52 97 64 /0 176 -4744 -1    
-557 -127 0 -97 0 -100 55 -54 
54 -53 97 0 100 0 127 53      
12 4 4 -244 1600 0 0 0        
0 -176 4744 1 97 52 97 64     
*/
    /*Q:
0 -100 -95 2 146 82 137 120
14 -2 7 51 92 -66 -6 -200
239 -38 -38 -9 -44 16 -22 48
14 50 57 34 106 169 -153 16
-83 -113 -111 -21 -114 80 -99 -64
8 -54 -46 28 66 -102 -105 120
1 4 4 -246 52 -9 -26 -32
0 -176 191 -8 -10 4 18 96
R:
1712 41 14 35 -6 39 -285 244
0 256 -3290 68 -282 -11 -1044 -110
0 0 3312 76 -151 72 -1110 -27
0 0 0 256 -1525 37 -580 -2
0 0 0 0 704 201 -1886 46
0 0 0 0 0 112 -3072 2
0 0 0 0 0 0 2320 10
0 0 0 0 0 0 0 32
    */
    /*張懷祐給的input
    R_xx_re[0]=0;R_xx_im[0]=-557;     R_xx_re[1]=-100;R_xx_im[1]=-127;    R_xx_re[2]=55;R_xx_im[2]=0;   R_xx_re[3]=-54;R_xx_im[3]=-97;
    R_xx_re[4]=100;R_xx_im[4]=54;     R_xx_re[5]=0;R_xx_im[5]=-53;     R_xx_re[6]=127;R_xx_im[6]=97;  R_xx_re[7]=53;R_xx_im[7]=0;
    R_xx_re[8]=1600;R_xx_im[8]=12;  R_xx_re[9]=0;R_xx_im[9]=4;   R_xx_re[10]=0;R_xx_im[10]=4;     R_xx_re[11]=0;R_xx_im[11]=-244;
    R_xx_re[12]=97;R_xx_im[12]=0;   R_xx_re[13]=52;R_xx_im[13]=-176;   R_xx_re[14]=97;R_xx_im[14]=4744;    R_xx_re[15]=64;R_xx_im[15]=1;
    */
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
        printf(" %.4f\n", S_MUSIC_dB[i]);

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
    float angle[100] = {15, -47, -28, -10, 18, 35, 45, 58}; // angle of array
    int number_angle = 1;
    int M = 4;
    int snr = 10;
    int qr_iter = 1;
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
   // printf(L_GREEN "Total MVDR time : \t%.3f(ms)\n" CLOSE, (timeMVDR_end - timeMVDR_start) / CLOCKS_PER_SEC * 1000);
    //printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //printf(L_GREEN "-> transpose time : \t%.3f(ms)\n" CLOSE, total_pre_transpose_time / 1000);
    //total_multiply_time = 0;      // set to 0
    //total_pre_transpose_time = 0; // set to 0
    //=====================================================================
    //printf("Size of int: %zu bytes\n", sizeof(int));
    // 20---------------------------------------------------------------------------------------------------
    // int rowA = 20;
    // int colA = rowA;
    // int rowB = 20;
    // int colB = 20;
    // float A_re[] = {36, -27, 45, 45, 24, -7, 30, 14, 40, -22, -29, 8, 48, 37, -17, 1, 45, 39, -5, 16, 2, 36, 35, 35, 4, -12, -8, 44, -19, 17, 12, 31, 45, 27, 10, 38, -12, -12, -21, 45, -22, -28, 49, 38, 8, 37, 48, 6, 12, 24, 48, -9, -7, 47, 45, -28, 36, 1, -25, -16, -8, 26, -14, -17, 35, 24, 23, 10, 38, 14, 33, 32, -23, 36, 49, 0, -11, 38, -21, 0, 40, -15, 38, -7, 21, 25, -16, 24, 49, 21, 24, 30, 24, 33, 18, 17, -11, -20, 22, 42, 13, 8, 33, -27, 5, -20, -30, 20, 7, 5, -1, 28, 2, 16, 46, -10, 23, 18, 28, 47, 12, 1, -23, 9, 45, 24, 10, -9, 22, 7, 38, 37, -22, 16, -22, 46, 47, 11, -27, -28, 15, 35, 44, 44, -23, -25, -9, 47, 14, 26, 33, 10, 20, 36, -20, 30, 23, -18, -4, 33, -13, -6, 7, 27, -5, 40, -4, 42, 5, 34, -24, -1, -20, 17, -30, 47, 45, -26, -15, 2, 42, -14, 48, -25, 42, -18, -15, -17, -26, 36, 39, -3, 46, 35, 42, -20, 2, -16, -25, -23, -23, 48, 29, 37, 46, -2, 34, 37, 30, -5, 12, 17, 43, 9, 40, -8, 44, -1, 22, 42, -21, 25, 11, 29, 6, 37, -2, -7, -2, 6, 27, 44, 27, 36, 4, -9, -11, -5, -22, -18, 28, 40, 8, 12, 2, 31, 41, -4, 44, -24, -3, -30, 5, -4, 8, 16, -5, 13, 35, -15, -17, 28, 41, 18, -23, 49, -21, -29, 34, -26, 13, 27, -24, -4, -10, 17, 30, 4, 3, -28, 9, -9, 21, 20, 36, 16, -1, -26, 13, 28, 41, -28, 6, 33, 48, -21, 5, -12, 4, -21, 36, 33, 8, -21, 1, -9, 5, 34, 47, -7, 23, 31, 36, 46, 14, 17, 25, -14, 31, -16, -3, 19, -15, -13, -14, 37, 32, 7, -18, 47, 30, 15, 11, 11, 44, 46, 24, 41, 37, 14, 38, -3, 4, 13, -21, 29, -7, 42, -14, 4, 24, -19, -6, 33, 26, 12, -29, -23, -22, 39, 36, -15, -23, 13, 23, -28, -6, -28, -5, -13, 23, 37, 49, 16, -6, 30, 32, -14, 37, -18, 17, 5, -5, 45, 36, 23, -28, 1, 9, -21, -19, 26, 39, 26, -27, 39, 41, 29, 33, 46};
    // float A_im[] = {-1, 18, -10, -27, -4, 23, -28, 43, -10, 23, -20, -22, 46, -16, -16, 19, -22, -27, -24, -20, -28, 45, 17, 12, -23, -30, -7, 16, -6, 3, -12, 8, 47, 2, 42, 26, -1, 39, 38, -2, 10, 11, -7, 33, 31, 14, -11, 35, -10, 18, -29, 24, 7, 30, 29, -21, 1, -4, 36, -26, 36, -6, -5, 27, -25, 45, 42, 14, -13, 9, 29, -8, 38, 26, -30, 6, 17, 31, 30, 24, 42, 10, 9, 3, -21, 33, -19, -16, 17, 0, -5, 26, -2, 20, 7, -28, 23, 17, 39, 0, 23, 46, 39, 9, 10, -28, 40, -22, 19, -30, -20, 20, -10, 12, -5, -20, -1, -25, -30, -21, -23, -18, -1, 13, 24, 33, -26, 2, 0, 16, 31, 32, 27, 20, -10, -1, 23, 39, 25, 28, -11, -9, -18, 38, 27, 24, 10, -12, 35, 31, -21, 13, -22, -9, 15, -18, 34, 31, 10, -24, 48, -6, -12, -17, 47, 45, 25, 28, 11, -11, -9, 28, 23, -17, 20, -17, 24, 30, 37, 9, 39, 25, -8, -11, -29, -6, 11, 45, -20, 48, 2, 19, -6, -3, -11, 12, -28, 41, 32, 31, 25, -19, -14, 42, 26, -7, 46, -3, -19, 47, 13, 45, 44, -27, -7, 35, 20, -25, 3, -22, -25, -29, 15, -11, 11, -5, 3, 39, -23, -10, -21, 8, 46, 47, -8, -10, -19, 37, 45, -30, -5, 9, 32, -17, 17, 20, 31, 7, 30, 14, -8, 22, -30, -8, 15, 13, 4, 35, 29, 32, 14, 6, 43, -24, -20, 44, -5, 33, 32, -13, -1, 13, -16, 47, -16, 0, 38, -11, 25, -30, 31, 0, -7, 16, -2, 31, 36, 27, 38, 37, 35, 7, -13, 8, 31, 45, 17, 35, 42, -6, 3, -12, -10, -18, 46, -18, 44, 14, -20, -8, 23, -29, 41, -20, 6, 8, -23, 41, 29, -1, 0, 49, 21, 29, -23, -20, -9, 41, 0, -12, -27, 34, -9, -9, -14, -10, 36, -24, 20, 43, 33, -3, -23, 26, 12, 18, 15, -1, -27, 21, 43, -22, 3, 24, 35, 33, 30, 3, 36, -4, 38, 31, -19, 23, 27, 12, -15, 4, 3, -21, 25, -26, -25, 37, 16, 39, 6, -14, 21, 35, -15, -29, 29, 11, -19, -7, 46, 35, 36, -21, -11, 4, -27, -12, 32, 27, 15, -29, 5, -24};
    // float B_re[] = {43, 7, 21, 8, 11, 35, 3, 38, -15, 3, -15, 34, 3, 2, -26, 37, 25, 13, -18, 1, 15, -11, 37, 25, -25, 32, 48, -13, 36, -9, -28, 29, 32, 41, 13, -27, 24, 7, -23, 48, 42, 22, -6, 4, 13, 16, -25, 46, 25, 19, -5, 13, -21, 33, -11, -28, -13, 46, 22, 28, -26, -10, 38, 16, -20, 20, 32, -1, 34, -5, 33, 14, 9, 49, 13, 19, 38, 47, 10, 11, -16, -30, -29, 5, -11, 29, 6, 20, 37, -11, -9, 1, 21, -28, -22, -9, 30, -25, -3, 6, 44, 32, 44, -16, 47, -12, 11, 14, 45, -19, 15, 12, 13, 9, -27, 41, 46, 26, 10, 9, 46, 21, 33, -29, 19, -13, 49, 18, 38, -14, -28, -24, -13, 36, 41, -26, 11, 15, 34, 22, 34, 28, -9, -14, 49, -7, 17, 10, 20, -25, 16, 33, 13, 12, -26, 32, 31, 2, -7, 34, -27, -10, -11, -9, -23, 41, 40, 49, -3, 35, -28, -22, 37, -5, 49, 11, 42, 16, 8, 22, -23, 4, 24, -2, 18, -4, -11, -6, -22, -15, -12, -10, -23, -15, 33, 33, 29, -1, -23, 16, 5, 31, 16, -6, 22, 11, -19, 41, -22, 48, 41, 15, 44, 1, -29, -17, 29, 36, 46, -16, -14, 0, -15, -18, 20, 21, 23, 9, 47, 20, -19, 19, -29, -29, -9, 6, 22, -6, -21, 28, 0, 40, -28, 5, 45, 15, 16, 4, -1, -28, 2, -14, 25, 16, 1, -6, 5, -14, 23, 28, 15, -22, 3, -5, -8, 15, 40, 46, 36, 42, -5, 34, -20, 25, 25, -26, -19, 29, 15, 14, -25, -20, 32, 17, -2, -16, -14, 43, 29, 38, 17, 44, 46, -16, 11, 38, 35, 3, -13, -29, 32, 42, -18, 29, 15, 1, -8, 20, -5, 27, 0, 28, -11, 29, -8, 44, -30, -16, 22, 32, -22, -13, 6, 39, -5, 38, 38, 38, -12, 16, 24, -28, 27, 42, 19, 28, 23, -30, 34, 30, 23, -6, 36, 14, -23, -26, -22, -29, 3, 17, -4, 25, 46, 40, 21, 16, 6, 45, 34, 26, 28, -27, 7, -15, 3, 43, 42, 31, 22, -5, 22, -28, -15, -8, 42, 20, 4, -25, 7, -10, -13, 33, -11, 6, 24, 37, -2, -12, -17, 4, 49, -29, 41, 35, 49, 42, -22, -20, 20, 23};
    // float B_im[] = {40, 12, 49, 33, -2, 26, 15, 26, 4, -18, 0, -15, -4, -19, 15, -14, 28, 44, -19, 0, 47, -26, 13, -18, 49, 44, 9, 22, 23, 48, -21, 37, 20, 41, -11, 29, -11, -29, 25, 34, 22, 28, -14, 14, 42, 7, 26, 45, 24, 45, 1, -15, 24, 23, 28, 15, 10, -11, -29, 33, 0, 27, 2, 18, -2, 46, 42, 4, 33, 11, 45, 13, -26, -16, 21, -10, 30, 1, 30, -5, 0, -11, -25, 14, 23, -9, 4, 4, 16, -27, 15, 6, 30, -28, -29, 34, 22, 26, 48, 7, 19, 2, 42, 44, 2, -20, 42, 25, 23, -21, -5, -3, 38, 37, 14, 39, 11, -26, 6, -20, 19, 45, -4, -25, 1, 29, -3, 20, 26, 29, 23, 21, -7, -23, 48, -18, -20, 15, 0, 24, -27, 3, 1, 17, 26, 44, 6, -6, -27, -28, 49, 24, -12, 12, 43, 20, 26, -27, 32, -5, -20, 24, 48, -2, -27, 15, 16, 35, -29, 10, 23, 18, -24, -5, 17, -23, 26, 26, -17, 25, -29, -30, -11, -26, 18, 2, 27, -21, -24, 45, 28, -28, 11, 42, 26, -30, 39, 36, 8, 11, 40, 27, 47, 23, -15, 18, -6, 4, 24, -1, 26, 3, 23, 15, -30, 40, 13, -17, 28, 46, 43, 19, -14, 1, 24, -29, -10, -30, 45, -23, 10, 7, -23, -28, -23, 8, 14, 8, -20, 18, 35, 12, 22, -24, 16, 27, 9, -3, 21, 35, 30, 12, 35, 9, 49, -15, 17, 19, -30, 29, -7, 12, 22, 21, 40, -28, 3, -29, 18, -18, -24, 36, 43, 0, 3, -24, -17, 38, 41, 38, -21, -17, 1, -27, 49, -2, -23, -28, 40, 27, 0, 19, 26, 47, -22, -9, -12, 12, 32, -6, 10, 34, -3, 9, 32, 11, 17, 20, 18, 33, 34, -30, 29, 18, 6, 26, 4, -29, -6, 26, 44, 25, -17, -28, -16, -14, -8, 45, -2, -6, -3, 28, 23, 22, 27, 33, 19, -3, 17, 11, 27, 42, 20, 16, -19, 12, 12, 26, -20, -10, 21, 20, -19, -1, -1, 49, -4, 35, 47, -11, 20, 48, -20, -22, -6, 42, -30, -20, 35, 23, -30, -19, 9, -13, 16, 3, -21, -18, 3, -11, 42, 9, 9, 1, 28, -19, -15, 25, -20, 36, 15, 33, 40, -3, 27, -14, 34, 28, 2, -21};

    // 10 ---------------------------------------------------------------------------------------------------
    /*
    int rowA = 32;
    int colA = 32;
    int rowB = 32;
    int colB = 32;
    float A_re[10000] = {0.0};
    float A_im[10000] = {0.0};
    float B_re[10000] = {0.0};
    float B_im[10000] = {0.0};
    for (int i = 0; i < rowA * rowB; i++)
    {
        A_re[i] = i;
        A_im[i] = -50 + i;
        B_re[i] = -40 + i;
        B_im[i] = 40 + i;
    }
    */
    // // 5 ---------------------------------------------------------------------------------------------------
    // int rowA = 5;
    // int colA = rowA;
    // int rowB = 5;
    // int colB = 5;
    // float A_re[] = {5, 17, 5, 12, 1, 9, 7, 15, 5, 3, 2, 7, 2, 11, 10, 8, -4, -7, -3, -12, -13, -8, -9, -17, 15};
    // float A_im[] = {9, 9, 12, 7, 7, 8, 6, -2, 4, 2, 3, 5, 6, 8, 3, -15, 9, 11, 13, -9, -4, -7, 5, 13, -7};
    // float B_re[] = {5, 5, 4, 8, 2, 8, 6, 7, 4, 2, 1, 15, 2, 11, 10, 12, 9, 11, 13, 9, 4, -7, 5, -13, 9};
    // float B_im[] = {1, 3, -3, 15, 5, 9, 7, -5, 5, 3, 2, 1, -6, 8, 3, -7, -4, -8, 5, 12, 13, -8, -9, -17, -1};

    // 4x4 ---------------------------------------------------------------------------------------------------
    // int rowA = 4;
    // int colA = rowA;
    // int rowB = 4;
    // int colB = 4;
    // float A_re[] = {5, 17, 5, 12, 1, 9, 7, 15, 5, 3, 2, 7, 2, 11, 10, 8};
    // float A_im[] = {9, 9, 12, 7, 7, 8, 6, -2, 4, 2, 3, 5, 6, 8, 3, -15};
    // float B_re[] = {5, 5, 4, 8, 2, 8, 6, 7, 4, 2, 1, 15, 2, 11, 10, 12};
    // float B_im[] = {1, 3, -3, 15, 5, 9, 7, -5, -5, 3, 2, 1, -6, 8, 3, -7};
    // ---------------------------------------------------------------------------------------------------

    // 3x3 ---------------------------------------------------------------------------------------------------
    // Gram-Schmidt
    // int rowA = 3;
    // int colA = rowA;
    // int rowB = 3;
    // int colB = 3;
    // float A_re[] = {1, 2, 4, 0, 0, 5, 0, 3, 6};
    // float A_im[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // float B_re[] = {5, 5, 4, 2, 8, 6, 4, 2, 1};
    // float B_im[] = {1, 3, -3, 5, 9, 7, -5, 3, 2};
    // --------------------------------------------------------------------------------------------------
    /*
    struct timeval start_Multiply, end_Multiply, diff_Multiply; // estimate matrix multiply time
    struct timeval start_Eigen, end_Eigen, diff_Eigen;          // estimate matrix multiply time
    float *C_re = (float *)malloc(rowA * colB * sizeof(float));
    float *C_im = (float *)malloc(rowA * colB * sizeof(float));
    float *D = (float *)malloc(rowA * colB * sizeof(float));

    // printf(L_BLUE "\n------------mat A--------------------------\n" CLOSE);
    // print_complex_matrix(&A_re[0], &A_im[0], rowA, colA);
    // printf(L_BLUE "\n------------mat B--------------------------\n" CLOSE);
    // print_complex_matrix(&B_re[0], &B_im[0], rowB, colB);
    // printf(L_BLUE "\n------------mat C--------------------------\n" CLOSE);

    gettimeofday(&start_Multiply, NULL);
    complex_matrix_multiplication(A_re, A_im, B_re, B_im, C_re, C_im, rowA, rowB, colB);
    gettimeofday(&end_Multiply, NULL);
    timersub(&end_Multiply, &start_Multiply, &diff_Multiply);
    // print_complex_matrix(C_re, C_im, rowA, colB);
    // printf("\n--------------------------------------\n");
    //     printf(YELLOW "Accumulate data process: %ld (ms)\n" CLOSE, data_process_time / 1000);
    //      printf(L_GREEN "Elapsed MUSIC Algorithm time: %ld(ms)\n" CLOSE, (long int)diff_MUSIC.tv_usec/1000);
    //    printf("--------------------------------------\n");
    printf(L_BLUE "\n--------------------------------------\n" CLOSE);
    printf(L_GREEN "\nElapsed AVX matrix[%d x %d] time: %ld(us)\n" CLOSE, rowA, rowB, (long int)diff_Multiply.tv_usec);
    //-----------------------------------------------------------------------------------------------------------------
    // print_complex_matrix_matlab(C, row, col);
    //  std::complex<double> *Q = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //  std::complex<double> *R = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //  qr(A, Q, R, 3, 3);
    float *Ve_re = (float *)malloc(rowA * colA * sizeof(float));
    float *Ve_im = (float *)malloc(rowA * colA * sizeof(float));
    memset(Ve_re, 0, rowA * colA * sizeof(float));
    memset(Ve_im, 0, rowA * colA * sizeof(float));
    //-----------------------------------------------------------------------------------------------------------------
    float *De_re = (float *)malloc(rowA * colA * sizeof(float));
    float *De_im = (float *)malloc(rowA * colA * sizeof(float));
    //-----------------------------------------------------------------------------------------------------------------
    float time_Eigen = 0.0;
    gettimeofday(&start_Eigen, NULL);
    eigen(A_re, A_im, Ve_re, Ve_im, De_re, De_im, rowA, colA, iter);
    gettimeofday(&end_Eigen, NULL);
    timersub(&end_Eigen, &start_Eigen, &diff_Eigen);
    time_Eigen = (float)diff_Eigen.tv_usec;
    printf(L_GREEN "\nElapsed Eigen time: %.3f(ms)\n" CLOSE, time_Eigen / 1000);
    */
    // printf("----------Vector------------\n");
    // print_complex_matrix(Ve_re, Ve_im, rowA, colA);
    // printf("----------Eigen------------\n");
    // print_complex_matrix(De_re, De_im, rowA, colA);
}