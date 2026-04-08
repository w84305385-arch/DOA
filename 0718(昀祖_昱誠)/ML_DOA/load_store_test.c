// #define DATA_CSV_MODE 1
// g++ -mavx512f -g -o load_store_test load_store_test.c -Wall -Wextra -std=c++14 math_func.a
// ./load_store_test
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

void matrix_transpose2(float *matA_re, float *matA_im, int rowA, int colA)
{
    float *temp_re = (float *)malloc(rowA * colA * sizeof(float));
    float *temp_im = (float *)malloc(rowA * colA * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));

    __m512 re_vec, im_vec,im_16;
    re_vec = _mm512_loadu_ps(&temp_re[0]);
    im_vec = _mm512_loadu_ps(&temp_im[1]);
    _mm512_storeu_ps(&matA_re[0], re_vec);
    _mm512_storeu_ps(&matA_im[0], im_vec);
        
    im_vec = _mm512_loadu_ps(&temp_im[16]);
    _mm512_storeu_ps(&matA_im[17], im_vec);  
    free(temp_re);
    free(temp_im);
}
void matrix_transpose_avx512(float *matA_re, float *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; j += 16)
        {
            __m512 re_vec = _mm512_loadu_ps(&matA_re[i * colA + j]);
            __m512 im_vec = _mm512_loadu_ps(&matA_im[i * colA + j]);

            // Interleave real and imaginary parts
            __m512 interleave = _mm512_unpacklo_ps(re_vec, im_vec);
            __m512 transposed = _mm512_shuffle_ps(interleave, interleave, _MM_SHUFFLE(2, 3, 0, 1));

            // Store the transposed values
            _mm512_storeu_ps(&matA_re[j * rowA + i], transposed);

            interleave = _mm512_unpackhi_ps(re_vec, im_vec);
            transposed = _mm512_shuffle_ps(interleave, interleave, _MM_SHUFFLE(2, 3, 0, 1));

            _mm512_storeu_ps(&matA_im[j * rowA + i], transposed);
        }
    }
}
void complex_matrix_subtraction2(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matB_re[i * colA + j] = matA_re[i * colA + j]-matB_re[i * colA + j];
            matB_im[i * colA + j] = matA_re[i * colA + j]-matB_im[i * colA + j];
        }
    }
}

int main()
{   
    int rowA = 5;
    int colA = rowA;
    int rowB = 5;
    int colB = 5;
    float A_re[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    float A_im[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    float B_re[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26};
    float B_im[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26};

    

    printf("A\n");
    print_complex_matrix(A_re, A_im, rowA, colA);
    printf("B\n");
    print_complex_matrix(B_re, B_im, rowB, colB);
    printf("B=A-B\n");
    complex_matrix_subtraction2(A_re, A_im, B_re, B_im, 5, 5);
    print_complex_matrix(B_re, B_im, rowB, colB);
}
