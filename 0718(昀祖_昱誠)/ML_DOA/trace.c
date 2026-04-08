// g++ -mavx512f -g -o trace trace.c -Wall -Wextra -std=c++14 math_func.a
// ./trace
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
void trace(int16_t *theta_re, int16_t *theta_im, int16_t *S_ML_re, int16_t *S_ML_im, int rowA, int colA, int i){
    int16_t temp_re = 0.0 ;
    int16_t temp_im = 0.0 ;
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

void print_complex_matrix(float *matA_re, float *matA_im, int rowA, int colA)
{
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colA; j++)
        {
            printf("\t%.3f ", matA_re[i * colA + j]);
            printf("+ %.3fi", matA_im[i * colA + j]);
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

int main(){
    int M = 4;
    int len_dth = 12;
    float S_re;
    float S_im;
    int16_t *a_re = (int16_t *)malloc(M * M * sizeof(int16_t));
    int16_t *a_im = (int16_t *)malloc(M * M * sizeof(int16_t));
    int16_t *b_re = (int16_t *)malloc(len_dth * sizeof(int16_t));
    int16_t *b_im = (int16_t *)malloc(len_dth * sizeof(int16_t));
    for(int w=0;w<M*M;w++){
        a_re[w]=w;
        a_im[w]=w+1;
    }
    print_complex_matrix_i(a_re, a_im, M, M);
    for(int w=0;w<len_dth;w++){
        //printf("w=%d\n",w);
        //trace(a_re, a_im, &b_re[w], &b_im[w], M, M, w);
        trace(a_re, a_im, b_re, b_im, M, M, w);
    }

    printf("trace結果:\n");
    print_complex_matrix_i(b_re, b_im, 1, len_dth);

}