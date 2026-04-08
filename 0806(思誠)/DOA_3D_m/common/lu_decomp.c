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
#include "lu_decomp.h"
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
// ======= LU decomposition =======
// ================================

void LU_decomposition(float *A_re, float *A_im, float *L_re, float *L_im, float *U_re, float *U_im, int N) {

    // Initialize L as the identity matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                L_re[i * N + j] = 1.0;
                L_im[i * N + j] = 0.0;
            } else {
                L_re[i * N + j] = 0.0;
                L_im[i * N + j] = 0.0;
            }
        }
    }

    // Initialize U as a copy of A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            U_re[i * N + j] = A_re[i * N + j];
            U_im[i * N + j] = A_im[i * N + j];
        }
    }

    // LU decomposition without pivoting
    for (int k = 0; k < N - 1; k++) {
        if (U_re[k * N + k] == 0.0 && U_im[k * N + k] == 0.0) {
            printf("Zero pivot encountered. No pivoting cannot proceed.\n");
            return;
        }

        for (int i = k + 1; i < N; i++) {
            // Division: L(i, k) = U(i, k) / U(k, k)
            float Re_a = U_re[i * N + k];
            float Im_b = U_im[i * N + k];
            float Re_c = U_re[k * N + k];
            float Im_d = U_im[k * N + k];

            cpp_division(&Re_a, &Im_b, &Re_c, &Im_d);

            L_re[i * N + k] = Re_a;
            L_im[i * N + k] = Im_b;

            // U(i, :) -= L(i, k) * U(k, :)
            for (int j = k; j < N; j++) {
                float U_ik_re = U_re[i * N + j];
                float U_ik_im = U_im[i * N + j];
                float U_kj_re = U_re[k * N + j];
                float U_kj_im = U_im[k * N + j];

                // L(i, k) * U(k, j)
                float mult_re = Re_a * U_kj_re - Im_b * U_kj_im;
                float mult_im = Re_a * U_kj_im + Im_b * U_kj_re;

                // Subtract the product from U(i, j)
                U_re[i * N + j] = U_ik_re - mult_re;
                U_im[i * N + j] = U_ik_im - mult_im;
            }
        }
    }
}

void matrix_inverse_LU(float *A_re, float *A_im, float *A_inv_re, float *A_inv_im, int N) {
    float *L_re = (float *)calloc(N * N , sizeof(float));
    float *L_im = (float *)calloc(N * N , sizeof(float));
    float *U_re = (float *)calloc(N * N , sizeof(float));
    float *U_im = (float *)calloc(N * N , sizeof(float));

    struct timeval time_LU_D_start, time_LU_D_end, time_LU_D_diff; // time initial
    // LU 分解
    gettimeofday(&time_LU_D_start, NULL);
    LU_decomposition(A_re, A_im, L_re, L_im, U_re, U_im, N);
    gettimeofday(&time_LU_D_end, NULL);
    timersub(&time_LU_D_end, &time_LU_D_start, &time_LU_D_diff);

    //float time_LU_D;
    //time_LU_D = time_LU_D_diff.tv_sec * 1000000 + time_LU_D_diff.tv_usec;
    //printf("LU time: \t\t%.3f(ms)\n", time_LU_D / 1000);
    
    float *L_inv_re = (float *)malloc(N * N * sizeof(float));
    float *L_inv_im = (float *)malloc(N * N * sizeof(float));
    float *U_inv_re = (float *)calloc(N * N , sizeof(float));
    float *U_inv_im = (float *)calloc(N * N , sizeof(float));

    // 初始化單位矩陣
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L_inv_re[i * N + j] = (i == j) ? 1.0 : 0.0;
            L_inv_im[i * N + j] = 0.0;
        }
    }

    // 求 L 的反矩陣（下三角矩陣）
    for (int i = 0; i < N; i++) {
        // 對角元素的處理
        if (L_re[i * N + i] == 0.0) {
            printf("矩陣不可逆！\n");
            return; // 如果對角元素為零，退出
        }
        L_inv_re[i * N + i] = 1.0 / L_re[i * N + i];
        L_inv_im[i * N + i] = 0.0; // 對角元素的虛部始終為0

        // 非對角元素的處理
        for (int j = 0; j < i; j++) {
            float sum_re = 0.0;
            float sum_im = 0.0;
            for (int k = j; k < i; k++) {
                sum_re += L_re[i * N + k] * L_inv_re[k * N + j] - L_im[i * N + k] * L_inv_im[k * N + j];
                sum_im += L_re[i * N + k] * L_inv_im[k * N + j] + L_im[i * N + k] * L_inv_re[k * N + j];
            }
            L_inv_re[i * N + j] = -sum_re / L_re[i * N + i];
            L_inv_im[i * N + j] = -sum_im / L_re[i * N + i];
        }
    }
    /*
    printf("---------- L ------------\n");
    print_complex_matrix(L_re, L_im, N, N);
    printf("---------- L_inv ------------\n");
    print_complex_matrix(L_inv_re, L_inv_im, N, N);
    //-----------------------------------------------------------
    // test inv
    float *L_test_I_re = (float *)malloc(N * N * sizeof(float));
    float *L_test_I_im = (float *)malloc(N * N * sizeof(float));
    complex_matrix_multiplication(L_re, L_im, L_inv_re, L_inv_im, L_test_I_re, L_test_I_im, N, N, N);
    printf("---- L_test_I -----\n");
    print_complex_matrix(L_test_I_re, L_test_I_im, N, N);
    */
   
    //-----------------------------------------------------------
    // 求 U 的反矩陣
    for (int i = N - 1; i >= 0; i--) {
        // 對角元素處理
        if (U_re[i * N + i] == 0.0) {
            printf("矩陣不可逆！\n");
            return; // 如果對角元素為零，退出
        }
        U_inv_re[i * N + i] = 1.0;
        U_inv_im[i * N + i] = 0.0; // 對角元素虛部為0
        cpp_division(&U_inv_re[i * N + i], &U_inv_im[i * N + i], &U_re[i * N + i], &U_im[i * N + i]);
        // 非對角元素處理
        for (int j = i + 1; j < N; j++) {
            float sum_re = 0.0;
            float sum_im = 0.0;
            for (int k = i + 1; k <= j; k++) {
                sum_re -= U_re[i * N + k] * U_inv_re[k * N + j] - U_im[i * N + k] * U_inv_im[k * N + j];
                sum_im -= U_re[i * N + k] * U_inv_im[k * N + j] + U_im[i * N + k] * U_inv_re[k * N + j];
            }

            cpp_division2(sum_re, sum_im, &U_re[i * N + i], &U_im[i * N + i], &U_inv_re[i * N + j], &U_inv_im[i * N + j]);
        }
    }
    /*
    printf("---------- U ------------\n");
    print_complex_matrix(U_re, U_im, N, N);
    printf("---------- U_inv ------------\n");
    print_complex_matrix(U_inv_re, U_inv_im, N, N);
    //-----------------------------------------------------------
    // test inv
    float *U_test_I_re = (float *)malloc(N * N * sizeof(float));
    float *U_test_I_im = (float *)malloc(N * N * sizeof(float));
    complex_matrix_multiplication(U_re, U_im, U_inv_re, U_inv_im, U_test_I_re, U_test_I_im, N, N, N);
    printf("---- U_test_I -----\n");
    print_complex_matrix(U_test_I_re, U_test_I_im, N, N);
    //-----------------------------------------------------------
    */
    
    // 計算 A 的反矩陣 A_inv =  U_inv * L_inv 
    complex_matrix_multiplication(U_inv_re, U_inv_im, L_inv_re, L_inv_im, A_inv_re, A_inv_im, N, N, N);

    // 釋放內存
    free(L_re); free(L_im); free(U_re); free(U_im);
    free(L_inv_re); free(L_inv_im); free(U_inv_re); free(U_inv_im);
}

void trace(float *theta_re, float *theta_im, float *S_ML_re, float *S_ML_im, int rowA, int colA, int i){
    float temp_re = 0.0 ;
    float temp_im = 0.0 ;
    rowA = rowA;
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