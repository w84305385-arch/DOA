//--------------------
#define PI acos(-1)
#define AVX 16            
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
    float *L_re = (float *)calloc(N * N + AVX, sizeof(float));
    float *L_im = (float *)calloc(N * N + AVX, sizeof(float));
    float *U_re = (float *)calloc(N * N + AVX, sizeof(float));
    float *U_im = (float *)calloc(N * N + AVX, sizeof(float));

    struct timeval time_LU_D_start, time_LU_D_end, time_LU_D_diff; // Timer initialization

    // LU decomposition
    gettimeofday(&time_LU_D_start, NULL);
    LU_decomposition(A_re, A_im, L_re, L_im, U_re, U_im, N);
    gettimeofday(&time_LU_D_end, NULL);
    timersub(&time_LU_D_end, &time_LU_D_start, &time_LU_D_diff);

    // Optional timing display
    // float time_LU_D;
    // time_LU_D = time_LU_D_diff.tv_sec * 1000000 + time_LU_D_diff.tv_usec;
    // printf("LU time: \t\t%.3f(ms)\n", time_LU_D / 1000);

    float *L_inv_re = (float *)malloc(N * N * sizeof(float) + AVX * sizeof(float));
    float *L_inv_im = (float *)malloc(N * N * sizeof(float) + AVX * sizeof(float));
    float *U_inv_re = (float *)calloc(N * N + AVX, sizeof(float));
    float *U_inv_im = (float *)calloc(N * N + AVX, sizeof(float));

    // Initialize identity matrix for L_inv
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L_inv_re[i * N + j] = (i == j) ? 1.0 : 0.0;
            L_inv_im[i * N + j] = 0.0;
        }
    }

    // Compute inverse of lower triangular matrix L
    for (int i = 0; i < N; i++) {
        // Handle diagonal element
        if (L_re[i * N + i] == 0.0) {
            printf("Matrix is not invertible!\n");
            return; // Exit if diagonal is zero
        }
        L_inv_re[i * N + i] = 1.0 / L_re[i * N + i];
        L_inv_im[i * N + i] = 0.0; // Diagonal of L is always real

        // Handle off-diagonal elements
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
    // Debug: Print L and its inverse
    printf("---------- L ------------\n");
    print_complex_matrix(L_re, L_im, N, N);
    printf("---------- L_inv ------------\n");
    print_complex_matrix(L_inv_re, L_inv_im, N, N);

    // Test: L * L_inv should be identity
    float *L_test_I_re = (float *)malloc(N * N * sizeof(float));
    float *L_test_I_im = (float *)malloc(N * N * sizeof(float));
    complex_matrix_multiplication(L_re, L_im, L_inv_re, L_inv_im, L_test_I_re, L_test_I_im, N, N, N);
    printf("---- L_test_I -----\n");
    print_complex_matrix(L_test_I_re, L_test_I_im, N, N);
    */

    // Compute inverse of upper triangular matrix U
    for (int i = N - 1; i >= 0; i--) {
        // Handle diagonal element
        if (U_re[i * N + i] == 0.0) {
            printf("Matrix is not invertible!\n");
            return; // Exit if diagonal is zero
        }
        U_inv_re[i * N + i] = 1.0;
        U_inv_im[i * N + i] = 0.0; // Diagonal of U is always real
        cpp_division(&U_inv_re[i * N + i], &U_inv_im[i * N + i], &U_re[i * N + i], &U_im[i * N + i]);

        // Handle off-diagonal elements
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
    // Debug: Print U and its inverse
    printf("---------- U ------------\n");
    print_complex_matrix(U_re, U_im, N, N);
    printf("---------- U_inv ------------\n");
    print_complex_matrix(U_inv_re, U_inv_im, N, N);

    // Test: U * U_inv should be identity
    float *U_test_I_re = (float *)malloc(N * N * sizeof(float));
    float *U_test_I_im = (float *)malloc(N * N * sizeof(float));
    complex_matrix_multiplication(U_re, U_im, U_inv_re, U_inv_im, U_test_I_re, U_test_I_im, N, N, N);
    printf("---- U_test_I -----\n");
    print_complex_matrix(U_test_I_re, U_test_I_im, N, N);
    */

    // Compute inverse of A: A_inv = U_inv * L_inv
    complex_matrix_multiplication(U_inv_re, U_inv_im, L_inv_re, L_inv_im, A_inv_re, A_inv_im, N, N, N);

    // free
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