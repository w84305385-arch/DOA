#ifndef COMPLEX_MATRIX_OPS_H
#define COMPLEX_MATRIX_OPS_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;

// matrix ops
void matrix_transpose(float *matA_re, float *matA_im, int rowA, int colA);
void print_complex_matrix(float *matA_re, float *matA_im, int rowA, int colA);
void multiply_matrix_avx(float *A, float multiplier, int row, int col);
void complex_matrix_addition(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA);
void complex_matrix_subtraction(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA);
// void complex_matrix_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB);
void complex_matrix_multiplication(float *restrict A_re,
    float *restrict A_im,
    float *restrict B_re,
    float *restrict B_im,
    float       *restrict C_re,
    float       *restrict C_im,
    int rowA, int rowB, int colB);
void complex_matrix_multiplication_scalar(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB);
void complex_matrix_get_columns(float *matA_re, float *matA_im, float *matCol_re, float *matCol_im, int rowA, int colA, int colTarget);
void complex_matrix_get_rows(float *matA_re, float *matA_im, float *matRow_re, float *matRow_im, int colA, int rowTarget);
void complex_matrix_conjugate_transpose(float *matA_re, float *matA_im, int rowA, int colA);
void complex_matrix_conjugate_transpose_multiplication(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA);

#endif