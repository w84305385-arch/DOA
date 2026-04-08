#ifndef LU_DECOMP_H
#define LU_DECOMP_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;

// LU
void LU_decomposition(float *A_re, float *A_im, float *L_re, float *L_im, float *U_re, float *U_im, int N);
void matrix_inverse_LU(float *A_re, float *A_im, float *A_inv_re, float *A_inv_im, int N);
void trace(float *theta_re, float *theta_im, float *S_ML_re, float *S_ML_im, int rowA, int colA, int i);
#endif