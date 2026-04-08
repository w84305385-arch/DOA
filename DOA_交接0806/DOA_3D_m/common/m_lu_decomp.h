#ifndef M_LU_DECOMP_H
#define M_LU_DECOMP_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;
extern int num_row_blocks;
// LU
void m_LU_decomposition(float *A_re, float *A_im, float *L_re, float *L_im, float *U_re, float *U_im, int N);
void m_matrix_inverse_LU(float *A_re, float *A_im, float *A_inv_re, float *A_inv_im, int N);


#endif