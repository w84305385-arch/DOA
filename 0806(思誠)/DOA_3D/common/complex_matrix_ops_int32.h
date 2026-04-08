#include <stdint.h>

#ifndef COMPLEX_MATRIX_OPS_INT32_H
#define COMPLEX_MATRIX_OPS_INT32_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;

// matrix ops int32_t
void matrix_transpose_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA);
void complex_matrix_conjugate_transpose_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA);
void complex_matrix_multiplication_int32(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int32_t *matC_re, int32_t *matC_im, int rowA, int rowB, int colB);
void float_matrix_to_q_format(int32_t *out_re, int32_t *out_im, float *in_re, float *in_im, int row, int col);
void q_format_to_float_matrix(float *out_re, float *out_im, int32_t *in_re, int32_t *in_im, int row, int col);
static inline float q_format_to_float(int32_t x, int q_shift)
{
    return (float)x / (float)(1 << q_shift);
}
void print_complex_matrix_int32(int32_t *matA_re, int32_t *matA_im, int rowA, int colA);
#endif

