#include <stdint.h>

#ifndef M_COMPLEX_MATRIX_OPS_INT32_H
#define M_COMPLEX_MATRIX_OPS_INT32_H
typedef struct {
    int32_t *A_re, *A_im;
    int32_t *B_re, *B_im;
    int32_t *C_re, *C_im;
    int    rowA;    
    int    rowB;  
    int    colB;  
    int    block_index;
    int    block_size; 
    int    col_start; 
} BlockTaskArg_int32;
// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;

void m_complex_matrix_multiplication_int32_parallel(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int32_t *matC_re, int32_t *matC_im, int rowA, int rowB, int colB);
void block_multiply_task_int32(void *arg);
void m_complex_matrix_multiplication_int32_row_block_parallel(
    int32_t *matA_re, int32_t *matA_im,
    int32_t *matB_re, int32_t *matB_im,
    int32_t *matC_re, int32_t *matC_im,
    int rowA, int rowB, int colB, int num_blocks);

void m_complex_matrix_conjugate_transpose_multiplication_int32_row_block_parallel(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int rowA, int colA, int num_blocks);
#endif