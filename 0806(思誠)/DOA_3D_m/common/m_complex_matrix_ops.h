#ifndef M_COMPLEX_MATRIX_OPS_H
#define M_COMPLEX_MATRIX_OPS_H
typedef struct {
    float *A_re, *A_im;
    float *B_re, *B_im;
    float *C_re, *C_im;
    int    rowA;    
    int    rowB;  
    int    colB;  
    int    block_index;
    int    block_size; 
    int    col_start; 
} BlockTaskArg;
// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;

void m_complex_matrix_multiplication_parallel(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB);
void block_multiply_task(void *arg);
void m_complex_matrix_multiplication_row_block_parallel(
    float *matA_re, float *matA_im,
    float *matB_re, float *matB_im,
    float *matC_re, float *matC_im,
    int rowA, int rowB, int colB, int block_size);

void m_complex_matrix_conjugate_transpose_multiplication_row_block_parallel(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA, int num_blocks);
#endif