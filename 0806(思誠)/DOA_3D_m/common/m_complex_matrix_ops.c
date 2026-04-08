
//--------------------
#define AVX 16
#define PI acos(-1)      
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "m_complex_matrix_ops.h"
#include "thread_pool.h"
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
#include <sys/syscall.h>
#include <pthread.h>
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//=======================================================
// only fot row block parallel
void m_complex_matrix_multiplication_parallel(float *matA_re, float *matA_im, float *matB_re, float *matB_im, float *matC_re, float *matC_im, int rowA, int rowB, int colB)
{
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    __m512 re_A, re_B, re_C; // simd 256 for matrix real part
    __m512 im_A, im_B, im_C; // simd 256 for matrix Imaginary part
    memset(matC_re, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Real)
    memset(matC_im, 0, rowA * colB * sizeof(float)); // Initial matC = 0 (Imaginary)
    //----------------------------------------------------------
    int full_chunks = rowB / AVX;
    int remainder   = rowB % AVX;
    
    for (int k = 0; k < full_chunks; k++) {
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_loadu_ps(&matA_re[i * rowB + k * AVX]);
            im_A = _mm512_loadu_ps(&matA_im[i * rowB + k * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_loadu_ps(&matB_re[j * rowB + k * AVX]);
                im_B = _mm512_loadu_ps(&matB_im[j * rowB + k * AVX]);
    
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B));
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B));
    
                matC_re[i * colB + j] += _mm512_reduce_add_ps(re_C);
                matC_im[i * colB + j] += _mm512_reduce_add_ps(im_C);
            }
        }
    }
    
    if (remainder > 0) {
        __mmask16 mask = (1 << remainder) - 1;
    
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_maskz_loadu_ps(mask, &matA_re[i * rowB + full_chunks * AVX]);
            im_A = _mm512_maskz_loadu_ps(mask, &matA_im[i * rowB + full_chunks * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_maskz_loadu_ps(mask, &matB_re[j * rowB + full_chunks * AVX]);
                im_B = _mm512_maskz_loadu_ps(mask, &matB_im[j * rowB + full_chunks * AVX]);
    
                re_C = _mm512_sub_ps(_mm512_mul_ps(re_A, re_B), _mm512_mul_ps(im_A, im_B));
                im_C = _mm512_add_ps(_mm512_mul_ps(re_A, im_B), _mm512_mul_ps(im_A, re_B));
    
                matC_re[i * colB + j] += _mm512_reduce_add_ps(re_C);
                matC_im[i * colB + j] += _mm512_reduce_add_ps(im_C);
            }
        }
    }
    // gettimeofday(&end_multiply, NULL);
    //  timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate
    //  printf(L_PURPLE "\nElapsed AVX512 multiply time: %ld(us)\n" CLOSE, (long int)diff_multiply.tv_usec);
    //------------------------------------------------------------
}

void block_multiply_task(void *arg) {
    BlockTaskArg *task = (BlockTaskArg *)arg;

    int offsetA = task->block_index * task->rowB;
    int offsetC = task->block_index * task->colB;

    float *A_block_re = task->A_re + offsetA;
    float *A_block_im = task->A_im + offsetA;
    float *C_block_re = task->C_re + offsetC;
    float *C_block_im = task->C_im + offsetC;

    // printf("Thread %d -> offsetA = %d, block_size = %d, rowB = %d (max A index = %d)\n",
    // get_thread_index(), offsetA, task->block_size, task->rowB, task->rowA * task->rowB);
    // if (offsetA + task->block_size * task->rowB > task->rowA * task->rowB) {
    //     fprintf(stderr, "Error: A_block read out-of-bound!\n");
    //     pthread_exit(NULL);
    // }
    m_complex_matrix_multiplication_parallel(
        A_block_re, A_block_im,
        task->B_re, task->B_im,
        C_block_re, C_block_im,
        task->block_size, task->rowB, task->colB
    );
    
    free(task);
}

void m_complex_matrix_multiplication_row_block_parallel(
    float *matA_re, float *matA_im,
    float *matB_re, float *matB_im,
    float *matC_re, float *matC_im,
    int rowA, int rowB, int colB, int num_blocks)
{
    // Parameter initialize
    struct timeval start_multiply, end_multiply, diff_multiply;    // multiplication variable
    gettimeofday(&start_multiply, NULL);                           // start
    if (num_blocks > rowA)
        num_blocks = rowA;
    int base_block_size = rowA / num_blocks;
    int remainder = rowA % num_blocks;

    matrix_transpose(matB_re, matB_im, rowB, colB); // first transpose B for row block multiplication

    int start_row = 0;

    for (int b = 0; b < num_blocks; b++) {
        BlockTaskArg *arg = malloc(sizeof(BlockTaskArg));
        arg->A_re = matA_re;
        arg->A_im = matA_im;
        arg->B_re = matB_re;
        arg->B_im = matB_im;
        arg->C_re = matC_re;
        arg->C_im = matC_im;
        arg->rowA = rowA;
        arg->rowB = rowB;
        arg->colB = colB;
    
        // 分配每個 block 的實際行數
        if (b == num_blocks - 1)
            arg->block_size = base_block_size + remainder;
        else
            arg->block_size = base_block_size;
    
        arg->block_index = start_row;  // 注意這裡改成 **實際 row 起始值**
    
        addThreadTask(block_multiply_task, arg);
    
        start_row += arg->block_size;  // 累加到下一個 block 起始點
    }
    wait_for_all_tasks();
    matrix_transpose(matB_re, matB_im, colB, rowB);
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply); // calculate total multiply time
    total_multiply_time += diff_multiply.tv_usec;             // global variable can store multiply time
}


void m_complex_matrix_conjugate_transpose_multiplication_row_block_parallel(float *matA_re, float *matA_im, float *matB_re, float *matB_im, int rowA, int colA, int num_blocks)
{
    // float *temp_re = (float *)malloc(colA * rowA * sizeof(float) + AVX*sizeof(float));
    // float *temp_im = (float *)malloc(colA * rowA * sizeof(float) + AVX*sizeof(float));
    float *temp_re = (float *)aligned_alloc(64, (colA * rowA + AVX) * sizeof(float));
    float *temp_im = (float *)aligned_alloc(64, (colA * rowA + AVX) * sizeof(float));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(float)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(float)));
    complex_matrix_conjugate_transpose(temp_re, temp_im, rowA, colA);
    m_complex_matrix_multiplication_row_block_parallel(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA, num_blocks);

    free(temp_re);
    free(temp_im);
}

