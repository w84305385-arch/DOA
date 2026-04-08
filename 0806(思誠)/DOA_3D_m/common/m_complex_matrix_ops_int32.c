
//--------------------
#define AVX 16
#define PI acos(-1)      
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "complex_matrix_ops_int32.h"
#include "m_complex_matrix_ops.h"
#include "m_complex_matrix_ops_int32.h"
#include "thread_pool.h"
#include "complex_matrix_ops_int32.h"
#include "q_format_config.h"

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
// static int64_t acc_re[1000000][8] __attribute__((aligned(64)));
// static int64_t acc_im[1000000][8] __attribute__((aligned(64)));
void m_complex_matrix_multiplication_int32_parallel(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int32_t *matC_re, int32_t *matC_im, int rowA, int rowB, int colB)
{
    // int thread_id = get_thread_index();
    // /* ---------- (a) 用 int64_t 暫存 C ----------- */
    // memset(acc_re[thread_id], 0, rowA * colB * sizeof(int64_t)); // Initial acc_re = 0
    // memset(acc_im[thread_id], 0, rowA * colB * sizeof(int64_t)); // Initial acc_im = 0
    int64_t* acc_re = aligned_alloc(64, rowA * colB * sizeof(int64_t));
    int64_t* acc_im = aligned_alloc(64, rowA * colB * sizeof(int64_t));
    memset(acc_re, 0, rowA * colB * sizeof(int64_t)); // Initial acc_re = 0
    memset(acc_im, 0, rowA * colB * sizeof(int64_t)); // Initial acc_im = 0
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------

    //-------------------------------------------------------------------
    // Initialize Global variable array = 0
    //----------------------------------------------------------
    __m512i re_A, re_B, re_C; 
    __m512i im_A, im_B, im_C;
    // memset(matC_re, 0, rowA * colB * sizeof(int32_t)); // Initial matC = 0 (Real)
    // memset(matC_im, 0, rowA * colB * sizeof(int32_t)); // Initial matC = 0 (Imaginary)
    //----------------------------------------------------------
    int full_chunks = rowB / AVX;
    int remainder   = rowB % AVX;
    
    for (int k = 0; k < full_chunks; k++) {
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_loadu_si512(&matA_re[i * rowB + k * AVX]);
            im_A = _mm512_loadu_si512(&matA_im[i * rowB + k * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_loadu_si512(&matB_re[j * rowB + k * AVX]);
                im_B = _mm512_loadu_si512(&matB_im[j * rowB + k * AVX]);
    
                re_C = _mm512_sub_epi32(_mm512_mullo_epi32(re_A, re_B), _mm512_mullo_epi32(im_A, im_B));
                im_C = _mm512_add_epi32(_mm512_mullo_epi32(re_A, im_B), _mm512_mullo_epi32(im_A, re_B));
                int idx = i * colB + j;
                // acc_re[thread_id][idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                // acc_im[thread_id][idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
                acc_re[idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                acc_im[idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
            }
        }
    }
    __mmask16 mask;
    if (remainder > 0) {
        mask = (1 << remainder) - 1;
    
        for (int i = 0; i < rowA; i++) {
            re_A = _mm512_maskz_loadu_epi32(mask, &matA_re[i * rowB + full_chunks * AVX]);
            im_A = _mm512_maskz_loadu_epi32(mask, &matA_im[i * rowB + full_chunks * AVX]);
    
            for (int j = 0; j < colB; j++) {
                re_B = _mm512_maskz_loadu_epi32(mask, &matB_re[j * rowB + full_chunks * AVX]);
                im_B = _mm512_maskz_loadu_epi32(mask, &matB_im[j * rowB + full_chunks * AVX]);
    
                re_C = _mm512_sub_epi32(_mm512_mullo_epi32(re_A, re_B), _mm512_mullo_epi32(im_A, im_B));
                im_C = _mm512_add_epi32(_mm512_mullo_epi32(re_A, im_B), _mm512_mullo_epi32(im_A, re_B));
    
                int idx = i * colB + j;
                // acc_re[thread_id][idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                // acc_im[thread_id][idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
                acc_re[idx] += (int64_t)_mm512_reduce_add_epi32(re_C);
                acc_im[idx] += (int64_t)_mm512_reduce_add_epi32(im_C);
            }
        }
    }
    //-------------------------------------------------------------------
    /* ---------- (c) 一次性右移回 int32_t ---------- */
    for (int i = 0; i < rowA * colB; ++i) {
        /* 可加入飽和或四捨五入：((acc + (1LL<<(Q_EXTRA-1))) >> Q_EXTRA) */
        //printf("acc_re[%d] = %ld, acc_im[%d] = %ld\n", i, acc_re[i], i, acc_im[i]);
        // matC_re[i] = (int32_t)(acc_re[thread_id][i] >> Q_SHIFT);
        // matC_im[i] = (int32_t)(acc_im[thread_id][i] >> Q_SHIFT);
        matC_re[i] = (int32_t)(acc_re[i] >> Q_SHIFT);
        matC_im[i] = (int32_t)(acc_im[i] >> Q_SHIFT);
        // printf("matC_re[%d] = %d, matC_im[%d] = %d\n", i, matC_re[i], i, matC_im[i]);
    }

    
    free(acc_re);  free(acc_im);
}

void block_multiply_task_int32(void *arg) {
    BlockTaskArg_int32 *task = (BlockTaskArg_int32 *)arg;

    int offsetA = task->block_index * task->rowB;
    int offsetC = task->block_index * task->colB;

    int32_t *A_block_re = task->A_re + offsetA;
    int32_t *A_block_im = task->A_im + offsetA;
    int32_t *C_block_re = task->C_re + offsetC;
    int32_t *C_block_im = task->C_im + offsetC;

    // printf("Thread %d -> offsetA = %d, block_size = %d, rowB = %d (max A index = %d)\n",
    // get_thread_index(), offsetA, task->block_size, task->rowB, task->rowA * task->rowB);
    // if (offsetA + task->block_size * task->rowB > task->rowA * task->rowB) {
    //     fprintf(stderr, "Error: A_block read out-of-bound!\n");
    //     pthread_exit(NULL);
    // }
    m_complex_matrix_multiplication_int32_parallel(
        A_block_re, A_block_im,
        task->B_re, task->B_im,
        C_block_re, C_block_im,
        task->block_size, task->rowB, task->colB
    );
    
    free(task);
}

void m_complex_matrix_multiplication_int32_row_block_parallel(
    int32_t *matA_re, int32_t *matA_im,
    int32_t *matB_re, int32_t *matB_im,
    int32_t *matC_re, int32_t *matC_im,
    int rowA, int rowB, int colB, int num_blocks)
{
    if (num_blocks > rowA)
        num_blocks = rowA;
    int base_block_size = rowA / num_blocks;
    int remainder = rowA % num_blocks;

    matrix_transpose_int32(matB_re, matB_im, rowB, colB); // first transpose B for row block multiplication

    int start_row = 0;

    for (int b = 0; b < num_blocks; b++) {
        BlockTaskArg_int32 *arg = malloc(sizeof(BlockTaskArg_int32));
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
    
        addThreadTask(block_multiply_task_int32, arg);
    
        start_row += arg->block_size;  // 累加到下一個 block 起始點
    }
    wait_for_all_tasks();
    matrix_transpose_int32(matB_re, matB_im, colB, rowB);
}


void m_complex_matrix_conjugate_transpose_multiplication_int32_row_block_parallel(int32_t *matA_re, int32_t *matA_im, int32_t *matB_re, int32_t *matB_im, int rowA, int colA, int num_blocks)
{

    int32_t *temp_re = (int32_t *)aligned_alloc(64, (colA * rowA + AVX) * sizeof(int32_t));
    int32_t *temp_im = (int32_t *)aligned_alloc(64, (colA * rowA + AVX) * sizeof(int32_t));
    memcpy(temp_re, matA_re, (rowA * colA * sizeof(int32_t)));
    memcpy(temp_im, matA_im, (rowA * colA * sizeof(int32_t)));
    complex_matrix_conjugate_transpose_int32(temp_re, temp_im, rowA, colA);
    m_complex_matrix_multiplication_int32_row_block_parallel(matA_re, matA_im, temp_re, temp_im, matB_re, matB_im, rowA, colA, rowA, num_blocks);

    free(temp_re);
    free(temp_im);
}

