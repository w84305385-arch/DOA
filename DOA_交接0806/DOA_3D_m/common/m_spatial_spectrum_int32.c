//--------------------
#define PI acos(-1)
#define AVX 16            
#define M_Antenna 64
#define ND 512
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "complex_matrix_ops_int32.h"
#include "m_complex_matrix_ops_int32.h"
#include "spatial_spectrum.h"
#include "m_complex_matrix_ops.h"
#include "m_spatial_spectrum.h"

#include "m_spatial_spectrum.h"
#include "multi_beam_weight.h"
#include "m_spatial_spectrum_int32.h"
#include "spatial_spectrum_int32.h"
#include "q_format_config.h"
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
#include <sys/syscall.h> // gettid()
#include <pthread.h>
int search_count;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
//----------------------------------------------------------------
// ================================
// ======= spatial spectrum =======
// ================================

void prune_and_search_worker_PAPR_int32(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData_int32* data = (SearchThreadData_int32*)arg; // 轉換指標
    //printf("data->position_theta_prune_prev = %d\n", data->position_theta_prune_prev);
    //printf("Thread ID = %ld, (%.2f, %.2f)\n", temp_index, data->theta_deg_prev, data->phi_deg_prev);
    
    //---------------------------------------------------------------
    // Search angle theta
    int search_len_theta_prune;
    float *search_theta_deg_prune = NULL;
    float *search_theta_rad_prune = NULL;
    calculate_search_theta_3D(
        data->theta_deg_prev, 
        &search_len_theta_prune, 
        &search_theta_deg_prune, 
        &search_theta_rad_prune, 
        data->search_step_theta
    );
    
    // Search angle phi
    int search_len_phi_prune;
    float *search_phi_deg_prune = NULL;
    float *search_phi_rad_prune = NULL;
    calculate_search_phi_3D(
        data->phi_deg_prev, 
        &search_len_phi_prune, 
        &search_phi_deg_prune, 
        &search_phi_rad_prune, 
        data->search_step_phi
    );
    pthread_mutex_lock(&lock);
    search_count += search_len_theta_prune * search_len_phi_prune;
    pthread_mutex_unlock(&lock);
    //---------------------------------------------------------------
    float *S_spatial_spectrum_dB_prune = (float *)malloc(search_len_theta_prune * search_len_phi_prune * sizeof(float) + AVX*sizeof(float));
    if (S_spatial_spectrum_dB_prune == NULL) {
        fprintf(stderr, "Memory allocation failed for S_spatial_spectrum_dB_prune\n");
        return;
    }

    calculate_spatial_spectrum_3D_multiBeam_int32(
        data->search_const_input_data, 
        search_len_theta_prune,
        search_len_phi_prune,
        search_theta_rad_prune,
        search_phi_rad_prune,
        S_spatial_spectrum_dB_prune,
        data->search_step_theta
    );

    //printf(" before data->position_theta_prune %d\n", data->position_theta_prune);
    // find peaks
    //pthread_mutex_lock(&lock);
    m_find_spatial_spectrum_peaks_3D_PAPR(
        S_spatial_spectrum_dB_prune, 
        search_len_theta_prune, 
        search_len_phi_prune, 
        search_theta_deg_prune, 
        search_phi_deg_prune,
        data->PAPR_threshold,
        // output
        data->result_theta_deg,   
        data->result_phi_deg,     
        data->result_dB_value,    
        &data->detected_peak_count
    );
    //pthread_mutex_unlock(&lock);
    //printf("data->position_theta_prune %d\n", data->position_theta_prune[0]);
    free(search_theta_deg_prune);
    free(search_theta_rad_prune);
    free(search_phi_deg_prune);
    free(search_phi_rad_prune);
    free(S_spatial_spectrum_dB_prune);
}

void prune_and_search_worker_PAPR_MVDR_int32(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData_int32* data = (SearchThreadData_int32*)arg; // 轉換指標
    //printf("data->position_theta_prune_prev = %d\n", data->position_theta_prune_prev);
    //printf("Thread ID = %ld, (%.2f, %.2f)\n", temp_index, data->theta_deg_prev, data->phi_deg_prev);
    
    //---------------------------------------------------------------
    // Search angle theta
    int search_len_theta_prune;
    float *search_theta_deg_prune = NULL;
    float *search_theta_rad_prune = NULL;
    calculate_search_theta_3D(
        data->theta_deg_prev, 
        &search_len_theta_prune, 
        &search_theta_deg_prune, 
        &search_theta_rad_prune, 
        data->search_step_theta
    );
    
    // Search angle phi
    int search_len_phi_prune;
    float *search_phi_deg_prune = NULL;
    float *search_phi_rad_prune = NULL;
    calculate_search_phi_3D(
        data->phi_deg_prev, 
        &search_len_phi_prune, 
        &search_phi_deg_prune, 
        &search_phi_rad_prune, 
        data->search_step_phi
    );
    pthread_mutex_lock(&lock);
    search_count += search_len_theta_prune * search_len_phi_prune;
    pthread_mutex_unlock(&lock);
    //---------------------------------------------------------------
    float *S_spatial_spectrum_dB_prune = (float *)malloc(search_len_theta_prune * search_len_phi_prune * sizeof(float) + AVX*sizeof(float));
    if (S_spatial_spectrum_dB_prune == NULL) {
        fprintf(stderr, "Memory allocation failed for S_spatial_spectrum_dB_prune\n");
        return;
    }

    calculate_spatial_spectrum_3D_int32(
        data->search_const_input_data, 
        search_len_theta_prune,
        search_len_phi_prune,
        search_theta_rad_prune,
        search_phi_rad_prune,
        S_spatial_spectrum_dB_prune
    );

    //printf(" before data->position_theta_prune %d\n", data->position_theta_prune);
    // find peaks
    //pthread_mutex_lock(&lock);
    m_find_spatial_spectrum_peaks_3D_PAPR(
        S_spatial_spectrum_dB_prune, 
        search_len_theta_prune, 
        search_len_phi_prune, 
        search_theta_deg_prune, 
        search_phi_deg_prune,
        data->PAPR_threshold,
        // output
        data->result_theta_deg,   
        data->result_phi_deg,     
        data->result_dB_value,    
        &data->detected_peak_count
    );
    //pthread_mutex_unlock(&lock);
    //printf("data->position_theta_prune %d\n", data->position_theta_prune[0]);
    free(search_theta_deg_prune);
    free(search_theta_rad_prune);
    free(search_phi_deg_prune);
    free(search_phi_rad_prune);
    free(S_spatial_spectrum_dB_prune);
}


void prune_and_search_worker_PAPR_MVDR_sub_int32(void *arg)
{
    /* 轉成擴充版指標 */
    SearchThreadData_MVDR_int32 *ex  = (SearchThreadData_MVDR_int32 *)arg;
    SearchThreadData_int32      *td  = &ex->base;          // 只是方便沿用舊變數名

    /* 1. 產生 θ, φ 搜尋格點 ---------------------- */
    int len_theta, len_phi;
    float *theta_deg, *theta_rad, *phi_deg, *phi_rad;

    calculate_search_theta_3D(td->theta_deg_prev,
                              &len_theta,&theta_deg,&theta_rad,
                              td->search_step_theta);

    calculate_search_phi_3D  (td->phi_deg_prev,
                              &len_phi,  &phi_deg, &phi_rad,
                              td->search_step_phi);

    size_t n_grid = (size_t)len_theta * len_phi;
    float *S_dB   = aligned_alloc(64, n_grid * sizeof(float));
    memset(S_dB, 0, n_grid * sizeof(float));

    /* 2. 空間譜計算 ------------------------------ */
    if (!ex->use_subarray) {
        calculate_spatial_spectrum_3D_int32(
            td->search_const_input_data,
            len_theta, len_phi,
            theta_rad,  phi_rad,
            S_dB);
    } else {
        for (int sid = 0; sid < ex->N_SUBARRAY; ++sid) {
            ex->search_const_sub_tpl->Pn_re = ex->Pn_re_sub[sid];
            ex->search_const_sub_tpl->Pn_im = ex->Pn_im_sub[sid];

            float *S_tmp = aligned_alloc(64, n_grid * sizeof(float));

            calculate_spatial_spectrum_3D_int32(
                ex->search_const_sub_tpl,
                len_theta, len_phi,
                theta_rad,  phi_rad,
                S_tmp);

            for (size_t k = 0; k < n_grid; ++k)
                S_dB[k] += S_tmp[k];   /* 亦可改成 fmaxf() 取最大 */

            free(S_tmp);
        }
        for (int k = 0; k < len_theta*len_phi; ++k)
            S_dB[k] = S_dB[k]/ex->N_SUBARRAY;
    }
    
    /* 3. Peak picking --------------------------- */
    m_find_spatial_spectrum_peaks_3D_PAPR(
        S_dB,
        len_theta, len_phi,
        theta_deg, phi_deg,
        td->PAPR_threshold,
        td->result_theta_deg,
        td->result_phi_deg,
        td->result_dB_value,
        &td->detected_peak_count);

    /* 4. 清理 ---------------------------------- */
    free(theta_deg);  free(theta_rad);
    free(phi_deg);    free(phi_rad);
    free(S_dB);
}
void prune_and_search_worker_max_int32(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData_int32* data = (SearchThreadData_int32*)arg; // 轉換指標
    //printf("data->position_theta_prune_prev = %d\n", data->position_theta_prune_prev);
    //printf("Thread ID = %ld, (%.2f, %.2f)\n", temp_index, data->theta_deg_prev, data->phi_deg_prev);
    
    //---------------------------------------------------------------
    // Search angle theta
    int search_len_theta_prune;
    float *search_theta_deg_prune = NULL;
    float *search_theta_rad_prune = NULL;
    calculate_search_theta_3D(
        data->theta_deg_prev, 
        &search_len_theta_prune, 
        &search_theta_deg_prune, 
        &search_theta_rad_prune, 
        data->search_step_theta
    );


    // Search angle phi
    int search_len_phi_prune;
    float *search_phi_deg_prune = NULL;
    float *search_phi_rad_prune = NULL;
    calculate_search_phi_3D(
        data->phi_deg_prev, 
        &search_len_phi_prune, 
        &search_phi_deg_prune, 
        &search_phi_rad_prune, 
        data->search_step_phi
    );

    pthread_mutex_lock(&lock);
    search_count += search_len_theta_prune * search_len_phi_prune;
    pthread_mutex_unlock(&lock);
    //---------------------------------------------------------------
    float *S_spatial_spectrum_dB_prune = (float *)malloc(search_len_theta_prune * search_len_phi_prune * sizeof(float)+ AVX*sizeof(float));
    if (S_spatial_spectrum_dB_prune == NULL) {
        fprintf(stderr, "Memory allocation failed for S_spatial_spectrum_dB_prune\n");
        return;
    }

    // calculate Spatial Spectrum
    // 可能有問題
    calculate_spatial_spectrum_3D_int32(
        data->search_const_input_data, 
        search_len_theta_prune,
        search_len_phi_prune,
        search_theta_rad_prune,
        search_phi_rad_prune,
        S_spatial_spectrum_dB_prune
    );
    //printf(" before data->position_theta_prune %d\n", data->position_theta_prune);
    // find peaks
    //pthread_mutex_lock(&lock);
    //printf("prev(%.2f, %.2f)\n", data->theta_deg_prev, data->phi_deg_prev);
    m_find_spatial_spectrum_peaks_3D_max(
        S_spatial_spectrum_dB_prune, 
        search_len_theta_prune, 
        search_len_phi_prune, 
        search_theta_deg_prune, 
        search_phi_deg_prune,
        // output
        data->result_theta_deg,   
        data->result_phi_deg,     
        data->result_dB_value,    
        &data->detected_peak_count
    );

    free(search_theta_deg_prune);
    free(search_theta_rad_prune);
    free(search_phi_deg_prune);
    free(search_phi_rad_prune);
    free(S_spatial_spectrum_dB_prune);
    //pthread_mutex_unlock(&lock);
    //printf("data->position_theta_prune %d\n", data->position_theta_prune[0]);
}




// =================== Spatial Spectrum multiplication row block parallel ======================

void m_compute_spatial_spectrum_value_row_block_parallel_int32(int32_t *a_vector_re, int32_t *a_vector_im, int32_t *Pn_re, int32_t *Pn_im, int M, float *result_re, float *result_im)
{
    struct timeval time_spatial_spectrum_value_start, time_spatial_spectrum_value_end, time_spatial_spectrum_value_diff; // time initial
    struct timeval time_complex_div_start, time_complex_div_end, time_complex_div_diff; // time initial
    //---------------------------------------------------------------
    int32_t *Pn_a_vector_temp_re = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *Pn_a_vector_temp_im = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *compute_spatial_spectrum_value_temp_re = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *compute_spatial_spectrum_value_temp_im = (int32_t *)malloc(M * sizeof(int32_t));
    float *compute_spatial_spectrum_value_temp_re_float = (float *)malloc(sizeof(float));
    float *compute_spatial_spectrum_value_temp_im_float = (float *)malloc(sizeof(float));
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    // gettimeofday(&time_spatial_spectrum_value_start, NULL);
    //---------------------------------------------------------------
    m_complex_matrix_multiplication_int32_row_block_parallel(
        Pn_re, Pn_im, a_vector_re, a_vector_im, 
        Pn_a_vector_temp_re, Pn_a_vector_temp_im, 
        M, M, 1, num_row_blocks);

    // printf("Pn_a_vector_temp\n");
    // print_complex_matrix_int32(Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, 1);

    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_int32(a_vector_re, a_vector_im, M, 1);
    // printf("a_vector_temp\n");
    // print_complex_matrix_int32(a_vector_re, a_vector_im, M, 1);

    //---------------------------------------------------------------
    complex_matrix_multiplication_int32(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, M, 1);
    // printf("compute_spatial_spectrum_value_temp\n");
    // print_complex_matrix_int32(compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, 1);

    //---------------------------------------------------------------
    // gettimeofday(&time_spatial_spectrum_value_end, NULL);
    // timersub(&time_spatial_spectrum_value_end, &time_spatial_spectrum_value_start, &time_spatial_spectrum_value_diff);
    // time_spatial_spectrum_value += time_spatial_spectrum_value_diff.tv_sec * 1000000 + time_spatial_spectrum_value_diff.tv_usec;
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    //---------------------------------------------------------------
    // int32_t to float
    // compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0] / (float)(1 << Q_SHIFT);
    // compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0] / (float)(1 << Q_SHIFT);
    compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0] / Q_SCALE;
    compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0] / Q_SCALE;
    // compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0];
    // compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0];
    // printf("compute_spatial_spectrum_value_temp_float = (%f,%f)\n", compute_spatial_spectrum_value_temp_re_float[0], compute_spatial_spectrum_value_temp_im_float[0]);

    //---------------------------------------------------------------
    
    // gettimeofday(&time_complex_div_start, NULL);

    cpp_division2(1, 0, &compute_spatial_spectrum_value_temp_re_float[0], &compute_spatial_spectrum_value_temp_im_float[0], result_re, result_im);
    
    // gettimeofday(&time_complex_div_end, NULL);
    // timersub(&time_complex_div_end, &time_complex_div_start, &time_complex_div_diff);
    // time_complex_div += time_complex_div_diff.tv_sec * 1000000 + time_complex_div_diff.tv_usec;
    // printf("music = (%.5f,%.5f)\n", *result_re, *result_re);
    free(Pn_a_vector_temp_re);
    free(Pn_a_vector_temp_im);
    free(compute_spatial_spectrum_value_temp_re);
    free(compute_spatial_spectrum_value_temp_im);
    free(compute_spatial_spectrum_value_temp_re_float);
    free(compute_spatial_spectrum_value_temp_im_float);
}

const int m_rx_m_index_int32[] = {64,256,576};
const int m_beam_deg_index_int32[] = {30, 10};
void m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB,
    float search_step_theta 
)
{
    // struct timeval time_cpp_exp2_to_int_start, time_cpp_exp2_to_int_end, time_cpp_exp2_to_int_diff; // time initial
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // ---------------------------------------------------------------
    // Select different weights based on Rx_M and search_step_theta
    int shift=4;
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    int rx_m_idx = find_index(m_rx_m_index_int32, 4, Rx_M);
    int theta_idx = find_index(m_beam_deg_index_int32, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || theta_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Configuration not found\n");
    } else if ((Rx_M == 64 || Rx_M == 256) && theta_idx == 1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        shift = 1;
        printf("Force use 1.0 beam for Rx_M = %d, theta = 10\n", Rx_M);
    } else {
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][theta_idx], Rx_M * sizeof(float));
        printf("Find the configuration rx_m_idx = %d, theta_idx = %d\n", rx_m_idx, theta_idx);
    }
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    int32_t *a_vector_re_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *a_vector_im_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *S_spatial_spectrum_re_int32 = (int32_t *)malloc(search_len_theta*search_len_phi * sizeof(int32_t) + AVX * sizeof(int32_t));
    int32_t *S_spatial_spectrum_im_int32 = (int32_t *)malloc(search_len_theta*search_len_phi * sizeof(int32_t) + AVX * sizeof(int32_t));
    float *S_spatial_spectrum_re = (float *)malloc(search_len_theta*search_len_phi * sizeof(float) + AVX * sizeof(float));
    float *S_spatial_spectrum_im = (float *)malloc(search_len_theta*search_len_phi * sizeof(float) + AVX * sizeof(float));
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M_x; ++j){ // steering vector
                for (int k = 0; k < Rx_M_y; ++k){ // steering vector
                    cpp_exp2_3D(&a_vector_re[j*Rx_M_y+k], &a_vector_im[j*Rx_M_y+k], search_theta_rad, search_phi_rad, d, kc, elevation, azimuth, j - Rx_M_x_delta, k - Rx_M_y_delta);
                    // printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                    a_vector_re[j*Rx_M_y+k] = a_vector_re[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k]*shift;
                    a_vector_im[j*Rx_M_y+k] = a_vector_im[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k]*shift;
                    // printf("a_vector_multi(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                }
            }
            // --------------------------------------
            // gettimeofday(&time_cpp_exp2_to_int_start, NULL);
            // float_matrix_to_q_format(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M, 1, q_shift_DOA);
            float_to_q_int32_avx512(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M);
            // gettimeofday(&time_cpp_exp2_to_int_end, NULL);
            // timersub(&time_cpp_exp2_to_int_end, &time_cpp_exp2_to_int_start, &time_cpp_exp2_to_int_diff);
            // time_cpp_exp2_to_int += time_cpp_exp2_to_int_diff.tv_sec * 1000000 + time_cpp_exp2_to_int_diff.tv_usec;

            // --------------------------------------
            // spatial_spectrum_value_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re_int32[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im_int32[elevation*search_len_phi + azimuth]);
            m_compute_spatial_spectrum_value_row_block_parallel_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);

            // printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    // q_int32_to_float_avx512(S_spatial_spectrum_re, S_spatial_spectrum_im, S_spatial_spectrum_re_int32, S_spatial_spectrum_im_int32, search_len_theta*search_len_phi);
    
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
        }
    }


    // printf("time_cpp_exp2_to_int = %.3f(us)\n", time_cpp_exp2_to_int);
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    // printf("time_spatial_spectrum_value_avg = %.3f(us)\n", time_spatial_spectrum_value/(search_len_theta*search_len_phi));
    // printf("time_complex_div = %.3f(us)\n", time_complex_div);
    
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(a_vector_re_int32);free(a_vector_im_int32);
    free(S_spatial_spectrum_re_int32);free(S_spatial_spectrum_im_int32);
}

void m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB
)
{
    struct timeval time_cpp_exp2_to_int_start, time_cpp_exp2_to_int_end, time_cpp_exp2_to_int_diff; // time initial


    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    int32_t *a_vector_re_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *a_vector_im_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    float *S_spatial_spectrum_re = (float *)malloc(search_len_theta*search_len_phi * sizeof(float) + AVX * sizeof(float));
    float *S_spatial_spectrum_im = (float *)malloc(search_len_theta*search_len_phi * sizeof(float) + AVX * sizeof(float));
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M_x; ++j){ // steering vector
                for (int k = 0; k < Rx_M_y; ++k){ // steering vector
                    cpp_exp2_3D(&a_vector_re[j*Rx_M_y+k], &a_vector_im[j*Rx_M_y+k], search_theta_rad, search_phi_rad, d, kc, elevation, azimuth, j - Rx_M_x_delta, k - Rx_M_y_delta);
                    //printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                }
            }
            // --------------------------------------
            // gettimeofday(&time_cpp_exp2_to_int_start, NULL);
            float_to_q_int32_avx512(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M);
            // gettimeofday(&time_cpp_exp2_to_int_end, NULL);
            // timersub(&time_cpp_exp2_to_int_end, &time_cpp_exp2_to_int_start, &time_cpp_exp2_to_int_diff);
            // time_cpp_exp2_to_int += time_cpp_exp2_to_int_diff.tv_sec * 1000000 + time_cpp_exp2_to_int_diff.tv_usec;

            // --------------------------------------
            m_compute_spatial_spectrum_value_row_block_parallel_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            // printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    // q_int32_to_float_avx512(S_spatial_spectrum_re, S_spatial_spectrum_im, S_spatial_spectrum_re_int32, S_spatial_spectrum_im_int32, search_len_theta*search_len_phi);
    
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            S_spatial_spectrum_im[elevation*search_len_phi + azimuth] = 0.0;
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
        }
    }


    // printf("time_cpp_exp2_to_int = %.3f(us)\n", time_cpp_exp2_to_int);
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    // printf("time_spatial_spectrum_value_avg = %.3f(us)\n", time_spatial_spectrum_value/(search_len_theta*search_len_phi));
    // printf("time_complex_div = %.3f(us)\n", time_complex_div);
    
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(a_vector_re_int32);free(a_vector_im_int32);

}
// =============================================================================================

// free Search Data
void free_SearchThreadData_int32(SearchThreadData_int32 *td) {
    if (!td) return;
    free(td->result_theta_deg);
    free(td->result_phi_deg);
    free(td->result_dB_value);
    free(td);
}