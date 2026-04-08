//--------------------
#define PI acos(-1)
#define AVX 16            
#define M_Antenna 64
#define ND 512
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "spatial_spectrum.h"
#include "m_complex_matrix_ops.h"
#include "m_spatial_spectrum.h"
#include "multi_beam_weight.h"
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

void prune_and_search_worker_PAPR(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData* data = (SearchThreadData*)arg; // 轉換指標
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

    calculate_spatial_spectrum_3D_multiBeam(
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

void prune_and_search_worker_PAPR_MVDR(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData* data = (SearchThreadData*)arg; // 轉換指標
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

    calculate_spatial_spectrum_3D(
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


void prune_and_search_worker_PAPR_MVDR_sub(void *arg)
{
    /* 轉成擴充版指標 */
    SearchThreadData_MVDR *ex  = (SearchThreadData_MVDR *)arg;
    SearchThreadData      *td  = &ex->base;          // 只是方便沿用舊變數名

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
        calculate_spatial_spectrum_3D(
            td->search_const_input_data,
            len_theta, len_phi,
            theta_rad,  phi_rad,
            S_dB);
    } else {
        for (int sid = 0; sid < ex->N_SUBARRAY; ++sid) {
            ex->search_const_sub_tpl->Pn_re = ex->Pn_re_sub[sid];
            ex->search_const_sub_tpl->Pn_im = ex->Pn_im_sub[sid];

            float *S_tmp = aligned_alloc(64, n_grid * sizeof(float));

            calculate_spatial_spectrum_3D(
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
void prune_and_search_worker_max(void* arg){
    //-------------------------------------------------------------------
    // Thread index
    int temp_index = get_thread_index();
    SearchThreadData* data = (SearchThreadData*)arg; // 轉換指標
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
    m_calculate_spatial_spectrum_3D(
        data->search_const_input_data, 
        search_len_theta_prune,
        search_len_phi_prune,
        search_theta_rad_prune,
        search_phi_rad_prune,
        S_spatial_spectrum_dB_prune
    );
    //printf(" before data->position_theta_prune %d\n", data->position_theta_prune);
    // find peaks
    // pthread_mutex_lock(&lock);
    // printf("prev(%.2f, %.2f)\n", data->theta_deg_prev, data->phi_deg_prev);
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
    // pthread_mutex_unlock(&lock);
    // printf("data->position_theta_prune %d\n", data->position_theta_prune[0]);
}

void m_calculate_spatial_spectrum_3D_multiBeam(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float *spatial_spectrum_value_dB, 
    float search_step_theta
){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // ---------------------------------------------------------------
    // Select different weights based on Rx_M and search_step_theta
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    int rx_m_idx = find_index(rx_m_index, 4, Rx_M);
    int beam_deg_idx = find_index(beam_deg_index, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || beam_deg_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Configuration not found\n");
    } else if ((Rx_M == 64 || Rx_M == 256) && beam_deg_idx == 1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Force use 1.0 beam for Rx_M = %d, theta = 10\n", Rx_M);
    } else {
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][beam_deg_idx], Rx_M * sizeof(float));
        printf("Find the configuration rx_m_idx = %d, beam_deg_idx = %d\n", rx_m_idx, beam_deg_idx);
    }
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
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
                    a_vector_re[j*Rx_M_y+k] = a_vector_re[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k];
                    a_vector_im[j*Rx_M_y+k] = a_vector_im[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k];
                }
            }
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(w_multi_beam_re);
}

void m_calculate_spatial_spectrum_3D(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,  
    float *spatial_spectrum_value_dB
){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX*sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float)+ AVX*sizeof(float));
    float *S_spatial_spectrum_re = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX*sizeof(float));
    float *S_spatial_spectrum_im = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX*sizeof(float));
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute compute_spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M_x; ++j){ // steering vector
                for (int k = 0; k < Rx_M_y; ++k){ // steering vector
                    cpp_exp2_3D(&a_vector_re[j*Rx_M_y+k], &a_vector_im[j*Rx_M_y+k], search_theta_rad, search_phi_rad, d, kc, elevation, azimuth, j - Rx_M_x_delta, k - Rx_M_y_delta);
                    //printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                }
            }
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("spatial_spectrum_value_dB[%d] = %.4f\n", elevation*search_len_phi + azimuth, spatial_spectrum_value_dB[elevation*search_len_phi + azimuth]);
        }
    }
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
}




// =================== Spatial Spectrum multiplication row block parallel ======================
void m_compute_spatial_spectrum_value_row_block_parallel(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *music_Real, float *music_Imag)
{
    //---------------------------------------------------------------
    float *Pn_a_vector_temp_re = (float *)malloc(M * sizeof(float)+ AVX*sizeof(float));
    float *Pn_a_vector_temp_im = (float *)malloc(M * sizeof(float)+ AVX*sizeof(float));
    float *compute_spatial_spectrum_value_temp_re = (float *)malloc(M * sizeof(float)+ AVX*sizeof(float));
    float *compute_spatial_spectrum_value_temp_im = (float *)malloc(M * sizeof(float)+ AVX*sizeof(float));
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    m_complex_matrix_multiplication_row_block_parallel(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1, num_row_blocks);
    complex_matrix_conjugate_transpose(a_vector_re, a_vector_im, M, 1);
    //m_complex_matrix_multiplication_blockwise_parallel_col(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, M, 1, 8);
    complex_matrix_multiplication(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, M, 1);
    cpp_division2(1, 0, &compute_spatial_spectrum_value_temp_re[0], &compute_spatial_spectrum_value_temp_im[0], music_Real, music_Imag);

    //printf("music = (%.5f,%.5f)\n", *music_Real, *music_Real);
    free(Pn_a_vector_temp_re);
    free(Pn_a_vector_temp_im);
    free(compute_spatial_spectrum_value_temp_re);
    free(compute_spatial_spectrum_value_temp_im);
}

void m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float *spatial_spectrum_value_dB, 
    float search_step_theta
){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // ---------------------------------------------------------------
    // Select different weights based on Rx_M and search_step_theta
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    int rx_m_idx = find_index(rx_m_index, 4, Rx_M);
    int beam_deg_idx = find_index(beam_deg_index, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || beam_deg_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Configuration not found\n");
    } else if ((Rx_M == 64 || Rx_M == 256) && beam_deg_idx == 1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Force use 1.0 beam for Rx_M = %d, theta = 10\n", Rx_M);
    } else {
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][beam_deg_idx], Rx_M * sizeof(float));
        printf("Find the configuration rx_m_idx = %d, beam_deg_idx = %d\n", rx_m_idx, beam_deg_idx);
    }
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
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
                    a_vector_re[j*Rx_M_y+k] = a_vector_re[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k];
                    a_vector_im[j*Rx_M_y+k] = a_vector_im[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k];
                }
            }
            m_compute_spatial_spectrum_value_row_block_parallel(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            // printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[elevation*search_len_phi + azimuth]);
        }
    }
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(w_multi_beam_re);
}

void m_calculate_spatial_spectrum_3D_row_block_parallel(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,  
    float *spatial_spectrum_value_dB
){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX*sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float)+ AVX*sizeof(float));
    float *S_spatial_spectrum_re = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX*sizeof(float));
    float *S_spatial_spectrum_im = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX*sizeof(float));
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute compute_spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M_x; ++j){ // steering vector
                for (int k = 0; k < Rx_M_y; ++k){ // steering vector
                    cpp_exp2_3D(&a_vector_re[j*Rx_M_y+k], &a_vector_im[j*Rx_M_y+k], search_theta_rad, search_phi_rad, d, kc, elevation, azimuth, j - Rx_M_x_delta, k - Rx_M_y_delta);
                    //printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                }
            }
            m_compute_spatial_spectrum_value_row_block_parallel(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("spatial_spectrum_value_dB[%d] = %.4f\n", elevation*search_len_phi + azimuth, spatial_spectrum_value_dB[elevation*search_len_phi + azimuth]);
        }
    }
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
}
// =============================================================================================


void m_find_spatial_spectrum_peaks_3D_PAPR(
    float *spatial_spectrum_value_dB,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_deg,
    float *search_phi_deg,
    float PAPR_threshold,
    // output
    float *result_theta_deg,
    float *result_phi_deg,
    float *result_dB_value,
    int *detected_peak_count
) {
    //printf("== Using PAPR thresholding ==\n");

    // ---------------------------------------------
    // Find background power
    int total_points = search_len_theta * search_len_phi;
    float *sorted_values = (float *)malloc(sizeof(float) * total_points);
    for (int i = 0; i < total_points; ++i) {
        sorted_values[i] = spatial_spectrum_value_dB[i];
    }

    // Bubble Sort Incremental
    for (int i = 0; i < total_points - 1; ++i) {
        for (int j = 0; j < total_points - 1 - i; ++j) {
            if (sorted_values[j] > sorted_values[j + 1]) {
                float temp = sorted_values[j];
                sorted_values[j] = sorted_values[j + 1];
                sorted_values[j + 1] = temp;
            }
        }
    }

    // Take the first 10% as the background power
    int bg_count = total_points * 0.1;
    if (bg_count < 1) bg_count = 1;
    float sum_bg = 0.0f;
    for (int i = 0; i < bg_count; ++i) {
        sum_bg += sorted_values[i];
    }
    free(sorted_values);
    float avg_bg_power = sum_bg / bg_count;
    float threshold = avg_bg_power + PAPR_threshold;

    //printf("Background average power = %.3f dB\n", avg_bg_power);
    //printf("PAPR_threshold = %.3f dB\n", PAPR_threshold);
    //printf("Threshold (PAPR + PAPR_threshold dB) = %.3f dB\n", threshold);
    
    // peak selection
    int index = 0;
    for (int i = 0; i < search_len_theta; ++i) {
        for (int j = 0; j < search_len_phi; ++j) {
            float val = spatial_spectrum_value_dB[i * search_len_phi + j];
            //printf("search_theta_deg[%d] = %.2f, search_phi_deg[%d] = %.2f, val = %.2f\n",
            //       i, search_theta_deg[i], j, search_phi_deg[j], val);
            if (val > threshold) {
                result_theta_deg[index] = search_theta_deg[i];
                result_phi_deg[index] = search_phi_deg[j];
                result_dB_value[index] = val;
                //printf("  pick = %.4f at (%d, %d), deg(%.2f, %.2f)\n",
                //       val, i, j, search_theta_deg[i], search_phi_deg[j]);
                index++;
                if (index == 25) {
                    printf("Too many peaks (limit 25)\n");
                    exit(0);
                }
            }
        }
    }
    
    // If no point, select the maximum value
    if (index == 0) {
        float max_val = -1e9f;
        int max_i = 0, max_j = 0;
        for (int i = 0; i < search_len_theta; ++i) {
            for (int j = 0; j < search_len_phi; ++j) {
                float val = spatial_spectrum_value_dB[i * search_len_phi + j];
                if (val > max_val) {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }
        result_theta_deg[index] = search_theta_deg[max_i];
        result_phi_deg[index] = search_phi_deg[max_j];
        result_dB_value[index] = max_val;
        detected_peak_count[0] = 1;
        //printf("No peak > threshold => Force pick max = %.4f at (%d, %d), deg(%.2f, %.2f)\n",
        //       max_val, max_i, max_j, search_theta_deg[max_i], search_phi_deg[max_j]);
    } else {
        detected_peak_count[0] = index;
    }
    /*
    for(int i =0; i<detected_peak_count[0]; i++){
        printf("result_theta_deg[%d] = %.2f\n", i, result_theta_deg[i]);
        printf("result_phi_deg[%d] = %.2f\n", i, result_phi_deg[i]);
        printf("result_dB_value[%d] = %.2f\n", i, result_dB_value[i]);
    }
    */
    //peak_number[0] = index;
    //printf("== Done: %d peaks selected ==\n\n", detected_peak_count[0]);
}


void m_find_spatial_spectrum_peaks_3D_max(
    float *spatial_spectrum_value_dB,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_deg,
    float *search_phi_deg,
    // output
    float *result_theta_deg,
    float *result_phi_deg,
    float *result_dB_value,
    int *detected_peak_count
){
    float max_temp = -10000000000;
    //float total = 0.0;
    int position_theta_temp=0;
    int position_phi_temp=0;
    for (int i = 0; i < search_len_theta; ++i)
    {
        for (int j = 0; j < search_len_phi; ++j)
        {
            // printf("theta = %.2f, phi = %.2f, spatial_spectrum_value_dB[%d] = %.2f\n", search_theta_deg[i], search_phi_deg[j], i*search_len_phi + j, spatial_spectrum_value_dB[i*search_len_phi + j]);
            // printf("spatial_spectrum_value_dB[%d] = %.2f\n", i*search_len_phi + j, spatial_spectrum_value_dB[i*search_len_phi + j]);
            //total+=spatial_spectrum_value_dB[i*search_len_phi + j];
            if (spatial_spectrum_value_dB[i*search_len_phi + j] > max_temp)
            {
                max_temp = spatial_spectrum_value_dB[i*search_len_phi + j];
                position_theta_temp = i;
                position_phi_temp = j;
            }
        }
    }

    
    result_theta_deg[0] = search_theta_deg[position_theta_temp];
    result_phi_deg[0] = search_phi_deg[position_phi_temp];
    result_dB_value[0] = max_temp;
    detected_peak_count[0] = 1;
    // printf("max (%.2f, %.2f) = %.2f\n", result_theta_deg[0], result_phi_deg[0], result_dB_value[0]);
    // printf("\n");
}

#define MAX_PEAKS 512
#define SUPPRESS_DEGREE 4.0f
void select_top_peaks(
    search_peune* input_peaks, int input_count,
    search_peune* result_peaks, int len_t_angle
) {
    int selected_count = 0;
    int selected_flags[MAX_PEAKS] = {0};  // 標記是否已被選取/抑制

    while (selected_count < len_t_angle) {
        int max_index = -1;
        float max_db = -1e10f;

        // 找出目前最大未被抑制的 peak
        for (int i = 0; i < input_count; ++i) {
            if (!selected_flags[i] && input_peaks[i].dB_value > max_db) {
                max_db = input_peaks[i].dB_value;
                max_index = i;
            }
        }

        // 如果找不到（都被抑制了），就停止
        if (max_index == -1) {
            break;
        }

        // 加入結果
        result_peaks[selected_count++] = input_peaks[max_index];

        float selected_theta = input_peaks[max_index].theta_deg;
        float selected_phi   = input_peaks[max_index].phi_deg;

        // 抑制 suppress_degree 度內的其他 peak
        for (int i = 0; i < input_count; ++i) {
            if (selected_flags[i]) continue;

            float d_theta = input_peaks[i].theta_deg - selected_theta;
            float d_phi   = input_peaks[i].phi_deg - selected_phi;
            float distance = sqrtf(d_theta * d_theta + d_phi * d_phi);
            if (distance < SUPPRESS_DEGREE) {
                selected_flags[i] = 1;
            }
        }
    }
}

// free Search Data
void free_SearchThreadData(SearchThreadData *td) {
    if (!td) return;
    free(td->result_theta_deg);
    free(td->result_phi_deg);
    free(td->result_dB_value);
    free(td);
}