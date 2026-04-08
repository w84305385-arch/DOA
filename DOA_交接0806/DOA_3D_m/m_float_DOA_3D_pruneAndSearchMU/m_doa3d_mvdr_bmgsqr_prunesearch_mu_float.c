// 3D DOA Estimation - MVDR Algorithm with BMGS QR Decomposition (float)
// Fast Search Implementation using Prune-and-Search Strategy (Multi-User Version)
//--------------------
#define PI acos(-1)
#define AVX 16
#define EPSILON 1e-3f
//--------------------
#include <immintrin.h>
#include "../common/math_func_3D.h"
#include "../common/complex_matrix_ops.h"
#include "../common/generate_signal_3D.h"
#include "../common/eigen_qr.h"
#include "../common/lu_decomp.h"
#include "../common/spatial_spectrum.h"
//--------------------
// Multithreading
#include "../common/thread_pool.h"
#include "../common/m_complex_matrix_ops.h"
#include "../common/m_spatial_spectrum.h"
//--------------------
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
#include <pthread.h>
#include <sys/syscall.h>


//----------------------global variable---------------------------
float total_multiply_time = 0;
int num_row_blocks;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//----------------------------------------------------------------
void m_doa3d_mvdr_bmgsqr_pns_mu_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff;
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;
    struct timeval time_subRxx_start, time_subRxx_end, time_subRxx_diff;
    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;
    struct timeval time_Spatial_Spectrum1_start, time_Spatial_Spectrum1_end, time_Spatial_Spectrum1_diff;
    struct timeval time_Spatial_Spectrum2_start, time_Spatial_Spectrum2_end, time_Spatial_Spectrum2_diff;
    struct timeval time_Spatial_Spectrum3_start, time_Spatial_Spectrum3_end, time_Spatial_Spectrum3_diff;
    struct timeval time_Spatial_Spectrum4_start, time_Spatial_Spectrum4_end, time_Spatial_Spectrum4_diff;
    struct timeval time_Spatial_Spectrum5_start, time_Spatial_Spectrum5_end, time_Spatial_Spectrum5_diff;

    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;
    struct timeval time_Peak_Search1_start, time_Peak_Search1_end, time_Peak_Search1_diff;
    struct timeval time_Peak_Search2_start, time_Peak_Search2_end, time_Peak_Search2_diff;
    struct timeval time_Peak_Search3_start, time_Peak_Search3_end, time_Peak_Search3_diff;
    struct timeval time_Peak_Search4_start, time_Peak_Search4_end, time_Peak_Search4_diff;
    struct timeval time_Peak_Search5_start, time_Peak_Search5_end, time_Peak_Search5_diff;
    //-------------------------------------------------------------------
    // === Physical Parameters ===
    float kc = phys.kc;
    // === Tx Parameters ===
    int len_t_angle = tx.number_angle;
    // === Rx Parameters ===
    int Rx_M_x = rx.Rx_M_x;
    int Rx_M_y = rx.Rx_M_y;       
    float d = rx.d;
    int nd = rx.nd;
    int qr_iter = rx.qr_iter;
    int BMGS_qr_num_blocks = rx.BMGS_qr_num_blocks;
    // ==================================================================
    // ===================== MVDR algorithm start ======================
    // ==================================================================
    // Time parameter initialization
    gettimeofday(&time_MVDR_start, NULL);
    total_multiply_time = 0.0;
    int Rx_M;
    Rx_M = Rx_M_x*Rx_M_y;

    //---------------------------------------------------------------
    gettimeofday(&time_Rxx_start, NULL);
    float *R_xx_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *R_xx_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    
    // Compute Covariance matrix : Rxx
    complex_matrix_conjugate_transpose_multiplication(Rx_sig_re, Rx_sig_im, R_xx_re, R_xx_im, Rx_M, nd);
    for (int i = 0; i < Rx_M * Rx_M; ++i)
    {
        //printf("\t(%f,%f) /= (%f,%f) = ", R_xx_re[i], R_xx_im[i], *M_ptr, *M_ptr_im);
        R_xx_re[i] /= nd;
        R_xx_im[i] /= nd;
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]);
    }
    //printf("----------Rxx-----------\n");
    //print_complex_matrix(R_xx_re, R_xx_im, Rx_M, Rx_M);
    gettimeofday(&time_Rxx_end, NULL);

    //---------------------------------------------------------------
    // Eigen
    gettimeofday(&time_Eigen_start, NULL);
    float *Ve_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *Ve_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *De_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *De_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *BMGS_qr_time = (float *)malloc(1 * sizeof(float));
    float *qr_time = (float *)malloc(1 * sizeof(float));

    // Compute eigenvector Ve (M, M), eigenvalue De(M, M)
    eigen_BMGS(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, Rx_M, Rx_M, qr_iter, BMGS_qr_num_blocks, BMGS_qr_time, qr_time);
    gettimeofday(&time_Eigen_end, NULL);
    //printf("----------Ve------------\n");
    //print_complex_matrix(Ve_re, Ve_im, M, M);
    //printf("----------De------------\n");
    //print_complex_matrix(De_re, De_im, M, M);

    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    //Rxx inv
    float *Pn_re = (float *)calloc(Rx_M * Rx_M + AVX, sizeof(float));
    float *Pn_im = (float *)calloc(Rx_M * Rx_M + AVX, sizeof(float));
    matrix_inverse_eigen(Ve_re, Ve_im, De_re, De_im, Pn_re, Pn_im, Rx_M);
    gettimeofday(&time_Pn_end, NULL);
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, M, M);
    // ---------------------------------------------------
    // ----------------- Subarray MVDR -------------------
    // ---------------------------------------------------
    gettimeofday(&time_subRxx_start, NULL);
    const int SUB_M_X = 4;  // number of subarray's antenna in x direction
    const int SUB_M_Y = 4;  // number of subarray's antenna in y direction
    const int SUB_M   = SUB_M_X * SUB_M_Y; // number of antennas in one subarray
    const size_t SUB_SIZE = (size_t)SUB_M * SUB_M;
    const int NX = Rx_M_x / SUB_M_X;  // number of subarrays in x direction
    const int NY = Rx_M_y / SUB_M_Y;  // number of subarrays in y direction
    const int N_SUBARRAY = NX * NY;   // number of subarrays

    float **Pn_re_sub = malloc(N_SUBARRAY * sizeof(float*));
    float **Pn_im_sub = malloc(N_SUBARRAY * sizeof(float*));

    float *tmp_sig_re = aligned_alloc(64, SUB_M * nd * sizeof(float));
    float *tmp_sig_im = aligned_alloc(64, SUB_M * nd * sizeof(float));

    float Rxx_sub_re[SUB_SIZE], Rxx_sub_im[SUB_SIZE];

    float *Ve_sub_re = (float *)malloc(SUB_SIZE * sizeof(float));
    float *Ve_sub_im = (float *)malloc(SUB_SIZE * sizeof(float));
    float *De_sub_re = (float *)malloc(SUB_SIZE * sizeof(float));
    float *De_sub_im = (float *)malloc(SUB_SIZE * sizeof(float));
    // Perform the complete process once for each subarray
    for (int sy = 0, sid = 0; sy < NY; ++sy){
        for (int sx = 0;       sx < NX; ++sx, ++sid)
        {
            // Take subarray sig: 4×4×nd snapshot to tmp_sig
            for (int j = 0; j < SUB_M_Y; ++j){
                for (int i = 0; i < SUB_M_X; ++i){
                    int g_idx = (sy * SUB_M_Y + j) * Rx_M_x + (sx * SUB_M_X + i);
                    int l_idx = j * SUB_M_X + i;
                    memcpy(&tmp_sig_re[l_idx * nd], &Rx_sig_re[g_idx * nd], nd * sizeof(float));
                    memcpy(&tmp_sig_im[l_idx * nd], &Rx_sig_im[g_idx * nd], nd * sizeof(float));
                }
            }
            
            // Compute Covariance matrix : Rxx (subarray)
            complex_matrix_conjugate_transpose_multiplication(
                tmp_sig_re, tmp_sig_im, Rxx_sub_re, Rxx_sub_im, SUB_M, nd);
            for (size_t k = 0; k < SUB_SIZE; ++k) {
                Rxx_sub_re[k] /= nd;
                Rxx_sub_im[k] /= nd;
            }
            float *BMGS_qr_time_sub = (float *)malloc(1 * sizeof(float));
            float *qr_time_sub = (float *)malloc(1 * sizeof(float));
            // Eigen (subarray)
            eigen_BMGS(Rxx_sub_re, Rxx_sub_im,
                    Ve_sub_re,  Ve_sub_im,
                    De_sub_re,  De_sub_im,
                    SUB_M, SUB_M,        /* 16×16 */
                    qr_iter, 1,          /* block=1 足夠 */
                    BMGS_qr_time_sub, qr_time_sub);

            //Rxx inv (subarray)
            Pn_re_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            Pn_im_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            matrix_inverse_eigen(Ve_sub_re, Ve_sub_im, De_sub_re, De_sub_im,
                                Pn_re_sub[sid], Pn_im_sub[sid],
                                SUB_M);
        }
    }
    // Create a dedicated SearchConst for 4×4 subarrays
    SearchConst *search_const_sub = malloc(sizeof(SearchConst));

    search_const_sub->Rx_M_x = SUB_M_X;
    search_const_sub->Rx_M_y = SUB_M_Y;
    search_const_sub->Rx_M   = SUB_M;
    search_const_sub->d      = d;
    search_const_sub->kc     = kc;  

    // In the search stage, Pn_re and Pn_im will be set in the loop
    search_const_sub->Pn_re  = NULL;
    search_const_sub->Pn_im  = NULL;
    gettimeofday(&time_subRxx_end, NULL);
    //---------------------------------------------------------------
    
    //---------------------------------------------------------------
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst *search_const = (SearchConst*)malloc(sizeof(SearchConst));
    search_const->Rx_M_x = Rx_M_x;
    search_const->Rx_M_y = Rx_M_y;
    search_const->Rx_M = Rx_M;
    search_const->d = d;
    search_const->kc = kc;
    search_const->Pn_re = Pn_re;
    search_const->Pn_im = Pn_im;
    float search_step_theta[6] = {30, 10, 3, 1, 0.3, 0.1};
    float search_step_phi[6] = {30, 10, 3, 1, 0.3, 0.1};

    //---------------------------------------------------------------
    // Search0 Step 30 deg
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float *search_start_phi = (float *)malloc(4 * sizeof(float));
    // parameter setting
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta 
    search_start_theta[0] = 0;
    int search_len_theta_prune0 = 4;
    float *search_theta_deg_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float));
    float *search_theta_rad_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float));
    for (int i = 0; i < search_len_theta_prune0-1; ++i){ 
        search_theta_deg_prune0[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad_prune0[i] = search_theta_deg_prune0[i] * PI / 180;
    }
    search_theta_deg_prune0[search_len_theta_prune0-1] = search_theta_deg_prune0[search_len_theta_prune0-2] + search_step_theta[0]/2;
    search_theta_rad_prune0[search_len_theta_prune0-1] = search_theta_deg_prune0[search_len_theta_prune0-1] * PI / 180;

    // Search angle phi
    search_start_phi[0] = -60;
    int search_len_phi_prune0 = 5;
    float *search_phi_deg_prune0 = (float *)malloc(search_len_phi_prune0 * sizeof(float));
    float *search_phi_rad_prune0 = (float *)malloc(search_len_phi_prune0 * sizeof(float));
    for (int i = 0; i < search_len_phi_prune0; ++i){ 
        search_phi_deg_prune0[i] = search_start_phi[0] + search_step_phi[0] * i;
        search_phi_rad_prune0[i] = search_phi_deg_prune0[i] * PI / 180;
    }
    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune0 = (float *)malloc(search_len_theta_prune0*search_len_phi_prune0 * sizeof(float) + AVX * sizeof(float));
    memset(S_MVDR_dB_prune0, 0, search_len_theta_prune0*search_len_phi_prune0 * sizeof(float));
    for (int sid = 0; sid < N_SUBARRAY; ++sid) {

        // Point to the corresponding subarray Pn
        search_const_sub->Pn_re = Pn_re_sub[sid];
        search_const_sub->Pn_im = Pn_im_sub[sid];
    
        
        float *S_tmp = aligned_alloc(64, search_len_theta_prune0*search_len_phi_prune0 * sizeof(float));
        calculate_spatial_spectrum_3D(
            search_const_sub,
            search_len_theta_prune0,
            search_len_phi_prune0,
            search_theta_rad_prune0,
            search_phi_rad_prune0,
            S_tmp);
        // Accumulate spatial spectrum
        for (int k = 0; k < search_len_theta_prune0*search_len_phi_prune0; ++k)
            S_MVDR_dB_prune0[k] += S_tmp[k];
    
        free(S_tmp);
    }
    gettimeofday(&time_Spatial_Spectrum_end, NULL);
    //printf("---\n");
    // Find peaks
    gettimeofday(&time_Peak_Search_start, NULL);
    float *result_theta_deg = (float *)malloc(25 * sizeof(float));
    float *result_phi_deg = (float *)malloc(25 * sizeof(float));
    float *dB_value_prune0 = (float *)malloc(25 * sizeof(float));
    int peak_count0;
    printf("=================== search 0 ===================\n");
    float search0_PAPR_threshold = 0.1; // Adjustable threshold
    m_find_spatial_spectrum_peaks_3D_PAPR(
        S_MVDR_dB_prune0, 
        search_len_theta_prune0, 
        search_len_phi_prune0, 
        search_theta_deg_prune0, 
        search_phi_deg_prune0, 
        search0_PAPR_threshold,
        // output
        result_theta_deg,
        result_phi_deg,
        dB_value_prune0, 
        &peak_count0);
    gettimeofday(&time_Peak_Search_end, NULL);
    //printf("---\n");
    printf("peak_count0 = %d\n", peak_count0);
    search_peune search_peune0[peak_count0];// = (search_peune*)malloc(sizeof(search_peune) * (peak_count0+1));

    for (int i = 0; i < peak_count0; i++) {
        search_peune0[i].theta_deg = result_theta_deg[i];
        search_peune0[i].phi_deg   = result_phi_deg[i];
        search_peune0[i].dB_value  = dB_value_prune0[i];
    }

    //---------------------------------------------------------------
    // Search1 Step 10 deg
    // parameter setting 1
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    printf("=================== search 1 ===================\n");
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    int search1_threshold = 9; // Adjustable threshold
    SearchThreadData *Search_td_ptrs1[peak_count0];

    for (int i = 0; i < peak_count0; i++) {
        SearchThreadData *td = malloc(sizeof(SearchThreadData));
        Search_td_ptrs1[i] = td;

        // ------------------------------------------------------------
        // Input Parameters
        // const data 
        td->search_const_input_data = search_const;

        // prev data
        td->theta_deg_prev = search_peune0[i].theta_deg;
        td->phi_deg_prev = search_peune0[i].phi_deg;


        // current data
        td->search_step_theta = search_step_theta[1];
        td->search_step_phi = search_step_phi[1];
        td->PAPR_threshold = search1_threshold;
        td->search_threshold = 0;

        // ------------------------------------------------------------
        // Output Results
        td -> result_theta_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_phi_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_dB_value = (float*)malloc(sizeof(float) * 25);
        td -> detected_peak_count = 0;

        //printf("thread task #%d set finish\n", i);
        addThreadTask(prune_and_search_worker_PAPR, td);
    }

    // wait for all tasks to finish
    wait_for_all_tasks();
    printf("threads (pool) finish\n");

    // ---------------------------------------------------------------
    // Sum the number of detected peaks from all threads
    int peak_count_total_1 = 0;
    for (int i = 0; i < peak_count0; i++) {
        peak_count_total_1 += Search_td_ptrs1[i]->detected_peak_count;
    }
    // Create search_peune[] and collect non-duplicate peaks
    search_peune search_peune1[peak_count_total_1];
    int index = 0;
    for (int i = 0; i < peak_count0; i++) {
        int peak_count = Search_td_ptrs1[i]->detected_peak_count;
        for (int j = 0; j < peak_count; j++) {
            float theta = Search_td_ptrs1[i]->result_theta_deg[j];
            float phi   = Search_td_ptrs1[i]->result_phi_deg[j];
            float dB    = Search_td_ptrs1[i]->result_dB_value[j];

            // Check if the peak is a duplicate (based on theta & phi)
            int is_duplicate = 0;
            for (int k = 0; k < index; k++) {
                if (fabsf(search_peune1[k].theta_deg - theta) < EPSILON &&
                    fabsf(search_peune1[k].phi_deg - phi) < EPSILON) {
                    is_duplicate = 1;
                    break;
                }
            }

            // If not duplicate, add to search_peune[]
            if (!is_duplicate) {
                search_peune1[index].theta_deg = theta;
                search_peune1[index].phi_deg   = phi;
                search_peune1[index].dB_value  = dB;
                index++;
            }
        }
    }
    // Update peak count to reflect unique peaks only
    peak_count_total_1 = index;

    gettimeofday(&time_Spatial_Spectrum1_end, NULL);
    gettimeofday(&time_Peak_Search1_start, NULL);
    gettimeofday(&time_Peak_Search1_end, NULL);

    //---------------------------------------------------------------
    // Search2 Step 3 deg
    // parameter setting 2
    gettimeofday(&time_Spatial_Spectrum2_start, NULL);
    printf("=================== search 2 ===================\n");
    int search2_threshold = 20; // Adjustable threshold
    printf("peak_count_total_1 = %d\n", peak_count_total_1);
    SearchThreadData *Search_td_ptrs2[peak_count_total_1];

    for (int i = 0; i < peak_count_total_1; i++) {
        SearchThreadData *td = malloc(sizeof(SearchThreadData));
        Search_td_ptrs2[i] = td;

        // ------------------------------------------------------------
        // Input Parameters
        // const data 
        td->search_const_input_data = search_const;

        // prev data
        td->theta_deg_prev = search_peune1[i].theta_deg;
        td->phi_deg_prev = search_peune1[i].phi_deg;

        // current data
        td->search_step_theta = search_step_theta[2];
        td->search_step_phi = search_step_phi[2];
        td->PAPR_threshold = search2_threshold;
        td->search_threshold = 0;

        // ------------------------------------------------------------
        // Output Results
        td -> result_theta_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_phi_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_dB_value = (float*)malloc(sizeof(float) * 25);
        td -> detected_peak_count = 0;;

        //printf("thread task #%d set finish\n", i);
        addThreadTask(prune_and_search_worker_PAPR, td);
    }

    // wait for all tasks to finish
    wait_for_all_tasks();
    printf("threads (pool) finish\n");

    // ---------------------------------------------------------------
    // Sum the number of detected peaks from all threads
    int peak_count_total_2 = 0;
    for (int i = 0; i < peak_count_total_1; i++) {
        peak_count_total_2 += Search_td_ptrs2[i]->detected_peak_count;
    }

    // Create search_peune[] and collect non-duplicate peaks
    search_peune search_peune2[peak_count_total_2];
    index = 0;
    for (int i = 0; i < peak_count_total_1; i++) {
        int peak_count = Search_td_ptrs2[i]->detected_peak_count;
        for (int j = 0; j < peak_count; j++) {
            float theta = Search_td_ptrs2[i]->result_theta_deg[j];
            float phi   = Search_td_ptrs2[i]->result_phi_deg[j];
            float dB    = Search_td_ptrs2[i]->result_dB_value[j];

            // Check if the peak is a duplicate (based on theta & phi)
            int is_duplicate = 0;
            for (int k = 0; k < index; k++) {
                if (fabsf(search_peune2[k].theta_deg - theta) < EPSILON &&
                    fabsf(search_peune2[k].phi_deg - phi) < EPSILON) {
                    is_duplicate = 1;
                    break;
                }
            }

            // If not duplicate, add to search_peune[]
            if (!is_duplicate) {
                search_peune2[index].theta_deg = theta;
                search_peune2[index].phi_deg   = phi;
                search_peune2[index].dB_value  = dB;
                index++;
            }
        }
    }
    // Update peak count to reflect unique peaks only
    peak_count_total_2 = index;

    gettimeofday(&time_Spatial_Spectrum2_end, NULL);
    gettimeofday(&time_Peak_Search2_start, NULL);
    gettimeofday(&time_Peak_Search2_end, NULL);

    //---------------------------------------------------------------
    // selection search 2 results
    int search_peune2_result_peaks_count = 0;
    search_peune search_peune2_result_peaks[peak_count_total_2];

    if (peak_count_total_2 > (2 * len_t_angle)) {
        search_peune2_result_peaks_count = 2 * len_t_angle;
        select_top_peaks(search_peune2, peak_count_total_2, search_peune2_result_peaks, search_peune2_result_peaks_count);
    } else {
        search_peune2_result_peaks_count = peak_count_total_2;
        for (int i = 0; i < peak_count_total_2; i++) {
            search_peune2_result_peaks[i].theta_deg = search_peune2[i].theta_deg;
            search_peune2_result_peaks[i].phi_deg   = search_peune2[i].phi_deg;
            search_peune2_result_peaks[i].dB_value  = search_peune2[i].dB_value;
        }
    }
    
    //---------------------------------------------------------------
    // Search3 Step 1 deg
    // parameter setting 3
    gettimeofday(&time_Spatial_Spectrum3_start, NULL);
    printf("=================== search 3 ===================\n");
    int search3_threshold = 6;
    printf("search_peune2_result_peaks_count = %d\n", search_peune2_result_peaks_count);
    SearchThreadData *Search_td_ptrs3[search_peune2_result_peaks_count];

    for (int i = 0; i < search_peune2_result_peaks_count; i++) {
        SearchThreadData *td = malloc(sizeof(SearchThreadData));
        Search_td_ptrs3[i] = td;

        // ------------------------------------------------------------
        // Input Parameters
        // const data 
        td->search_const_input_data = search_const;

        // prev data
        td->theta_deg_prev = search_peune2_result_peaks[i].theta_deg;
        td->phi_deg_prev = search_peune2_result_peaks[i].phi_deg;


        // current data
        td->search_step_theta = search_step_theta[3];
        td->search_step_phi = search_step_phi[3];
        td->PAPR_threshold = search3_threshold;
        td->search_threshold = 0;

        // ------------------------------------------------------------
        // Output Results
        td -> result_theta_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_phi_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_dB_value = (float*)malloc(sizeof(float) * 25);
        td -> detected_peak_count = 0;;

        //printf("thread task #%d set finish\n", i);
        addThreadTask(prune_and_search_worker_max, td);
    }

    // wait for all tasks to finish
    wait_for_all_tasks();
    printf("threads (pool) finish\n");

    // ---------------------------------------------------------------
    // Sum the number of detected peaks from all threads
    int peak_count_total_3 = 0;
    for (int i = 0; i < search_peune2_result_peaks_count; i++) {
        peak_count_total_3 += Search_td_ptrs3[i]->detected_peak_count;
    }

    // Create search_peune[] and collect non-duplicate peaks
    search_peune search_peune3[peak_count_total_3];
    index = 0;
    for (int i = 0; i < search_peune2_result_peaks_count; i++) {
        int peak_count = Search_td_ptrs3[i]->detected_peak_count;
        for (int j = 0; j < peak_count; j++) {
            float theta = Search_td_ptrs3[i]->result_theta_deg[j];
            float phi   = Search_td_ptrs3[i]->result_phi_deg[j];
            float dB    = Search_td_ptrs3[i]->result_dB_value[j];

            // Check if the peak is a duplicate (based on theta & phi)
            int is_duplicate = 0;
            for (int k = 0; k < index; k++) {
                if (fabsf(search_peune3[k].theta_deg - theta) < EPSILON &&
                    fabsf(search_peune3[k].phi_deg - phi) < EPSILON) {
                    is_duplicate = 1;
                    break;
                }
            }

            // If not duplicate, add to search_peune[]
            if (!is_duplicate) {
                search_peune3[index].theta_deg = theta;
                search_peune3[index].phi_deg   = phi;
                search_peune3[index].dB_value  = dB;
                index++;
            }
        }
    }
    // Update peak count to reflect unique peaks only
    peak_count_total_3 = index;
    gettimeofday(&time_Spatial_Spectrum3_end, NULL);
    gettimeofday(&time_Peak_Search3_start, NULL);
    gettimeofday(&time_Peak_Search3_end, NULL);

    //---------------------------------------------------------------
    // Search3 Step 0.3 deg
    // parameter setting 4
    gettimeofday(&time_Spatial_Spectrum4_start, NULL);
    printf("=================== search 4 ===================\n");
    int search4_threshold = 6;
    printf("peak_count_total_3 = %d\n", peak_count_total_3);
    SearchThreadData *Search_td_ptrs4[peak_count_total_3];

    for (int i = 0; i < peak_count_total_3; i++) {
        SearchThreadData *td = malloc(sizeof(SearchThreadData));
        Search_td_ptrs4[i] = td;

        // ------------------------------------------------------------
        // Input Parameters
        // const data 
        td->search_const_input_data = search_const;

        // prev data
        td->theta_deg_prev = search_peune3[i].theta_deg;
        td->phi_deg_prev = search_peune3[i].phi_deg;


        // current data
        td->search_step_theta = search_step_theta[4];
        td->search_step_phi = search_step_phi[4];
        td->PAPR_threshold = search4_threshold;
        td->search_threshold = 0;

        // ------------------------------------------------------------
        // Output Results
        td -> result_theta_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_phi_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_dB_value = (float*)malloc(sizeof(float) * 25);
        td -> detected_peak_count = 0;;

        //printf("thread task #%d set finish\n", i);
        addThreadTask(prune_and_search_worker_max, td);
    }

    // wait for all tasks to finish
    wait_for_all_tasks();
    printf("threads (pool) finish\n");

    // ---------------------------------------------------------------
    // Sum the number of detected peaks from all threads
    int peak_count_total_4 = 0;
    for (int i = 0; i < peak_count_total_3; i++) {
        peak_count_total_4 += Search_td_ptrs4[i]->detected_peak_count;
    }

    // Create search_peune[] and collect non-duplicate peaks
    search_peune search_peune4[peak_count_total_4];
    index = 0;
    for (int i = 0; i < peak_count_total_3; i++) {
        int peak_count = Search_td_ptrs4[i]->detected_peak_count;
        for (int j = 0; j < peak_count; j++) {
            float theta = Search_td_ptrs4[i]->result_theta_deg[j];
            float phi   = Search_td_ptrs4[i]->result_phi_deg[j];
            float dB    = Search_td_ptrs4[i]->result_dB_value[j];

            // Check if the peak is a duplicate (based on theta & phi)
            int is_duplicate = 0;
            for (int k = 0; k < index; k++) {
                if (fabsf(search_peune4[k].theta_deg - theta) < EPSILON &&
                    fabsf(search_peune4[k].phi_deg - phi) < EPSILON) {
                    is_duplicate = 1;
                    break;
                }
            }

            // If not duplicate, add to search_peune[]
            if (!is_duplicate) {
                search_peune4[index].theta_deg = theta;
                search_peune4[index].phi_deg   = phi;
                search_peune4[index].dB_value  = dB;
                index++;
            }
        }
    }
    // Update peak count to reflect unique peaks only
    peak_count_total_4 = index;

    gettimeofday(&time_Spatial_Spectrum4_end, NULL);
    gettimeofday(&time_Peak_Search4_start, NULL);
    gettimeofday(&time_Peak_Search4_end, NULL);

    //---------------------------------------------------------------
    // Search3 Step 0.1 deg
    // parameter setting 5
    gettimeofday(&time_Spatial_Spectrum5_start, NULL);
    printf("=================== search 5 ===================\n");
    int search5_threshold = 6;
    printf("peak_count_total_4 = %d\n", peak_count_total_4);
    SearchThreadData *Search_td_ptrs5[peak_count_total_4];

    for (int i = 0; i < peak_count_total_4; i++) {
        SearchThreadData *td = malloc(sizeof(SearchThreadData));
        Search_td_ptrs5[i] = td;

        // ------------------------------------------------------------
        // Input Parameters
        // const data 
        td->search_const_input_data = search_const;

        // prev data
        td->theta_deg_prev = search_peune4[i].theta_deg;
        td->phi_deg_prev = search_peune4[i].phi_deg;


        // current data
        td->search_step_theta = search_step_theta[5];
        td->search_step_phi = search_step_phi[5];
        td->PAPR_threshold = search5_threshold;
        td->search_threshold = 0;

        // ------------------------------------------------------------
        // Output Results
        td -> result_theta_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_phi_deg = (float*)malloc(sizeof(float) * 25);
        td -> result_dB_value = (float*)malloc(sizeof(float) * 25);
        td -> detected_peak_count = 0;;

        //printf("thread task #%d set finish\n", i);
        addThreadTask(prune_and_search_worker_max, td);
    }

    // wait for all tasks to finish
    wait_for_all_tasks();
    printf("threads (pool) finish\n");

    // ---------------------------------------------------------------
    // Sum the number of detected peaks from all threads
    int peak_count_total_5 = 0;
    for (int i = 0; i < peak_count_total_4; i++) {
        peak_count_total_5 += Search_td_ptrs5[i]->detected_peak_count;
    }

    // Create search_peune[] and collect non-duplicate peaks
    search_peune search_peune5[peak_count_total_5];
    index = 0;
    for (int i = 0; i < peak_count_total_4; i++) {
        int peak_count = Search_td_ptrs5[i]->detected_peak_count;
        for (int j = 0; j < peak_count; j++) {
            float theta = Search_td_ptrs5[i]->result_theta_deg[j];
            float phi   = Search_td_ptrs5[i]->result_phi_deg[j];
            float dB    = Search_td_ptrs5[i]->result_dB_value[j];

            // Check if the peak is a duplicate (based on theta & phi)
            int is_duplicate = 0;
            for (int k = 0; k < index; k++) {
                if (fabsf(search_peune5[k].theta_deg - theta) < EPSILON &&
                    fabsf(search_peune5[k].phi_deg - phi) < EPSILON) {
                    is_duplicate = 1;
                    break;
                }
            }

            // If not duplicate, add to search_peune[]
            if (!is_duplicate) {
                search_peune5[index].theta_deg = theta;
                search_peune5[index].phi_deg   = phi;
                search_peune5[index].dB_value  = dB;
                index++;
            }
        }
    }
    // Update peak count to reflect unique peaks only
    peak_count_total_5 = index;

    gettimeofday(&time_Spatial_Spectrum5_end, NULL);
    gettimeofday(&time_Peak_Search5_start, NULL);
    gettimeofday(&time_Peak_Search5_end, NULL);

    //---------------------------------------------------------------
    // Final result
    search_peune result_peaks[len_t_angle];
    select_top_peaks(search_peune5, peak_count_total_5, result_peaks, len_t_angle);
    gettimeofday(&time_MVDR_end, NULL);
    //---------------------------------------------------------------
    //------------------------ Search emd ---------------------------
    //---------------------------------------------------------------

    // ==================================================================
    // ====================== MVDR algorithm end =======================
    // ==================================================================
    
    //---------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Eigen, time_Pn, time_subRxx, time_Spatial_Spectrum, time_Peak_Search, time_MVDR;      // create float parameter in order to convert (us) to (ms)
    float time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4, time_Spatial_Spectrum5;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4, time_Peak_Search5;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Eigen_end, &time_Eigen_start, &time_Eigen_diff);           // calculate Eigen
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);                    // calculate Pn
    timersub(&time_subRxx_end, &time_subRxx_start, &time_subRxx_diff);        // calculate subRxx
    timersub(&time_Spatial_Spectrum_end, &time_Spatial_Spectrum_start, &time_Spatial_Spectrum_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search_end, &time_Peak_Search_start, &time_Peak_Search_diff);                    // calculate Peak Search
    // 1
    timersub(&time_Spatial_Spectrum1_end, &time_Spatial_Spectrum1_start, &time_Spatial_Spectrum1_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search1_end, &time_Peak_Search1_start, &time_Peak_Search1_diff);                         // calculate Peak Search

    // 2
    timersub(&time_Spatial_Spectrum2_end, &time_Spatial_Spectrum2_start, &time_Spatial_Spectrum2_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search2_end, &time_Peak_Search2_start, &time_Peak_Search2_diff);                         // calculate Peak Search

    // 3
    timersub(&time_Spatial_Spectrum3_end, &time_Spatial_Spectrum3_start, &time_Spatial_Spectrum3_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search3_end, &time_Peak_Search3_start, &time_Peak_Search3_diff); 
    
    // 4
    timersub(&time_Spatial_Spectrum4_end, &time_Spatial_Spectrum4_start, &time_Spatial_Spectrum4_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search4_end, &time_Peak_Search4_start, &time_Peak_Search4_diff); 

    // 5
    timersub(&time_Spatial_Spectrum5_end, &time_Spatial_Spectrum5_start, &time_Spatial_Spectrum5_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search5_end, &time_Peak_Search5_start, &time_Peak_Search5_diff); 

    timersub(&time_MVDR_end, &time_MVDR_start, &time_MVDR_diff);                    // calculate MVDR

    // Compute time
    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Eigen = time_Eigen_diff.tv_sec * 1000000 + time_Eigen_diff.tv_usec;
    time_Pn = time_Pn_diff.tv_sec * 1000000 + time_Pn_diff.tv_usec;
    time_subRxx = time_subRxx_diff.tv_sec * 1000000 + time_subRxx_diff.tv_usec;
    time_Spatial_Spectrum = time_Spatial_Spectrum_diff.tv_sec * 1000000 + time_Spatial_Spectrum_diff.tv_usec;
    time_Peak_Search = time_Peak_Search_diff.tv_sec * 1000000 + time_Peak_Search_diff.tv_usec;
    // 1
    time_Spatial_Spectrum1 = time_Spatial_Spectrum1_diff.tv_sec * 1000000 + time_Spatial_Spectrum1_diff.tv_usec;
    time_Peak_Search1 = time_Peak_Search1_diff.tv_sec * 1000000 + time_Peak_Search1_diff.tv_usec;

    // 2
    time_Spatial_Spectrum2 = time_Spatial_Spectrum2_diff.tv_sec * 1000000 + time_Spatial_Spectrum2_diff.tv_usec;
    time_Peak_Search2 = time_Peak_Search2_diff.tv_sec * 1000000 + time_Peak_Search2_diff.tv_usec;

    // 3
    time_Spatial_Spectrum3 = time_Spatial_Spectrum3_diff.tv_sec * 1000000 + time_Spatial_Spectrum3_diff.tv_usec;
    time_Peak_Search3 = time_Peak_Search3_diff.tv_sec * 1000000 + time_Peak_Search3_diff.tv_usec;

    // 4
    time_Spatial_Spectrum4 = time_Spatial_Spectrum4_diff.tv_sec * 1000000 + time_Spatial_Spectrum4_diff.tv_usec;
    time_Peak_Search4 = time_Peak_Search4_diff.tv_sec * 1000000 + time_Peak_Search4_diff.tv_usec;

    // 5
    time_Spatial_Spectrum5 = time_Spatial_Spectrum5_diff.tv_sec * 1000000 + time_Spatial_Spectrum5_diff.tv_usec;
    time_Peak_Search5 = time_Peak_Search5_diff.tv_sec * 1000000 + time_Peak_Search5_diff.tv_usec;

    time_MVDR = time_MVDR_diff.tv_sec * 1000000 + time_MVDR_diff.tv_usec;;
    float total_time_Spatial_Spectrum, total_time_Peak_Search;
    total_time_Spatial_Spectrum = time_Spatial_Spectrum+time_Spatial_Spectrum1+time_Spatial_Spectrum2+time_Spatial_Spectrum3+time_Spatial_Spectrum4+time_Spatial_Spectrum5+ time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4+time_Peak_Search5;
    total_time_Peak_Search = time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4+time_Peak_Search5;
    // -----------------------------------------------------------------
    // print parameter
    printf("\n\t-----Estimated results------\n");
    printf("------ Search 0 ------\n");
    for(int i=0;i<peak_count0;i++){
        printf("--- position0 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune0[i].theta_deg, search_peune0[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune0[i].dB_value);
    }
    printf("\n");
    printf("------ Search 1 ------\n");
    for(int i=0;i<peak_count_total_1;i++){
        printf("--- position1 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune1[i].theta_deg, search_peune1[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune1[i].dB_value);
    }
    
    printf("\n");
    printf("------ Search 2 ------\n");
    for(int i=0;i<peak_count_total_2;i++){
        printf("--- position2 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune2[i].theta_deg, search_peune2[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune2[i].dB_value);
    }
    printf("\n");
    printf("------ Selection Search 2 results ------\n");
    for(int i=0;i<search_peune2_result_peaks_count;i++){
        printf("--- position2_result #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune2_result_peaks[i].theta_deg, search_peune2_result_peaks[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune2_result_peaks[i].dB_value);
    }
    printf("\n");
    printf("------ Search 3 ------\n");
    for(int i=0;i<peak_count_total_3;i++){
        printf("--- position3 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune3[i].theta_deg, search_peune3[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune3[i].dB_value);
    }
    printf("\n");
    printf("------ Search 4 ------\n");
    for(int i=0;i<peak_count_total_4;i++){
        printf("--- position4 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune4[i].theta_deg, search_peune4[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune4[i].dB_value);
    }
    printf("\n");

    printf("------ Search 5 ------\n");
    for(int i=0;i<peak_count_total_5;i++){
        printf("--- position5 #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, search_peune5[i].theta_deg, search_peune5[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", search_peune5[i].dB_value);
    }
    printf("\n");

    printf("-----------------------------------------\n");
    printf("----------------MVDR DOA-----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("BMGS-QR num of blocks:\t%d\n", BMGS_qr_num_blocks);
    printf("\n");
    printf("------ Final result ------\n");
    for(int i=0;i<len_t_angle;i++){
        printf("--- position #%d ---\n", i);
        printf(RED "Estimation :\t\t(%.3f, %.3f) (degree)\n" CLOSE, result_peaks[i].theta_deg, result_peaks[i].phi_deg);
        printf("Spatial Spectrum dB :\t%.3f\n", result_peaks[i].dB_value);
    }
    printf("\n");


    printf("\n\t------------Time------------\n");
    printf("\t  -------DOA start-------\n");
    printf("Total Rxx time: \t\t%.3f(ms)\n", time_Rxx / 1000);
    printf("BMGS QR time: \t\t\t%.3f(ms)\n", *BMGS_qr_time / 1000);
    printf("---> QR time: \t\t\t%.3f(ms)\n", *qr_time / 1000);
    printf("---> synchronous time: \t\t%.3f(ms)\n", (*BMGS_qr_time / 1000) - (*qr_time / 1000));
    printf("Total Eigen time: \t\t%.3f(ms)\n", time_Eigen / 1000 - (*BMGS_qr_time / 1000));
    printf("Total Pn time: \t\t\t%.3f(ms)\n", time_Pn / 1000);
    printf("Total subRxx time: \t\t%.3f(ms)\n", time_subRxx / 1000);
    printf("Total Spatial Spectrum time: \t%.3f(ms)\n", total_time_Spatial_Spectrum / 1000);
    printf("Total Peak Search time: \t%.3f(ms)\n", total_time_Peak_Search / 1000);
    printf("----> Spatial Spectrum time: \t%.3f(ms)\n", time_Spatial_Spectrum / 1000);
    printf("----> Peak Search time: \t%.3f(ms)\n", time_Peak_Search / 1000);
    printf("----> Spatial Spectrum1 time: \t%.3f(ms)\n", time_Spatial_Spectrum1 / 1000);
    printf("----> Peak Search1 time: \t%.3f(ms)\n", time_Peak_Search1 / 1000);
    printf("----> Spatial Spectrum2 time: \t%.3f(ms)\n", time_Spatial_Spectrum2 / 1000);
    printf("----> Peak Search2 time: \t%.3f(ms)\n", time_Peak_Search2 / 1000);
    printf("----> Spatial Spectrum3 time: \t%.3f(ms)\n", time_Spatial_Spectrum3 / 1000);
    printf("----> Peak Search3 time: \t%.3f(ms)\n", time_Peak_Search3 / 1000);
    printf("----> Spatial Spectrum4 time: \t%.3f(ms)\n", time_Spatial_Spectrum4 / 1000);
    printf("----> Peak Search4 time: \t%.3f(ms)\n", time_Peak_Search4 / 1000);
    printf("----> Spatial Spectrum5 time: \t%.3f(ms)\n", time_Spatial_Spectrum5 / 1000);
    printf("----> Peak Search5 time: \t%.3f(ms)\n", time_Peak_Search5 / 1000);
    printf(L_GREEN "Total MVDR REAL time : \t%.3f(ms)\n" CLOSE, time_MVDR / 1000);
    
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune0.csv", S_MVDR_dB_prune0, search_len_theta_prune0*search_len_phi_prune0);
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune1.csv", S_MVDR_dB_prune1, search_len_theta_prune1*search_len_phi_prune1);
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune2.csv", S_MVDR_dB_prune2, search_len_theta_prune2*search_len_phi_prune2);
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune3.csv", S_MVDR_dB_prune3, search_len_theta_prune3*search_len_phi_prune3);
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune4.csv", S_MVDR_dB_prune4, search_len_theta_prune4*search_len_phi_prune4);
    // save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MVDR_spectrum_dB/S_MVDR_dB_prune5.csv", S_MVDR_dB_prune5, search_len_theta_prune5*search_len_phi_prune5);

    // free
    free(R_xx_re);
    free(R_xx_im);

    free(Ve_re);
    free(Ve_im);
    free(De_re);
    free(De_im);

    free(BMGS_qr_time);
    free(qr_time);

    //free(vet_noise_re);
    //free(vet_noise_im);
    free(Pn_re);
    free(Pn_im);

    free(search_start_theta);
    free(search_start_phi);

    // Prune 0
    free(search_theta_deg_prune0);
    free(search_theta_rad_prune0);
    free(search_phi_deg_prune0);
    free(search_phi_rad_prune0);
    free(S_MVDR_dB_prune0);

    free(search_const);
    // search_peune
    free(result_theta_deg);
    free(result_phi_deg);
    free(dB_value_prune0);
    
    for (int i = 0; i < peak_count0; i++) {
        free_SearchThreadData(Search_td_ptrs1[i]);
    }
    for (int i = 0; i < peak_count_total_1; i++) {
        free_SearchThreadData(Search_td_ptrs2[i]);
    }
    for (int i = 0; i < search_peune2_result_peaks_count; i++) {
        free_SearchThreadData(Search_td_ptrs3[i]);
    }
    for (int i = 0; i < peak_count_total_3; i++) {
        free_SearchThreadData(Search_td_ptrs4[i]);
    }
    for (int i = 0; i < peak_count_total_4; i++) {
        free_SearchThreadData(Search_td_ptrs5[i]);
    }
}

int main()
{
    bind_main_to_core(2);
    int threads_number = 8;
    init_thread_pool(threads_number, 3);
    //
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    //-------------------------------------------------------------------
    // Physical Parameters initialization
    PhysicalParameters phys;
    phys.fc = 28e9;
    phys.c = 3e8;
    phys.lambda = (float)phys.c / (float)phys.fc;
    phys.kc = 2 * PI / phys.lambda;
    phys.snr = 10;

    int M = 8;
    float angle_theta[100] = { 15, 45, 45.0, 15, 70, 45, 58}; // elevation
    float angle_phi[100] = { 15, 45, -45.0, -15.0, 0, 45, 58}; // azimuth
    // Tx Parameters initialization
    TxParameters tx;
    tx.Tx_M_x = M;
    tx.Tx_M_y = M;
    tx.Tx_beamwidth = 0.1;
    tx.Tx_beamwidth_samples = 100;
    tx.angle_theta = angle_theta;
    tx.angle_phi = angle_phi;
    tx.number_angle = 4;

    // Rx Parameters initialization
    RxParameters rx;
    rx.Rx_M_x = M;
    rx.Rx_M_y = M;       
    rx.d = phys.lambda * 0.5f;
    rx.nd = 1024;
    rx.qr_iter = 2;
    rx.BMGS_qr_num_blocks = M;
    //-------------------------------------------------------------------
    gettimeofday(&time_MVDR_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    //=================== MVDR Algorithm =================================
    m_doa3d_mvdr_bmgsqr_pns_mu_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_MVDR_end, NULL);
    timersub(&time_MVDR_end, &time_MVDR_start, &time_MVDR_diff);
    float time_MVDR;
    time_MVDR = time_MVDR_diff.tv_sec * 1000000 + time_MVDR_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MVDR time : \t\t%.3f(ms)\n" CLOSE, time_MVDR / 1000);


    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}


 