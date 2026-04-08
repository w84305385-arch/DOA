// 2D DOA Estimation - MVDR Algorithm with BMGS QR Decomposition (float)
// Fast Search Implementation using Prune-and-Search Strategy
// - Merged sub-spectrum coarse search for wide angular coverage
//--------------------
#define PI acos(-1)
#define AVX 16
//--------------------
#include <immintrin.h>
#include "../common/complex_matrix_ops.h"
#include "../common/generate_signal.h"
#include "../common/eigen_qr.h"
#include "../common/lu_decomp.h"
#include "../common/spatial_spectrum.h"
#include "../common/doa_parameters.h"
#include <sys/syscall.h>
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
//----------------------global variable---------------------------
float total_multiply_time = 0;
int search_count = 0;
//----------------------------------------------------------------
void doa2d_mvdr_lu_prunesearch_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
    struct timeval time_Rxx_inv_start, time_Rxx_inv_end, time_Rxx_inv_diff;
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;
    struct timeval time_subRxx_start, time_subRxx_end, time_subRxx_diff;
    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;
    struct timeval time_Spatial_Spectrum1_start, time_Spatial_Spectrum1_end, time_Spatial_Spectrum1_diff;
    struct timeval time_Spatial_Spectrum2_start, time_Spatial_Spectrum2_end, time_Spatial_Spectrum2_diff;
    struct timeval time_Spatial_Spectrum3_start, time_Spatial_Spectrum3_end, time_Spatial_Spectrum3_diff;
    struct timeval time_Spatial_Spectrum4_start, time_Spatial_Spectrum4_end, time_Spatial_Spectrum4_diff;

    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;
    struct timeval time_Peak_Search1_start, time_Peak_Search1_end, time_Peak_Search1_diff;
    struct timeval time_Peak_Search2_start, time_Peak_Search2_end, time_Peak_Search2_diff;
    struct timeval time_Peak_Search3_start, time_Peak_Search3_end, time_Peak_Search3_diff;
    struct timeval time_Peak_Search4_start, time_Peak_Search4_end, time_Peak_Search4_diff;
    //-------------------------------------------------------------------
    // === Physical Parameters ===
    float kc = phys.kc;
    // === Tx Parameters ===
    int len_t_angle = tx.number_angle;
    // === Rx Parameters ===
    int Rx_M = rx.Rx_M_x;    
    float d = rx.d;
    int nd = rx.nd;
    
    // ==================================================================
    // ===================== MVDR algorithm start =======================
    // ==================================================================
    // Time parameter initialization
    gettimeofday(&time_MVDR_start, NULL);
    total_multiply_time = 0;
    //---------------------------------------------------------------
    gettimeofday(&time_Rxx_start, NULL);
    float *R_xx_re = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *R_xx_im = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
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
    gettimeofday(&time_Rxx_inv_start, NULL);
    float *Pn_re = (float *)calloc(Rx_M * Rx_M, sizeof(float));
    float *Pn_im = (float *)calloc(Rx_M * Rx_M, sizeof(float));
    matrix_inverse_LU(R_xx_re, R_xx_im, Pn_re , Pn_im, Rx_M);
    gettimeofday(&time_Rxx_inv_end, NULL);
    gettimeofday(&time_Pn_end, NULL);
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, M, M);
    //---------------------------------------------------------------

    // ---------------------------------------------------
    // ----------------- Subarray MVDR -------------------
    // ---------------------------------------------------
    gettimeofday(&time_subRxx_start, NULL);
    const int SUB_M = 8;                  // Number of antennas in each subarray
    const int N_SUBARRAY = Rx_M / SUB_M;  // Total number of subarrays
    const size_t SUB_SIZE = (size_t)SUB_M * SUB_M;

    float **Pn_re_sub = malloc(N_SUBARRAY * sizeof(float *));
    float **Pn_im_sub = malloc(N_SUBARRAY * sizeof(float *));
    SearchConst *search_const_sub = malloc(sizeof(SearchConst));
    if (Rx_M > 8) {
        
    
        // Temporary buffer to store subarray signals
        float *tmp_sig_re = aligned_alloc(64, SUB_M * nd * sizeof(float));
        float *tmp_sig_im = aligned_alloc(64, SUB_M * nd * sizeof(float));
    
        float Rxx_sub_re[SUB_SIZE], Rxx_sub_im[SUB_SIZE];
        float *Ve_sub_re = (float *)malloc(SUB_SIZE * sizeof(float));
        float *Ve_sub_im = (float *)malloc(SUB_SIZE * sizeof(float));
        float *De_sub_re = (float *)malloc(SUB_SIZE * sizeof(float));
        float *De_sub_im = (float *)malloc(SUB_SIZE * sizeof(float));
    
        for (int sid = 0; sid < N_SUBARRAY; ++sid) {
            // Extract subarray signals (continuous SUB_M antennas)
            for (int i = 0; i < SUB_M; ++i) {
                int g_idx = (sid * SUB_M) + i;  // Global antenna index
                memcpy(&tmp_sig_re[i * nd], &Rx_sig_re[g_idx * nd], nd * sizeof(float));
                memcpy(&tmp_sig_im[i * nd], &Rx_sig_im[g_idx * nd], nd * sizeof(float));
            }
    
            // Compute subarray covariance matrix: Rxx
            complex_matrix_conjugate_transpose_multiplication(
                tmp_sig_re, tmp_sig_im, Rxx_sub_re, Rxx_sub_im, SUB_M, nd);
    
            // Compute Covariance matrix : Rxx (subarray)
            complex_matrix_conjugate_transpose_multiplication(
                tmp_sig_re, tmp_sig_im, Rxx_sub_re, Rxx_sub_im, SUB_M, nd);
            for (size_t k = 0; k < SUB_SIZE; ++k) {
                Rxx_sub_re[k] /= nd;
                Rxx_sub_im[k] /= nd;
            }

            //Rxx inv (subarray)
            Pn_re_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            Pn_im_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            matrix_inverse_LU(Rxx_sub_re, Rxx_sub_im, Pn_re_sub[sid], Pn_im_sub[sid], SUB_M);
        }
    
        // Create a SearchConst for subarrays (if needed for search stage)
        
        search_const_sub->Rx_M   = SUB_M;
        search_const_sub->d      = d;
        search_const_sub->kc     = kc;
        search_const_sub->Pn_re  = NULL;  // Will be set in search loop
        search_const_sub->Pn_im  = NULL;
    
        gettimeofday(&time_subRxx_end, NULL);
    }else {
        // For small arrays (Rx_M <= 8), use the original Pn_re and Pn_im directly
        // No subarray splitting is required
        search_const_sub->Rx_M   = Rx_M;
        search_const_sub->d      = d;
        search_const_sub->kc     = kc;
        search_const_sub->Pn_re  = Pn_re;
        search_const_sub->Pn_im  = Pn_im;

        gettimeofday(&time_subRxx_end, NULL);
    }
    //---------------------------------------------------------------

    //---------------------------------------------------------------
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst *search_const = (SearchConst*)malloc(sizeof(SearchConst));
    search_const->Rx_M = Rx_M;
    search_const->d = d;
    search_const->kc = kc;
    search_const->Pn_re = Pn_re;
    search_const->Pn_im = Pn_im;

    float search_step_theta[6] = {30, 10, 3, 1, 0.1};
    float *search_start_theta = (float *)malloc(4 * sizeof(float) + AVX * sizeof(float));

    //---------------- Coarse Search (with subarrays if Rx_M > 8) ----------------
    gettimeofday(&time_Spatial_Spectrum_start, NULL);

    // Define coarse search grid
    search_start_theta[0] = -60;
    int search_len_theta_prune0 = 5;
    float *search_theta_deg_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float) + AVX * sizeof(float));
    float *search_theta_rad_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float) + AVX * sizeof(float));
    for (int i = 0; i < search_len_theta_prune0; ++i) {
        search_theta_deg_prune0[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad_prune0[i] = search_theta_deg_prune0[i] * PI / 180.0f;
    }

    // Allocate coarse spectrum
    float *S_MVDR_dB_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float) + AVX * sizeof(float));
    memset(S_MVDR_dB_prune0, 0, search_len_theta_prune0 * sizeof(float));

    // If array size is large, use subarray combination for coarse search
    if (Rx_M > 8) {
        const int SUB_M = 8;
        const int N_SUBARRAY = Rx_M / SUB_M;
        for (int sid = 0; sid < N_SUBARRAY; ++sid) {
            search_const_sub->Pn_re = Pn_re_sub[sid];
            search_const_sub->Pn_im = Pn_im_sub[sid];

            float *S_tmp = aligned_alloc(64, search_len_theta_prune0 * sizeof(float));
            calculate_spatial_spectrum(search_const_sub,
                                    search_len_theta_prune0,
                                    search_theta_rad_prune0,
                                    S_tmp);

            // Accumulate spatial spectrum from all subarrays
            for (int k = 0; k < search_len_theta_prune0; ++k)
            {
                S_MVDR_dB_prune0[k] += S_tmp[k];
                S_MVDR_dB_prune0[k] /= N_SUBARRAY;  // Average over subarrays
            }

            free(S_tmp);
        }
    } else {
        calculate_spatial_spectrum(search_const_sub,
                                search_len_theta_prune0,
                                search_theta_rad_prune0,
                                S_MVDR_dB_prune0);
    }
    gettimeofday(&time_Spatial_Spectrum_end, NULL);

    // Find peaks
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta_prune0 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MVDR_dB_prune0, position_theta_prune0, search_len_theta_prune0, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);

    //---------------------------------------------------------------
    // parameter setting 1
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    float *search_theta_deg_prune1 = NULL;
    float *search_theta_rad_prune1 = NULL;
    int search_len_theta_prune1;
    calculate_search_theta(search_theta_deg_prune0[position_theta_prune0[0]],
                        &search_len_theta_prune1,
                        &search_theta_deg_prune1,
                        &search_theta_rad_prune1,
                        search_step_theta[1]);
    
    printf("search_theta_deg_prune1\n");
    for(int i = 0; i < search_len_theta_prune1; ++i) {
        printf("%f ", search_theta_deg_prune1[i]);
    }
    printf("\n");

    float *S_MVDR_dB_prune1 = (float *)malloc(search_len_theta_prune1 * sizeof(float));
    memset(S_MVDR_dB_prune1, 0, search_len_theta_prune1 * sizeof(float));
    // If array size is large, use subarray combination for coarse search
    if (Rx_M > 8) {
        for (int sid = 0; sid < N_SUBARRAY; ++sid) {
            search_const_sub->Pn_re = Pn_re_sub[sid];
            search_const_sub->Pn_im = Pn_im_sub[sid];

            float *S_tmp = aligned_alloc(64, search_len_theta_prune1 * sizeof(float));
            calculate_spatial_spectrum(search_const_sub,
                                    search_len_theta_prune1,
                                    search_theta_rad_prune1,
                                    S_tmp);

            // Accumulate spatial spectrum from all subarrays
            for (int k = 0; k < search_len_theta_prune1; ++k)
            {
                S_MVDR_dB_prune1[k] += S_tmp[k];
                S_MVDR_dB_prune1[k] /= N_SUBARRAY;  // Average over subarrays
            }
            free(S_tmp);
        }
    } else {
        calculate_spatial_spectrum(search_const_sub,
                                search_len_theta_prune1,
                                search_theta_rad_prune1,
                                S_MVDR_dB_prune1);
    }
    gettimeofday(&time_Spatial_Spectrum1_end, NULL);

    // Find peaks
    gettimeofday(&time_Peak_Search1_start, NULL);
    int *position_theta_prune1 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MVDR_dB_prune1, position_theta_prune1, search_len_theta_prune1, len_t_angle);
    gettimeofday(&time_Peak_Search1_end, NULL);

    //---------------------------------------------------------------
    // parameter setting 2
    gettimeofday(&time_Spatial_Spectrum2_start, NULL);
    float *search_theta_deg_prune2 = NULL;
    float *search_theta_rad_prune2 = NULL;
    int search_len_theta_prune2;
    calculate_search_theta(search_theta_deg_prune1[position_theta_prune1[0]],
                        &search_len_theta_prune2,
                        &search_theta_deg_prune2,
                        &search_theta_rad_prune2,
                        search_step_theta[2]);

    float *S_MVDR_dB_prune2 = (float *)malloc(search_len_theta_prune2 * sizeof(float));
    memset(S_MVDR_dB_prune2, 0, search_len_theta_prune2 * sizeof(float));
    // If array size is large, use subarray combination for coarse search
    if (Rx_M > 8) {
        for (int sid = 0; sid < N_SUBARRAY; ++sid) {
            search_const_sub->Pn_re = Pn_re_sub[sid];
            search_const_sub->Pn_im = Pn_im_sub[sid];

            float *S_tmp = aligned_alloc(64, search_len_theta_prune2 * sizeof(float));
            calculate_spatial_spectrum(search_const_sub,
                                    search_len_theta_prune2,
                                    search_theta_rad_prune2,
                                    S_tmp);

            // Accumulate spatial spectrum from all subarrays
            for (int k = 0; k < search_len_theta_prune2; ++k)
            {
                S_MVDR_dB_prune2[k] += S_tmp[k];
                S_MVDR_dB_prune2[k] /= N_SUBARRAY;  // Average over subarrays
            }
            free(S_tmp);
        }
    } else {
        calculate_spatial_spectrum(search_const_sub,
                                search_len_theta_prune2,
                                search_theta_rad_prune2,
                                S_MVDR_dB_prune2);
    }
    gettimeofday(&time_Spatial_Spectrum2_end, NULL);

    // Find peaks
    gettimeofday(&time_Peak_Search2_start, NULL);
    int *position_theta_prune2 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MVDR_dB_prune2, position_theta_prune2, search_len_theta_prune2, len_t_angle);
    gettimeofday(&time_Peak_Search2_end, NULL);

    //---------------------------------------------------------------
    // parameter setting 3
    gettimeofday(&time_Spatial_Spectrum3_start, NULL);
    float *search_theta_deg_prune3 = NULL;
    float *search_theta_rad_prune3 = NULL;
    int search_len_theta_prune3;
    calculate_search_theta_high_accuracy(search_theta_deg_prune2[position_theta_prune2[0]],
                                        &search_len_theta_prune3,
                                        &search_theta_deg_prune3,
                                        &search_theta_rad_prune3,
                                        search_step_theta[3]);

    float *S_MVDR_dB_prune3 = (float *)malloc(search_len_theta_prune3 * sizeof(float));
    calculate_spatial_spectrum(search_const,
                            search_len_theta_prune3,
                            search_theta_rad_prune3,
                            S_MVDR_dB_prune3);
    gettimeofday(&time_Spatial_Spectrum3_end, NULL);

    // Find peaks
    gettimeofday(&time_Peak_Search3_start, NULL);
    int *position_theta_prune3 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MVDR_dB_prune3, position_theta_prune3, search_len_theta_prune3, len_t_angle);
    gettimeofday(&time_Peak_Search3_end, NULL);

    //---------------------------------------------------------------
    // parameter setting 4
    gettimeofday(&time_Spatial_Spectrum4_start, NULL);
    float *search_theta_deg_prune4 = NULL;
    float *search_theta_rad_prune4 = NULL;
    int search_len_theta_prune4;
    calculate_search_theta_last(search_theta_deg_prune3[position_theta_prune3[0]],
                                &search_len_theta_prune4,
                                &search_theta_deg_prune4,
                                &search_theta_rad_prune4,
                                search_step_theta[4]);

    printf("search_theta_deg_prune4\n");
    for(int i = 0; i < search_len_theta_prune4; ++i) {
        printf("%f ", search_theta_deg_prune4[i]);
    }
    printf("\n");
    float *S_MVDR_dB_prune4 = (float *)malloc(search_len_theta_prune4 * sizeof(float));
    calculate_spatial_spectrum(search_const,
                            search_len_theta_prune4,
                            search_theta_rad_prune4,
                            S_MVDR_dB_prune4);
    gettimeofday(&time_Spatial_Spectrum4_end, NULL);

    // Find peaks
    gettimeofday(&time_Peak_Search4_start, NULL);
    int *position_theta_prune4 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MVDR_dB_prune4, position_theta_prune4, search_len_theta_prune4, len_t_angle);
    gettimeofday(&time_Peak_Search4_end, NULL);

    gettimeofday(&time_MVDR_end, NULL);
    // ==================================================================
    // ====================== MVDR algorithm end ========================
    // ==================================================================

    //-------------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Rxx_inv, time_Pn, time_subRxx, time_Spatial_Spectrum, time_Peak_Search,time_MVDR;                           // create float parameter in order to convert (us) to (ms)
    float time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Rxx_inv_end, &time_Rxx_inv_start, &time_Rxx_inv_diff);
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);                    // calculate Pn
    timersub(&time_subRxx_end, &time_subRxx_start, &time_subRxx_diff);
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


    timersub(&time_MVDR_end, &time_MVDR_start, &time_MVDR_diff);                    // calculate MVDR

    // Compute time
    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Rxx_inv = time_Rxx_inv_diff.tv_sec * 1000000 + time_Rxx_inv_diff.tv_usec;
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


    time_MVDR = time_MVDR_diff.tv_sec * 1000000 + time_MVDR_diff.tv_usec;;
    float total_time_Spatial_Spectrum, total_time_Peak_Search;
    total_time_Spatial_Spectrum = time_Spatial_Spectrum+time_Spatial_Spectrum1+time_Spatial_Spectrum2+time_Spatial_Spectrum3+time_Spatial_Spectrum4;
    total_time_Peak_Search = time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4;
    
    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MVDR DOA----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("Search step theta:\t%.2f, %.2f, %.2f, %.2f, %.2f (degree)\n", 
        search_step_theta[0], search_step_theta[1], search_step_theta[2], search_step_theta[3], search_step_theta[4]);
    printf("\n");
    printf("\n\t-----Estimated results------\n");
    printf("position0 theta : \t%d\n", position_theta_prune0[0]);
    printf(RED "Theta estimation0 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg_prune0[position_theta_prune0[0]]);
    //printf("Max_theta0 :\t\t%f(dB)\n", max_temp);
    printf("\n");
    printf("position1 theta : \t%d\n", position_theta_prune1[0]);
    printf(RED "Theta estimation1 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg_prune1[position_theta_prune1[0]]);
    //printf("Max_theta1 :\t\t%f(dB)\n", max_temp1);
    printf("\n");
    printf("position2 theta : \t%d\n", position_theta_prune2[0]);
    printf(RED "Theta estimation2 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg_prune2[position_theta_prune2[0]]);
    //printf("Max_theta2 :\t\t%f(dB)\n", max_temp2);
    printf("\n");
    printf("position3 theta : \t%d\n", position_theta_prune3[0]);
    printf(RED "Theta estimation3 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg_prune3[position_theta_prune3[0]]);
    //printf("Max_theta3 :\t\t%f(dB)\n", max_temp3);
    printf("\n");
    printf("position4 theta : \t%d\n", position_theta_prune4[0]);
    printf(RED "Theta estimation4 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg_prune4[position_theta_prune4[0]]);
    //printf("Max_theta4 :\t\t%f(dB)\n", max_temp4);
    printf("\n");
    printf("Total search count: \t%d\n", search_count);
    
    printf("\n\t------------Time------------\n");
    printf("\t  -------DOA start-------\n");
    printf("Total Rxx time: \t\t%.3f(ms)\n", time_Rxx / 1000);
    printf("Total Rxx LU inv time: \t\t%.3f(ms)\n", time_Rxx_inv / 1000);
    printf("Total sub Rxx time: \t\t%.3f(ms)\n", time_subRxx / 1000);
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
    printf("Total MVDR time: \t\t%.3f(ms)\n", time_MVDR / 1000);
    printf(L_GREEN "->Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //-------------------------------------------------------------------
    // free
    free(R_xx_re);
    free(R_xx_im);


    free(Pn_re);
    free(Pn_im);

    free(search_start_theta);

    free(search_theta_deg_prune0);
    free(search_theta_rad_prune0);

    free(S_MVDR_dB_prune0);

    free(position_theta_prune0);

    // Prune 1
    free(search_theta_deg_prune1);
    free(search_theta_rad_prune1);
    free(S_MVDR_dB_prune1);
    free(position_theta_prune1);

    // Prune 2
    free(search_theta_deg_prune2);
    free(search_theta_rad_prune2);
    free(S_MVDR_dB_prune2);
    free(position_theta_prune2);

    // Prune 3
    free(search_theta_deg_prune3);
    free(search_theta_rad_prune3);
    free(S_MVDR_dB_prune3);
    free(position_theta_prune3);

    
    // Prune 4
    free(search_theta_deg_prune4);
    free(search_theta_rad_prune4);
    free(S_MVDR_dB_prune4);
    free(position_theta_prune4);
    
}

int main()
{
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    //-------------------------------------------------------------------
    // Physical Parameters initialization
    PhysicalParameters phys;
    phys.fc = 28e9;
    phys.c = 3e8;
    phys.lambda = (float)phys.c / (float)phys.fc;
    phys.kc = 2 * PI / phys.lambda;
    phys.snr = 10;

    int M =16;
    float angle_theta[100] = {35.0, 45.2, 10.0, 80.0, 70, 45, 58}; // elevation
    // Tx Parameters initialization
    TxParameters tx;
    tx.Tx_M_x = M;
    tx.Tx_beamwidth = 0.1;
    tx.Tx_beamwidth_samples = 100;
    tx.angle_theta = angle_theta;
    tx.number_angle = 1;

    // Rx Parameters initialization
    RxParameters rx;
    rx.Rx_M_x = M;
    rx.d = phys.lambda * 0.5f;
    rx.nd = 1024;
    //-------------------------------------------------------------------
    gettimeofday(&time_MVDR_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);
    //=================== MVDR Algorithm =================================
    doa2d_mvdr_lu_prunesearch_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_MVDR_end, NULL);
    timersub(&time_MVDR_end, &time_MVDR_start, &time_MVDR_diff);
    float time_MVDR;
    time_MVDR = time_MVDR_diff.tv_sec * 1000000 + time_MVDR_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MVDR time : \t\t%.3f(ms)\n" CLOSE, time_MVDR / 1000); //Contains generated signals

    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}

 