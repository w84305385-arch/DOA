// 3D DOA MVDR LU float-int32 fast search (prune-and-search) version
// All modules use parallel matrix multiplication, except for the search stage.
// search stage use float32-int32
//--------------------
#define PI acos(-1)
#define AVX 16
//--------------------
#include <immintrin.h>
#include "../common/q_format_config.h"
#include "../common/complex_matrix_ops.h"
#include "../common/generate_signal_3D.h"
#include "../common/eigen_qr.h"
#include "../common/lu_decomp.h"
#include "../common/spatial_spectrum.h"
#include "../common/doa_parameters.h"
//--------------------
// Multithreading
#include "../common/thread_pool.h"
#include "../common/m_complex_matrix_ops.h"
#include "../common/m_spatial_spectrum.h"
#include "../common/m_lu_decomp.h"
//--------------------
// int32
#include "../common/complex_matrix_ops_int32.h"
#include "../common/spatial_spectrum_int32.h"
#include "../common/m_complex_matrix_ops_int32.h"
#include "../common/m_spatial_spectrum_int32.h"
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
#include <sys/syscall.h>
//----------------------global variable---------------------------
float total_multiply_time = 0;
int search_count = 0;
int num_row_blocks = 1; // number of row blocks for parallel matrix multiplication
//----------------------------------------------------------------
void m_doa3d_mvdr_lu_pns_parMul_cpu_float_int32_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
    struct timeval time_Rxx_inv_start, time_Rxx_inv_end, time_Rxx_inv_diff;
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

    // ==================================================================
    // ===================== MVDR algorithm start =======================
    // ==================================================================
    // Time parameter initialization
    gettimeofday(&time_MVDR_start, NULL);
    total_multiply_time = 0;
    int Rx_M;
    Rx_M = Rx_M_x*Rx_M_y;
    //---------------------------------------------------------------
    gettimeofday(&time_Rxx_start, NULL);
    float *R_xx_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *R_xx_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));


    // Compute Covariance matrix : Rxx
    m_complex_matrix_conjugate_transpose_multiplication_row_block_parallel(Rx_sig_re, Rx_sig_im, R_xx_re, R_xx_im, Rx_M, nd, num_row_blocks);
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
    m_matrix_inverse_LU(R_xx_re, R_xx_im, Pn_re , Pn_im, Rx_M);
    gettimeofday(&time_Rxx_inv_end, NULL);
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

            //Rxx inv (subarray)
            Pn_re_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            Pn_im_sub[sid] = aligned_alloc(64, SUB_SIZE * sizeof(float));
            matrix_inverse_LU(Rxx_sub_re, Rxx_sub_im, Pn_re_sub[sid], Pn_im_sub[sid], SUB_M);
        }
    }
    // float to int32_t conversion
    int32_t **Pn_re_sub_int32 = malloc(N_SUBARRAY * sizeof(int32_t*));
    int32_t **Pn_im_sub_int32 = malloc(N_SUBARRAY * sizeof(int32_t*));
    for (int i = 0; i < N_SUBARRAY; ++i) {
        Pn_re_sub_int32[i] = aligned_alloc(64, SUB_SIZE * sizeof(int32_t));
        Pn_im_sub_int32[i] = aligned_alloc(64, SUB_SIZE * sizeof(int32_t));
    }
    for (int i = 0; i < N_SUBARRAY; ++i) {
        float_matrix_to_q_format(
            Pn_re_sub_int32[i], Pn_im_sub_int32[i],
            Pn_re_sub[i], Pn_im_sub[i],
            SUB_M, SUB_M);  // 加上 Q_SHIFT 若有定義
    }
    SearchConst_int32 *search_const_sub_int32 = (SearchConst_int32*)malloc(sizeof(SearchConst_int32));
    search_const_sub_int32->Rx_M_x = SUB_M_X;
    search_const_sub_int32->Rx_M_y = SUB_M_Y;
    search_const_sub_int32->Rx_M = SUB_M;
    search_const_sub_int32->d = d;
    search_const_sub_int32->kc = kc;
    search_const_sub_int32->Pn_re = NULL;
    search_const_sub_int32->Pn_im = NULL;
    gettimeofday(&time_subRxx_end, NULL);
    // ---------------------------------------------------
    // float to int32_t conversion
    int32_t *Pn_re_int32 = (int32_t*)malloc(Rx_M * Rx_M* sizeof(int32_t));
    int32_t *Pn_im_int32 = (int32_t *)malloc(Rx_M * Rx_M* sizeof(int32_t));
    // Convert float to int32_t
    float_matrix_to_q_format(Pn_re_int32, Pn_im_int32, Pn_re, Pn_im, Rx_M, Rx_M);
    //---------------------------------------------------------------
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst_int32 *search_const_int32 = (SearchConst_int32*)malloc(sizeof(SearchConst_int32));
    search_const_int32->Rx_M_x = Rx_M_x;
    search_const_int32->Rx_M_y = Rx_M_y;
    search_const_int32->Rx_M = Rx_M;
    search_const_int32->d = d;
    search_const_int32->kc = kc;
    search_const_int32->Pn_re = Pn_re_int32;
    search_const_int32->Pn_im = Pn_im_int32;
    float search_step_theta[6] = {30, 10, 3, 1, 0.3, 0.1};
    float search_step_phi[6] = {30, 10, 3, 1, 0.3, 0.1};
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float *search_start_phi = (float *)malloc(4 * sizeof(float));
    //---------------------------------------------------------------
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
        search_const_sub_int32->Pn_re = Pn_re_sub_int32[sid];
        search_const_sub_int32->Pn_im = Pn_im_sub_int32[sid];
    
        
        float *S_tmp = aligned_alloc(64, search_len_theta_prune0*search_len_phi_prune0 * sizeof(float));
        calculate_spatial_spectrum_3D_int32(
            search_const_sub_int32,
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
    // find peaks
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta_prune0 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune0 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune0, position_theta_prune0, position_phi_prune0, search_len_theta_prune0, search_len_phi_prune0, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    //---------------------------------------------------------------
    // parameter setting 1
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune1 = NULL;
    float *search_theta_rad_prune1 = NULL;
    int search_len_theta_prune1;
    calculate_search_theta_3D(search_theta_deg_prune0[position_theta_prune0[0]], &search_len_theta_prune1, &search_theta_deg_prune1, &search_theta_rad_prune1, search_step_theta[1]);
    //printf("len_dth1 = %d\n", len_dth1);
    // Search angle phi
    float *search_phi_deg_prune1 = NULL;
    float *search_phi_rad_prune1 = NULL;
    int search_len_phi_prune1;
    calculate_search_phi_3D(search_phi_deg_prune0[position_phi_prune0[0]], &search_len_phi_prune1, &search_phi_deg_prune1, &search_phi_rad_prune1, search_step_phi[1]);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune1 = (float *)malloc(search_len_theta_prune1*search_len_phi_prune1 * sizeof(float) + AVX * sizeof(float));
    if(Rx_M == 576){
        // Calculate Spatial Spectrum and Peak Search
        memset(S_MVDR_dB_prune1, 0, search_len_theta_prune1*search_len_phi_prune1 * sizeof(float));
        for (int sid = 0; sid < N_SUBARRAY; ++sid) {

            // Point to the corresponding subarray Pn
            search_const_sub_int32->Pn_re = Pn_re_sub_int32[sid];
            search_const_sub_int32->Pn_im = Pn_im_sub_int32[sid];
        
            
            float *S_tmp = aligned_alloc(64, search_len_theta_prune1*search_len_phi_prune1 * sizeof(float));
            calculate_spatial_spectrum_3D_int32(
                search_const_sub_int32,
                search_len_theta_prune1,
                search_len_phi_prune1,
                search_theta_rad_prune1,
                search_phi_rad_prune1,
                S_tmp);
        
            // Accumulate spatial spectrum
            for (int k = 0; k < search_len_theta_prune1*search_len_phi_prune1; ++k)
                S_MVDR_dB_prune1[k] += S_tmp[k];
        
            free(S_tmp);
        }
    } else {
        //float *S_MVDR_dB_prune1 = (float *)malloc(search_len_theta_prune1*search_len_phi_prune1 * sizeof(float) + AVX * sizeof(float));
        m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
            search_const_int32, 
            search_len_theta_prune1, 
            search_len_phi_prune1, 
            search_theta_rad_prune1, 
            search_phi_rad_prune1,
            S_MVDR_dB_prune1
        );
    } 
    gettimeofday(&time_Spatial_Spectrum1_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search1_start, NULL);
    int *position_theta_prune1 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune1 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune1, position_theta_prune1, position_phi_prune1, search_len_theta_prune1, search_len_phi_prune1, len_t_angle);
    gettimeofday(&time_Peak_Search1_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 2
    gettimeofday(&time_Spatial_Spectrum2_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune2 = NULL;
    float *search_theta_rad_prune2 = NULL;
    int search_len_theta_prune2;
    calculate_search_theta_3D(search_theta_deg_prune1[position_theta_prune1[0]], &search_len_theta_prune2, &search_theta_deg_prune2, &search_theta_rad_prune2, search_step_theta[2]);
    //printf("len_dth2 = %d\n", len_dth2);
    // Search angle phi
    float *search_phi_deg_prune2 = NULL;
    float *search_phi_rad_prune2 = NULL;
    int search_len_phi_prune2;
    calculate_search_phi_3D(search_phi_deg_prune1[position_phi_prune1[0]], &search_len_phi_prune2, &search_phi_deg_prune2, &search_phi_rad_prune2, search_step_phi[2]);

    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune2 = (float *)malloc(search_len_theta_prune2*search_len_phi_prune2 * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune2, 
        search_len_phi_prune2, 
        search_theta_rad_prune2, 
        search_phi_rad_prune2,
        S_MVDR_dB_prune2
    );  

    gettimeofday(&time_Spatial_Spectrum2_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search2_start, NULL);
    int *position_theta_prune2 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune2 = (int *)malloc(25 * sizeof(int));
    position_theta_prune2[0] = 0;
    position_phi_prune2[0] = 0; 
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune2, position_theta_prune2, position_phi_prune2, search_len_theta_prune2, search_len_phi_prune2, len_t_angle);
    gettimeofday(&time_Peak_Search2_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 3
    gettimeofday(&time_Spatial_Spectrum3_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune3 = NULL;
    float *search_theta_rad_prune3 = NULL;
    int search_len_theta_prune3;
    calculate_search_theta_3D(search_theta_deg_prune2[position_theta_prune2[0]], &search_len_theta_prune3, &search_theta_deg_prune3, &search_theta_rad_prune3, search_step_theta[3]);
    //printf("len_dth3 = %d\n", len_dth3);
    // Search angle phi
    float *search_phi_deg_prune3 = NULL;
    float *search_phi_rad_prune3 = NULL;
    int search_len_phi_prune3;
    calculate_search_phi_3D(search_phi_deg_prune2[position_phi_prune2[0]], &search_len_phi_prune3, &search_phi_deg_prune3, &search_phi_rad_prune3, search_step_phi[3]);

    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune3 = (float *)malloc(search_len_theta_prune3*search_len_phi_prune3 * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune3, 
        search_len_phi_prune3, 
        search_theta_rad_prune3, 
        search_phi_rad_prune3,
        S_MVDR_dB_prune3
    );

    gettimeofday(&time_Spatial_Spectrum3_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search3_start, NULL);
    int *position_theta_prune3 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune3 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune3, position_theta_prune3, position_phi_prune3, search_len_theta_prune3, search_len_phi_prune3, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune3, position_theta_prune3, position_phi_prune3, search_len_theta_prune3, search_len_phi_prune3, search_theta_deg_prune3, search_phi_deg_prune3, len_t_angle);
    gettimeofday(&time_Peak_Search3_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 4
    gettimeofday(&time_Spatial_Spectrum4_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune4 = NULL;
    float *search_theta_rad_prune4 = NULL;
    int search_len_theta_prune4;
    calculate_search_theta_3D(search_theta_deg_prune3[position_theta_prune3[0]], &search_len_theta_prune4, &search_theta_deg_prune4, &search_theta_rad_prune4, search_step_theta[4]);
    //printf("len_dth4 = %d\n", len_dth4);
    // Search angle phi
    float *search_phi_deg_prune4 = NULL;
    float *search_phi_rad_prune4 = NULL;
    int search_len_phi_prune4;
    calculate_search_phi_3D(search_phi_deg_prune3[position_phi_prune3[0]], &search_len_phi_prune4, &search_phi_deg_prune4, &search_phi_rad_prune4, search_step_phi[4]);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune4 = (float *)malloc(search_len_theta_prune4*search_len_phi_prune4 * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune4, 
        search_len_phi_prune4, 
        search_theta_rad_prune4, 
        search_phi_rad_prune4,
        S_MVDR_dB_prune4
    );

    gettimeofday(&time_Spatial_Spectrum4_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search4_start, NULL);
    int *position_theta_prune4 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune4 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune4, position_theta_prune4, position_phi_prune4, search_len_theta_prune4, search_len_phi_prune4, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune4, position_theta_prune4, position_phi_prune4, search_len_theta_prune4, search_len_phi_prune4, search_theta_deg_prune4, search_phi_deg_prune4, len_t_angle);
    gettimeofday(&time_Peak_Search4_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 5
    gettimeofday(&time_Spatial_Spectrum5_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune5 = NULL;
    float *search_theta_rad_prune5 = NULL;
    int search_len_theta_prune5;
    calculate_search_theta_3D(search_theta_deg_prune4[position_theta_prune4[0]], &search_len_theta_prune5, &search_theta_deg_prune5, &search_theta_rad_prune5, search_step_theta[5]);
    //printf("len_dth5 = %d\n", len_dth5);
    // Search angle phi
    float *search_phi_deg_prune5 = NULL;
    float *search_phi_rad_prune5 = NULL;
    int search_len_phi_prune5;
    calculate_search_phi_3D(search_phi_deg_prune4[position_phi_prune4[0]], &search_len_phi_prune5, &search_phi_deg_prune5, &search_phi_rad_prune5, search_step_phi[5]);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB_prune5 = (float *)malloc(search_len_theta_prune5*search_len_phi_prune5 * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune5, 
        search_len_phi_prune5, 
        search_theta_rad_prune5, 
        search_phi_rad_prune5,
        S_MVDR_dB_prune5
    );

    gettimeofday(&time_Spatial_Spectrum5_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search5_start, NULL);
    int *position_theta_prune5 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune5 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune5, position_theta_prune5, position_phi_prune5, search_len_theta_prune5, search_len_phi_prune5, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MVDR_dB_prune5, position_theta_prune5, position_phi_prune5, search_len_theta_prune5, search_len_phi_prune5, search_theta_deg_prune5, search_phi_deg_prune5, len_t_angle);
    gettimeofday(&time_Peak_Search5_end, NULL);
    //printf("---\n");
    gettimeofday(&time_MVDR_end, NULL);
    // ==================================================================
    // ====================== MVDR algorithm end ========================
    // ==================================================================

    //-------------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Rxx_inv, time_subRxx, time_Spatial_Spectrum, time_Peak_Search,time_MVDR;                           // create float parameter in order to convert (us) to (ms)
    float time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4, time_Spatial_Spectrum5;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4, time_Peak_Search5;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Rxx_inv_end, &time_Rxx_inv_start, &time_Rxx_inv_diff);                    // calculate Pn
    timersub(&time_subRxx_end, &time_subRxx_start, &time_subRxx_diff);                    // calculate sub Rxx
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
    time_Rxx_inv = time_Rxx_inv_diff.tv_sec * 1000000 + time_Rxx_inv_diff.tv_usec;
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
    total_time_Spatial_Spectrum = time_Spatial_Spectrum+time_Spatial_Spectrum1+time_Spatial_Spectrum2+time_Spatial_Spectrum3+time_Spatial_Spectrum4+time_Spatial_Spectrum5;
    total_time_Peak_Search = time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4+time_Peak_Search5;
    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MVDR DOA-----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-----Estimated results------\n");
    printf("position0 theta : \t%d\n", position_theta_prune0[0]);
    printf("position0 phi : \t%d\n", position_phi_prune0[0]);
    printf(RED "Theta estimation0 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune0[position_theta_prune0[0]], search_phi_deg_prune0[position_phi_prune0[0]]);
    //printf("Max_theta0 :\t\t%f(dB)\n", max_temp);
    printf("\n");
    printf("position1 theta : \t%d\n", position_theta_prune1[0]);
    printf("position1 phi : \t%d\n", position_phi_prune1[0]);
    printf(RED "Theta estimation1 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune1[position_theta_prune1[0]], search_phi_deg_prune1[position_phi_prune1[0]]);
    //printf("Max_theta1 :\t\t%f(dB)\n", max_temp1);
    printf("\n");
    printf("position2 theta : \t%d\n", position_theta_prune2[0]);
    printf("position2 phi : \t%d\n", position_phi_prune2[0]);
    printf(RED "Theta estimation2 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune2[position_theta_prune2[0]], search_phi_deg_prune2[position_phi_prune2[0]]);
    //printf("Max_theta2 :\t\t%f(dB)\n", max_temp2);
    printf("\n");
    printf("position3 theta : \t%d\n", position_theta_prune3[0]);
    printf("position3 phi : \t%d\n", position_phi_prune3[0]);
    printf(RED "Theta estimation3 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune3[position_theta_prune3[0]], search_phi_deg_prune3[position_phi_prune3[0]]);
    //printf("Max_theta3 :\t\t%f(dB)\n", max_temp3);
    printf("\n");
    printf("position4 theta : \t%d\n", position_theta_prune4[0]);
    printf("position4 phi : \t%d\n", position_phi_prune4[0]);
    printf(RED "Theta estimation4 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune4[position_theta_prune4[0]], search_phi_deg_prune4[position_phi_prune4[0]]);
    //printf("Max_theta4 :\t\t%f(dB)\n", max_temp4);
    printf("\n");
    printf("position5 theta : \t%d\n", position_theta_prune5[0]);
    printf("position5 phi : \t%d\n", position_phi_prune5[0]);
    printf(RED "Theta estimation5 :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg_prune5[position_theta_prune5[0]], search_phi_deg_prune5[position_phi_prune5[0]]);
    //printf("Max_theta5 :\t\t%f(dB)\n", max_temp5);

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
    printf("----> Spatial Spectrum5 time: \t%.3f(ms)\n", time_Spatial_Spectrum5 / 1000);
    printf("----> Peak Search5 time: \t%.3f(ms)\n", time_Peak_Search5 / 1000);
    printf(L_GREEN "Total MVDR REAL time : \t\t%.3f(ms)\n" CLOSE, time_MVDR / 1000);

    //-------------------------------------------------------------------
    // free memory
    free(R_xx_re);
    free(R_xx_im);
    free(Pn_re);
    free(Pn_im);
    free(search_const_int32);
    free(search_start_theta);
    free(search_start_phi);
    free(search_theta_deg_prune0);
    free(search_theta_rad_prune0);
    free(search_phi_deg_prune0);
    free(search_phi_rad_prune0);
    free(S_MVDR_dB_prune0);
    free(position_theta_prune0);
    free(position_phi_prune0);
    free(search_theta_deg_prune1);
    free(search_theta_rad_prune1);
    free(search_phi_deg_prune1);
    free(search_phi_rad_prune1);
    free(S_MVDR_dB_prune1);
    free(position_theta_prune1);
    free(position_phi_prune1);
    free(search_theta_deg_prune2);
    free(search_theta_rad_prune2);
    free(search_phi_deg_prune2);
    free(search_phi_rad_prune2);
    free(S_MVDR_dB_prune2);
    free(position_theta_prune2);
    free(position_phi_prune2);
    free(search_theta_deg_prune3);
    free(search_theta_rad_prune3);
    free(search_phi_deg_prune3);
    free(search_phi_rad_prune3);
    free(S_MVDR_dB_prune3);
    free(position_theta_prune3);
    free(position_phi_prune3);
    free(search_theta_deg_prune4);
    free(search_theta_rad_prune4);
    free(search_phi_deg_prune4);
    free(search_phi_rad_prune4);
    free(S_MVDR_dB_prune4);
    free(position_theta_prune4);
    free(position_phi_prune4);
    free(search_theta_deg_prune5);
    free(search_theta_rad_prune5);
    free(search_phi_deg_prune5);
    free(search_phi_rad_prune5);
    free(S_MVDR_dB_prune5);
    free(position_theta_prune5);
    free(position_phi_prune5);
}

int main()
{
    // Bind main thread to core
    bind_main_to_core(2);
    // Initialize thread pool
    int threads_number = 8; // Number of threads
    num_row_blocks = threads_number; // Number of row blocks for parallel matrix multiplication
    init_thread_pool(threads_number, 3);
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff;
    //-------------------------------------------------------------------
    // Physical Parameters initialization
    PhysicalParameters phys;
    phys.fc = 28e9;
    phys.c = 3e8;
    phys.lambda = (float)phys.c / (float)phys.fc;
    phys.kc = 2 * PI / phys.lambda;
    phys.snr = 20;

    int M = 24;
    float angle_theta[100] = { 15, 70, 20.0, 80.0, 70, 60, 58}; // elevation
    float angle_phi[100]   = { 15, 20, -45.0, 45.0, 0, 45, 58}; // azimuth
    // Tx Parameters initialization
    TxParameters tx;
    tx.Tx_M_x = M;
    tx.Tx_M_y = M;
    tx.Tx_beamwidth = 0.1;
    tx.Tx_beamwidth_samples = 100;
    tx.angle_theta = angle_theta;
    tx.angle_phi = angle_phi;
    tx.number_angle = 1;

    // Rx Parameters initialization
    RxParameters rx;
    rx.Rx_M_x = M;
    rx.Rx_M_y = M;       
    rx.d = phys.lambda * 0.5f;
    rx.nd = 1024;

    //-------------------------------------------------------------------
    gettimeofday(&time_MVDR_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    //=================== MVDR Algorithm ==================================
    m_doa3d_mvdr_lu_pns_parMul_cpu_float_int32_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

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