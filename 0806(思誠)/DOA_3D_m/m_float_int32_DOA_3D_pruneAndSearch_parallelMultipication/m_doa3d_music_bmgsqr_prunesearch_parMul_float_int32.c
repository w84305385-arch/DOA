// 3D DOA Estimation - MUSIC Algorithm with BMGS QR Decomposition
// Mixed Precision (float32 / int32)
// Fast Search Implementation using Prune-and-Search Strategy (Multi-User Version)
// All Modules Use Parallel Matrix Multiplication (Search Stage Excluded)
//--------------------
#define PI acos(-1)
#define AVX 16
//--------------------
#include <immintrin.h>
#include "../common/q_format_config.h"
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
#include "../common/m_eigen_qr.h"
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
#include <pthread.h>
#include <sys/syscall.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//----------------------global variable---------------------------
float total_multiply_time = 0;
int num_row_blocks = 1; // number of row blocks for parallel matrix multiplication
//----------------------------------------------------------------
void m_doa3d_music_bmgsqr_pns_parMul_cpu_float_int32_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff;
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;

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
    // ===================== MUSIC algorithm start ======================
    // ==================================================================
    // Time parameter initialization
    gettimeofday(&time_MUSIC_start, NULL);
    total_multiply_time = 0.0;
    int Rx_M;
    Rx_M = Rx_M_x*Rx_M_y;
    //---------------------------------------------------------------
    gettimeofday(&time_Rxx_start, NULL);
    float *R_xx_re = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *R_xx_im = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
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
    gettimeofday(&time_Eigen_start, NULL);
    float *Ve_re = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *Ve_im = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *De_re = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *De_im = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX * sizeof(float));
    float *BMGS_qr_time = (float *)malloc(1 * sizeof(float) + AVX * sizeof(float));
    float *qr_time = (float *)malloc(1 * sizeof(float) + AVX * sizeof(float));
    // Compute eigenvector Ve (M, M), eigenvalue De(M, M)
    m_eigen_BMGS(R_xx_re, R_xx_im, Ve_re, Ve_im, De_re, De_im, Rx_M, Rx_M, qr_iter, BMGS_qr_num_blocks, BMGS_qr_time, qr_time);
    gettimeofday(&time_Eigen_end, NULL);
    //printf("----------Ve------------\n");
    //print_complex_matrix(Ve_re, Ve_im, Rx_M, Rx_M);
    //printf("----------De------------\n");
    //print_complex_matrix(De_re, De_im, Rx_M, Rx_M);

    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    float *vet_noise_re = (float *)malloc(Rx_M * (Rx_M - len_t_angle) * sizeof(float) + AVX * sizeof(float));
    float *vet_noise_im = (float *)malloc(Rx_M * (Rx_M - len_t_angle) * sizeof(float) + AVX * sizeof(float));
    // Extract noise subspace
    for (int i = 0; i < Rx_M; ++i)
    {
        for (int j = len_t_angle; j < Rx_M; ++j)
        {
            vet_noise_re[i * (Rx_M - len_t_angle) + j - len_t_angle] = Ve_re[i * Rx_M + j];
            vet_noise_im[i * (Rx_M - len_t_angle) + j - len_t_angle] = Ve_im[i * Rx_M + j];
            //printf("\t(%f,%f)\n", Ve_re[i * M + j], Ve_im[i * M + j]);
        }
    }
    //printf("----------noise subspace------------\n");
    //print_complex_matrix(vet_noise_re, vet_noise_im, Rx_M, (Rx_M - len_t_angle));

    //---------------------------------------------------------------
    float *Pn_re = (float *)malloc(Rx_M * Rx_M* sizeof(float) + AVX * sizeof(float));
    float *Pn_im = (float *)malloc(Rx_M * Rx_M* sizeof(float) + AVX * sizeof(float));
    // Compute Noise subspace
    m_complex_matrix_conjugate_transpose_multiplication_row_block_parallel(vet_noise_re, vet_noise_im, Pn_re, Pn_im, Rx_M,  Rx_M - len_t_angle, num_row_blocks);
    gettimeofday(&time_Pn_end, NULL);
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, Rx_M, Rx_M);
    
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
    float *search_start_theta = (float *)malloc(4 * sizeof(float) + AVX * sizeof(float));
    float *search_start_phi = (float *)malloc(4 * sizeof(float) + AVX * sizeof(float));
    //---------------------------------------------------------------
    // parameter setting
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta 
    search_start_theta[0] = 0;
    int search_len_theta_prune0 = 4;
    float *search_theta_deg_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float) + AVX * sizeof(float));
    float *search_theta_rad_prune0 = (float *)malloc(search_len_theta_prune0 * sizeof(float) + AVX * sizeof(float));
    for (int i = 0; i < search_len_theta_prune0-1; ++i){ 
        search_theta_deg_prune0[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad_prune0[i] = search_theta_deg_prune0[i] * PI / 180;
    }
    search_theta_deg_prune0[search_len_theta_prune0-1] = search_theta_deg_prune0[search_len_theta_prune0-2] + search_step_theta[0]/2;
    search_theta_rad_prune0[search_len_theta_prune0-1] = search_theta_deg_prune0[search_len_theta_prune0-1] * PI / 180;

    // Search angle phi
    search_start_phi[0] = -60;
    int search_len_phi_prune0 = 5;
    float *search_phi_deg_prune0 = (float *)malloc(search_len_phi_prune0 * sizeof(float) + AVX * sizeof(float));
    float *search_phi_rad_prune0 = (float *)malloc(search_len_phi_prune0 * sizeof(float) + AVX * sizeof(float));
    for (int i = 0; i < search_len_phi_prune0; ++i){ 
        search_phi_deg_prune0[i] = search_start_phi[0] + search_step_phi[0] * i;
        search_phi_rad_prune0[i] = search_phi_deg_prune0[i] * PI / 180;
    }
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB_prune0 = (float *)malloc(search_len_theta_prune0*search_len_phi_prune0 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune0, 
        search_len_phi_prune0, 
        search_theta_rad_prune0, 
        search_phi_rad_prune0,
        S_MUSIC_dB_prune0, 
        search_step_theta[0]
    );  
    gettimeofday(&time_Spatial_Spectrum_end, NULL);
    //printf("---\n");
    // find peaks
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta_prune0 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune0 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune0, position_theta_prune0, position_phi_prune0, search_len_theta_prune0, search_len_phi_prune0, len_t_angle);
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
    float *S_MUSIC_dB_prune1 = (float *)malloc(search_len_theta_prune1*search_len_phi_prune1 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune1, 
        search_len_phi_prune1, 
        search_theta_rad_prune1, 
        search_phi_rad_prune1,
        S_MUSIC_dB_prune1, 
        search_step_theta[1]
    );  
    gettimeofday(&time_Spatial_Spectrum1_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search1_start, NULL);
    int *position_theta_prune1 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune1 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune1, position_theta_prune1, position_phi_prune1, search_len_theta_prune1, search_len_phi_prune1, len_t_angle);
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
    float *S_MUSIC_dB_prune2 = (float *)malloc(search_len_theta_prune2*search_len_phi_prune2 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune2, 
        search_len_phi_prune2, 
        search_theta_rad_prune2, 
        search_phi_rad_prune2,
        S_MUSIC_dB_prune2
    );  

    gettimeofday(&time_Spatial_Spectrum2_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search2_start, NULL);
    int *position_theta_prune2 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune2 = (int *)malloc(25 * sizeof(int));
    position_theta_prune2[0] = 0;
    position_phi_prune2[0] = 0; 
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune2, position_theta_prune2, position_phi_prune2, search_len_theta_prune2, search_len_phi_prune2, len_t_angle);
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
    float *S_MUSIC_dB_prune3 = (float *)malloc(search_len_theta_prune3*search_len_phi_prune3 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune3, 
        search_len_phi_prune3, 
        search_theta_rad_prune3, 
        search_phi_rad_prune3,
        S_MUSIC_dB_prune3
    );

    gettimeofday(&time_Spatial_Spectrum3_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search3_start, NULL);
    int *position_theta_prune3 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune3 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune3, position_theta_prune3, position_phi_prune3, search_len_theta_prune3, search_len_phi_prune3, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune3, position_theta_prune3, position_phi_prune3, search_len_theta_prune3, search_len_phi_prune3, search_theta_deg_prune3, search_phi_deg_prune3, len_t_angle);
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
    float *S_MUSIC_dB_prune4 = (float *)malloc(search_len_theta_prune4*search_len_phi_prune4 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune4, 
        search_len_phi_prune4, 
        search_theta_rad_prune4, 
        search_phi_rad_prune4,
        S_MUSIC_dB_prune4
    );

    gettimeofday(&time_Spatial_Spectrum4_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search4_start, NULL);
    int *position_theta_prune4 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune4 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune4, position_theta_prune4, position_phi_prune4, search_len_theta_prune4, search_len_phi_prune4, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune4, position_theta_prune4, position_phi_prune4, search_len_theta_prune4, search_len_phi_prune4, search_theta_deg_prune4, search_phi_deg_prune4, len_t_angle);
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
    float *S_MUSIC_dB_prune5 = (float *)malloc(search_len_theta_prune5*search_len_phi_prune5 * sizeof(float) + AVX * sizeof(float));
    m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
        search_const_int32, 
        search_len_theta_prune5, 
        search_len_phi_prune5, 
        search_theta_rad_prune5, 
        search_phi_rad_prune5,
        S_MUSIC_dB_prune5
    );

    gettimeofday(&time_Spatial_Spectrum5_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search5_start, NULL);
    int *position_theta_prune5 = (int *)malloc(25 * sizeof(int));
    int *position_phi_prune5 = (int *)malloc(25 * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune5, position_theta_prune5, position_phi_prune5, search_len_theta_prune5, search_len_phi_prune5, len_t_angle);
    //find_spatial_spectrum_peaks_3D(S_MUSIC_dB_prune5, position_theta_prune5, position_phi_prune5, search_len_theta_prune5, search_len_phi_prune5, search_theta_deg_prune5, search_phi_deg_prune5, len_t_angle);
    gettimeofday(&time_Peak_Search5_end, NULL);
    //printf("---\n");

    gettimeofday(&time_MUSIC_end, NULL);
    // ==================================================================
    // ====================== MUSIC algorithm end =======================
    // ==================================================================
    
    //---------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Eigen, time_Pn, time_Spatial_Spectrum, time_Peak_Search, time_MUSIC;      // create float parameter in order to convert (us) to (ms)
    float time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4, time_Spatial_Spectrum5;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4, time_Peak_Search5;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Eigen_end, &time_Eigen_start, &time_Eigen_diff);           // calculate Eigen
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);                    // calculate Pn
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

    timersub(&time_MUSIC_end, &time_MUSIC_start, &time_MUSIC_diff);                    // calculate MUSIC

    // Compute time
    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Eigen = time_Eigen_diff.tv_sec * 1000000 + time_Eigen_diff.tv_usec;
    time_Pn = time_Pn_diff.tv_sec * 1000000 + time_Pn_diff.tv_usec;
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

    time_MUSIC = time_MUSIC_diff.tv_sec * 1000000 + time_MUSIC_diff.tv_usec;;
    float total_time_Spatial_Spectrum, total_time_Peak_Search;
    total_time_Spatial_Spectrum = time_Spatial_Spectrum+time_Spatial_Spectrum1+time_Spatial_Spectrum2+time_Spatial_Spectrum3+time_Spatial_Spectrum4+time_Spatial_Spectrum5;
    total_time_Peak_Search = time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4+time_Peak_Search5;

    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MUSIC DOA----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("BMGS-QR num of blocks:\t%d\n", BMGS_qr_num_blocks);
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
    printf("\n");
    printf("Total search count: \t%d\n", search_count);

    printf("\n\t------------Time------------\n");
    printf("\t  -------DOA start-------\n");
    printf("Total Rxx time: \t\t%.3f(ms)\n", time_Rxx / 1000);
    printf("BMGS QR time: \t\t\t%.3f(ms)\n", *BMGS_qr_time / 1000);
    printf("---> QR time: \t\t\t%.3f(ms)\n", *qr_time / 1000);
    printf("---> synchronous time: \t\t%.3f(ms)\n", (*BMGS_qr_time / 1000) - (*qr_time / 1000));
    printf("Total Eigen time: \t\t%.3f(ms)\n", time_Eigen / 1000 - (*BMGS_qr_time / 1000));

    printf("Total Pn time: \t\t\t%.3f(ms)\n", time_Pn / 1000);
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
    printf(L_GREEN "Total MUSIC REAL time : \t%.3f(ms)\n" CLOSE, time_MUSIC / 1000);
    
    //-------------------------------------------------------------------
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune0.csv", S_MUSIC_dB_prune0, search_len_theta_prune0*search_len_phi_prune0);
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune1.csv", S_MUSIC_dB_prune1, search_len_theta_prune1*search_len_phi_prune1);
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune2.csv", S_MUSIC_dB_prune2, search_len_theta_prune2*search_len_phi_prune2);
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune3.csv", S_MUSIC_dB_prune3, search_len_theta_prune3*search_len_phi_prune3);
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune4.csv", S_MUSIC_dB_prune4, search_len_theta_prune4*search_len_phi_prune4);
    save_Spectrum_to_csv("../float_DOA_3D_prune_and_search/MUSIC_spectrum_dB/S_MUSIC_dB_prune5.csv", S_MUSIC_dB_prune5, search_len_theta_prune5*search_len_phi_prune5);
    //-------------------------------------------------------------------
    // free
    free(R_xx_re);
    free(R_xx_im);
    free(Ve_re);
    free(Ve_im);
    free(De_re);
    free(De_im);
    free(BMGS_qr_time);
    free(qr_time);
    free(vet_noise_re);
    free(vet_noise_im);
    free(Pn_re);
    free(Pn_im);
    free(search_start_theta);
    free(search_start_phi);

    // Prune 0
    free(search_theta_deg_prune0);
    free(search_theta_rad_prune0);
    free(search_phi_deg_prune0);
    free(search_phi_rad_prune0);
    free(S_MUSIC_dB_prune0);
    free(position_theta_prune0);
    free(position_phi_prune0);

    // Prune 1
    free(search_theta_deg_prune1);
    free(search_theta_rad_prune1);
    free(search_phi_deg_prune1);
    free(search_phi_rad_prune1);
    free(S_MUSIC_dB_prune1);
    free(position_theta_prune1);
    free(position_phi_prune1);

    // Prune 2
    free(search_theta_deg_prune2);
    free(search_theta_rad_prune2);
    free(search_phi_deg_prune2);
    free(search_phi_rad_prune2);
    free(S_MUSIC_dB_prune2);
    free(position_theta_prune2);
    free(position_phi_prune2);

    // Prune 3
    free(search_theta_deg_prune3);
    free(search_theta_rad_prune3);
    free(search_phi_deg_prune3);
    free(search_phi_rad_prune3);
    free(S_MUSIC_dB_prune3);
    free(position_theta_prune3);
    free(position_phi_prune3);

    // Prune 4
    free(search_theta_deg_prune4);
    free(search_theta_rad_prune4);
    free(search_phi_deg_prune4);
    free(search_phi_rad_prune4);
    free(S_MUSIC_dB_prune4);
    free(position_theta_prune4);
    free(position_phi_prune4);

    // Prune 5
    free(search_theta_deg_prune5);
    free(search_theta_rad_prune5);
    free(search_phi_deg_prune5);
    free(search_phi_rad_prune5);
    free(S_MUSIC_dB_prune5);
    free(position_theta_prune5);
    free(position_phi_prune5);

    free(search_const_int32);
}

int main()
{
    // Bind main thread to core
    bind_main_to_core(2);
    // Initialize thread pool
    int threads_number = 8; // Number of threads
    num_row_blocks = threads_number; // Number of row blocks for parallel matrix multiplication
    init_thread_pool(threads_number, 3);
    // printf("Q_SHIFT = %d\n", Q_SHIFT);
    //-------------------------------------------------------------------
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff;
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
    tx.Tx_beamwidth_samples = 1;
    tx.angle_theta = angle_theta;
    tx.angle_phi = angle_phi;
    tx.number_angle = 1;

    // Rx Parameters initialization
    RxParameters rx;
    rx.Rx_M_x = M;
    rx.Rx_M_y = M;       
    rx.d = phys.lambda * 0.5f;
    rx.nd = 1024;
    rx.qr_iter = 2;
    rx.BMGS_qr_num_blocks = M;
    //-------------------------------------------------------------------
    gettimeofday(&time_MUSIC_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);
    //=================== MUSIC Algorithm =================================
    m_doa3d_music_bmgsqr_pns_parMul_cpu_float_int32_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_MUSIC_end, NULL);
    timersub(&time_MUSIC_end, &time_MUSIC_start, &time_MUSIC_diff);
    float time_MUSIC;
    time_MUSIC = time_MUSIC_diff.tv_sec * 1000000 + time_MUSIC_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MUSIC time : \t\t%.3f(ms)\n" CLOSE, time_MUSIC / 1000);

    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}

 