// 3D DOA Estimation - ML Algorithm (float)
// Fast Search Implementation using Prune-and-Search Strategy
//--------------------
#define PI acos(-1)
#define AVX 16            
//--------------------
#include <immintrin.h>
#include "../common/complex_matrix_ops.h"
#include "../common/generate_signal_3D.h"
#include "../common/eigen_qr.h"
#include "../common/lu_decomp.h"
#include "../common/spatial_spectrum.h"
#include "../common/doa_parameters.h"
#include <sys/syscall.h>
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
void doa3d_ml_prunesearch_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx){
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_ML_start, time_ML_end, time_ML_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;

    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;
    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;

    struct timeval time_Spatial_Spectrum1_start, time_Spatial_Spectrum1_end, time_Spatial_Spectrum1_diff;
    struct timeval time_Peak_Search1_start, time_Peak_Search1_end, time_Peak_Search1_diff;

    struct timeval time_Spatial_Spectrum2_start, time_Spatial_Spectrum2_end, time_Spatial_Spectrum2_diff;
    struct timeval time_Peak_Search2_start, time_Peak_Search2_end, time_Peak_Search2_diff;

    struct timeval time_Spatial_Spectrum3_start, time_Spatial_Spectrum3_end, time_Spatial_Spectrum3_diff;
    struct timeval time_Peak_Search3_start, time_Peak_Search3_end, time_Peak_Search3_diff;

    struct timeval time_Spatial_Spectrum4_start, time_Spatial_Spectrum4_end, time_Spatial_Spectrum4_diff;
    struct timeval time_Peak_Search4_start, time_Peak_Search4_end, time_Peak_Search4_diff;

    struct timeval time_Spatial_Spectrum5_start, time_Spatial_Spectrum5_end, time_Spatial_Spectrum5_diff;
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
    // ====================== ML algorithm start ========================
    // ==================================================================
    gettimeofday(&time_ML_start, NULL);
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
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    // parameter setting
    SearchConst *search_const = (SearchConst*)malloc(sizeof(SearchConst));
    search_const->Rx_M_x = Rx_M_x;
    search_const->Rx_M_y = Rx_M_y;
    search_const->Rx_M = Rx_M;
    search_const->d = d;
    search_const->kc = kc;
    search_const->Pn_re = R_xx_re;
    search_const->Pn_im = R_xx_im;
    float search_step_theta[6] = {30, 10, 3, 1, 0.3, 0.1};
    float search_step_phi[6] = {30, 10, 3, 1, 0.3, 0.1};
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float *search_start_phi = (float *)malloc(4 * sizeof(float));
    //---------------------------------------------------------------
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
    float *S_ML_dB_prune0 = (float *)malloc(search_len_theta_prune0*search_len_phi_prune0 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune0, 
        search_len_phi_prune0, 
        search_theta_rad_prune0, 
        search_phi_rad_prune0,
        S_ML_dB_prune0
    );
    //printf("---\n");
    gettimeofday(&time_Spatial_Spectrum_end, NULL);
    // find peaks
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta_prune0 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune0 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune0, position_theta_prune0, position_phi_prune0, search_len_theta_prune0, search_len_phi_prune0, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 1
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
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    float *S_ML_dB_prune1 = (float *)malloc(search_len_theta_prune1*search_len_phi_prune1 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune1, 
        search_len_phi_prune1, 
        search_theta_rad_prune1, 
        search_phi_rad_prune1,
        S_ML_dB_prune1
    );
    gettimeofday(&time_Spatial_Spectrum1_end, NULL);
    // find peaks
    gettimeofday(&time_Peak_Search1_start, NULL);
    int *position_theta_prune1 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune1 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune1, position_theta_prune1, position_phi_prune1, search_len_theta_prune1, search_len_phi_prune1, len_t_angle);
    gettimeofday(&time_Peak_Search1_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 2
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
    gettimeofday(&time_Spatial_Spectrum2_start, NULL);
    float *S_ML_dB_prune2 = (float *)malloc(search_len_theta_prune2*search_len_phi_prune2 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune2, 
        search_len_phi_prune2, 
        search_theta_rad_prune2, 
        search_phi_rad_prune2,
        S_ML_dB_prune2
    );
    gettimeofday(&time_Spatial_Spectrum2_end, NULL);
    // find peaks
    gettimeofday(&time_Peak_Search2_start, NULL);
    int *position_theta_prune2 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune2 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune2, position_theta_prune2, position_phi_prune2, search_len_theta_prune2, search_len_phi_prune2, len_t_angle);
    gettimeofday(&time_Peak_Search2_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 3
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

    gettimeofday(&time_Spatial_Spectrum3_start, NULL);
    // Calculate Spatial Spectrum and Peak Search
    float *S_ML_dB_prune3 = (float *)malloc(search_len_theta_prune3*search_len_phi_prune3 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune3, 
        search_len_phi_prune3, 
        search_theta_rad_prune3, 
        search_phi_rad_prune3,
        S_ML_dB_prune3
    );
    gettimeofday(&time_Spatial_Spectrum3_end, NULL);
    // find peaks
    gettimeofday(&time_Peak_Search3_start, NULL);
    int *position_theta_prune3 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune3 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune3, position_theta_prune3, position_phi_prune3, search_len_theta_prune3, search_len_phi_prune3, len_t_angle);
    gettimeofday(&time_Peak_Search3_end, NULL);
    //printf("---\n");
    
    //---------------------------------------------------------------
    // parameter setting 4
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

    gettimeofday(&time_Spatial_Spectrum4_start, NULL);
    // Calculate Spatial Spectrum and Peak Search
    float *S_ML_dB_prune4 = (float *)malloc(search_len_theta_prune4*search_len_phi_prune4 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune4, 
        search_len_phi_prune4, 
        search_theta_rad_prune4, 
        search_phi_rad_prune4,
        S_ML_dB_prune4
    );
    gettimeofday(&time_Spatial_Spectrum4_end, NULL);
    // find peaks
    gettimeofday(&time_Peak_Search4_start, NULL);
    int *position_theta_prune4 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune4 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune4, position_theta_prune4, position_phi_prune4, search_len_theta_prune4, search_len_phi_prune4, len_t_angle);
    gettimeofday(&time_Peak_Search4_end, NULL);
    //printf("---\n");
    
    //---------------------------------------------------------------
    // parameter setting 5
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

    gettimeofday(&time_Spatial_Spectrum5_start, NULL);
    // Calculate Spatial Spectrum and Peak Search
    float *S_ML_dB_prune5 = (float *)malloc(search_len_theta_prune5*search_len_phi_prune5 * sizeof(float));
    calculate_spatial_spectrum_ML_3D(
        search_const,
        search_len_theta_prune5, 
        search_len_phi_prune5, 
        search_theta_rad_prune5, 
        search_phi_rad_prune5,
        S_ML_dB_prune5
    );
    gettimeofday(&time_Spatial_Spectrum5_end, NULL);

    // find peaks
    gettimeofday(&time_Peak_Search5_start, NULL);
    int *position_theta_prune5 = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi_prune5 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_ML_dB_prune5, position_theta_prune5, position_phi_prune5, search_len_theta_prune5, search_len_phi_prune5, len_t_angle);
    gettimeofday(&time_Peak_Search5_end, NULL);
    //printf("---\n");
    gettimeofday(&time_ML_end, NULL);
    
    // ==================================================================
    // ======================= ML algorithm end =========================
    // ==================================================================
    float time_Rxx, time_ML ,time_Peak_Search, time_total_Spatial_Spectrum,  time_total_Peak_Search;
    float time_Spatial_Spectrum, time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4, time_Spatial_Spectrum5;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4, time_Peak_Search5;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Spatial_Spectrum_end, &time_Spatial_Spectrum_start, &time_Spatial_Spectrum_diff);
    timersub(&time_Spatial_Spectrum1_end, &time_Spatial_Spectrum1_start, &time_Spatial_Spectrum1_diff);
    timersub(&time_Spatial_Spectrum2_end, &time_Spatial_Spectrum2_start, &time_Spatial_Spectrum2_diff);
    timersub(&time_Spatial_Spectrum3_end, &time_Spatial_Spectrum3_start, &time_Spatial_Spectrum3_diff);
    timersub(&time_Spatial_Spectrum4_end, &time_Spatial_Spectrum4_start, &time_Spatial_Spectrum4_diff);
    timersub(&time_Spatial_Spectrum5_end, &time_Spatial_Spectrum5_start, &time_Spatial_Spectrum5_diff);

    timersub(&time_Peak_Search_end, &time_Peak_Search_start, &time_Peak_Search_diff);
    timersub(&time_Peak_Search1_end, &time_Peak_Search1_start, &time_Peak_Search1_diff);
    timersub(&time_Peak_Search2_end, &time_Peak_Search2_start, &time_Peak_Search2_diff);
    timersub(&time_Peak_Search3_end, &time_Peak_Search3_start, &time_Peak_Search3_diff); 
    timersub(&time_Peak_Search4_end, &time_Peak_Search4_start, &time_Peak_Search4_diff); 
    timersub(&time_Peak_Search5_end, &time_Peak_Search5_start, &time_Peak_Search5_diff); 
    timersub(&time_ML_end, &time_ML_start, &time_ML_diff);

    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Spatial_Spectrum = time_Spatial_Spectrum_diff.tv_sec * 1000000 + time_Spatial_Spectrum_diff.tv_usec;
    time_Spatial_Spectrum1 = time_Spatial_Spectrum1_diff.tv_sec * 1000000 + time_Spatial_Spectrum1_diff.tv_usec;
    time_Spatial_Spectrum2 = time_Spatial_Spectrum2_diff.tv_sec * 1000000 + time_Spatial_Spectrum2_diff.tv_usec;
    time_Spatial_Spectrum3 = time_Spatial_Spectrum3_diff.tv_sec * 1000000 + time_Spatial_Spectrum3_diff.tv_usec;
    time_Spatial_Spectrum4 = time_Spatial_Spectrum4_diff.tv_sec * 1000000 + time_Spatial_Spectrum4_diff.tv_usec;
    time_Spatial_Spectrum5 = time_Spatial_Spectrum5_diff.tv_sec * 1000000 + time_Spatial_Spectrum5_diff.tv_usec;
    time_total_Spatial_Spectrum = time_Spatial_Spectrum + time_Spatial_Spectrum1 + time_Spatial_Spectrum2 + time_Spatial_Spectrum3 + time_Spatial_Spectrum4 + time_Spatial_Spectrum5;

    time_Peak_Search = time_Peak_Search_diff.tv_sec * 1000000 + time_Peak_Search_diff.tv_usec;
    time_Peak_Search1 = time_Peak_Search1_diff.tv_sec * 1000000 + time_Peak_Search1_diff.tv_usec;
    time_Peak_Search2 = time_Peak_Search2_diff.tv_sec * 1000000 + time_Peak_Search2_diff.tv_usec;
    time_Peak_Search3 = time_Peak_Search3_diff.tv_sec * 1000000 + time_Peak_Search3_diff.tv_usec;
    time_Peak_Search4 = time_Peak_Search4_diff.tv_sec * 1000000 + time_Peak_Search4_diff.tv_usec;
    time_Peak_Search5 = time_Peak_Search5_diff.tv_sec * 1000000 + time_Peak_Search5_diff.tv_usec;
    time_total_Peak_Search = time_Peak_Search + time_Peak_Search1 + time_Peak_Search2 + time_Peak_Search3 + time_Peak_Search4 + time_Peak_Search5;
    time_ML = time_ML_diff.tv_sec * 1000000 + time_ML_diff.tv_usec;

    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("-----------------ML DOA------------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("Search step theta:\t%.2f, %.2f, %.2f, %.2f, %.2f, %.2f (degree)\n", 
        search_step_theta[0], search_step_theta[1], search_step_theta[2], search_step_theta[3], search_step_theta[4], search_step_theta[5]);
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
    printf("Total Spatial Spectrum time: \t%.3f(ms)\n", time_total_Spatial_Spectrum / 1000);
    printf("Total Peak Search time: \t%.3f(ms)\n", time_total_Peak_Search / 1000);
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
    printf(L_GREEN "Total ML REAL time :\t\t%.3f(ms)\n" CLOSE, time_ML / 1000);
    printf(L_GREEN "->Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //-------------------------------------------------------------------
    // free memory
    free(R_xx_re);
    free(R_xx_im);
    free(search_start_theta);
    free(search_start_phi);
    free(search_theta_deg_prune0);
    free(search_theta_rad_prune0);
    free(search_phi_deg_prune0);
    free(search_phi_rad_prune0);
    free(S_ML_dB_prune0);
    free(position_theta_prune0);
    free(position_phi_prune0);

    // Prune 1
    free(search_theta_deg_prune1);
    free(search_theta_rad_prune1);
    free(search_phi_deg_prune1);
    free(search_phi_rad_prune1);
    free(S_ML_dB_prune1);
    free(position_theta_prune1);
    free(position_phi_prune1);

    // Prune 2
    free(search_theta_deg_prune2);
    free(search_theta_rad_prune2);
    free(search_phi_deg_prune2);
    free(search_phi_rad_prune2);
    free(S_ML_dB_prune2);
    free(position_theta_prune2);
    free(position_phi_prune2);

    // Prune 3
    free(search_theta_deg_prune3);
    free(search_theta_rad_prune3);
    free(search_phi_deg_prune3);
    free(search_phi_rad_prune3);
    free(S_ML_dB_prune3);
    free(position_theta_prune3);
    free(position_phi_prune3);

    // Prune 4
    free(search_theta_deg_prune4);
    free(search_theta_rad_prune4);
    free(search_phi_deg_prune4);
    free(search_phi_rad_prune4);
    free(S_ML_dB_prune4);
    free(position_theta_prune4);
    free(position_phi_prune4);

    // Prune 5
    free(search_theta_deg_prune5);
    free(search_theta_rad_prune5);
    free(search_phi_deg_prune5);
    free(search_phi_rad_prune5);
    free(S_ML_dB_prune5);
    free(position_theta_prune5);
    free(position_phi_prune5);

    free(search_const);
}

int main()
{
    struct timeval time_ML_start, time_ML_end, time_ML_diff;
    //-------------------------------------------------------------------
    // Physical Parameters initialization
    PhysicalParameters phys;
    phys.fc = 28e9;
    phys.c = 3e8;
    phys.lambda = (float)phys.c / (float)phys.fc;
    phys.kc = 2 * PI / phys.lambda;
    phys.snr = 10;

    int M = 8;
    float angle_theta[100] = { 45.0, 45, 45.0, 15, 70, 45, 58}; // elevation
    float angle_phi[100]    = { 0.0, 45, -45.0, -15.0, 0, 45, 58}; // azimuth
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

    
    //-------------------------------------------------------------------
    gettimeofday(&time_ML_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x * rx.Rx_M_y * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    //=================== ML Algorithm =================================
    doa3d_ml_prunesearch_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_ML_end, NULL);
    timersub(&time_ML_end, &time_ML_start, &time_ML_diff);
    float time_ML;
    time_ML = time_ML_diff.tv_sec * 1000000 + time_ML_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total ML time : \t\t%.3f(ms)\n" CLOSE, time_ML / 1000);

    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}