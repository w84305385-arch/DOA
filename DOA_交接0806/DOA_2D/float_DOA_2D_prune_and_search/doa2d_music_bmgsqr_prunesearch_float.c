// 2D DOA Estimation - MUSIC Algorithm with BMGS QR Decomposition (float)
// Fast Search Implementation using Prune-and-Search Strategy
// - Multi-beam coarse search for wide angular coverage
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
void doa2d_music_bmgsqr_prunesearch_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff; // time initial
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff; // time initial
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff; // time initial
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;          // time initial

    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;          // time initial
    struct timeval time_Spatial_Spectrum1_start, time_Spatial_Spectrum1_end, time_Spatial_Spectrum1_diff;          // time initial
    struct timeval time_Spatial_Spectrum2_start, time_Spatial_Spectrum2_end, time_Spatial_Spectrum2_diff;          // time initial
    struct timeval time_Spatial_Spectrum3_start, time_Spatial_Spectrum3_end, time_Spatial_Spectrum3_diff;          // time initial
    struct timeval time_Spatial_Spectrum4_start, time_Spatial_Spectrum4_end, time_Spatial_Spectrum4_diff;          // time initial

    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;          // time initial
    struct timeval time_Peak_Search1_start, time_Peak_Search1_end, time_Peak_Search1_diff;          // time initial
    struct timeval time_Peak_Search2_start, time_Peak_Search2_end, time_Peak_Search2_diff;          // time initial
    struct timeval time_Peak_Search3_start, time_Peak_Search3_end, time_Peak_Search3_diff;          // time initial
    struct timeval time_Peak_Search4_start, time_Peak_Search4_end, time_Peak_Search4_diff;          // time initial
    //-------------------------------------------------------------------
    // === Physical Parameters ===
    float kc = phys.kc;
    // === Tx Parameters ===
    int len_t_angle = tx.number_angle;
    // === Rx Parameters ===
    int Rx_M = rx.Rx_M_x;    
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
    //print_complex_matrix(Ve_re, Ve_im, Rx_M, Rx_M);
    //printf("----------De------------\n");
    //print_complex_matrix(De_re, De_im, Rx_M, Rx_M);

    //---------------------------------------------------------------
    gettimeofday(&time_Pn_start, NULL);
    float *vet_noise_re = (float *)malloc(Rx_M * (Rx_M - len_t_angle) * sizeof(float));
    float *vet_noise_im = (float *)malloc(Rx_M * (Rx_M - len_t_angle) * sizeof(float));
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
    float *Pn_re = (float *)malloc(Rx_M * Rx_M* sizeof(float));
    float *Pn_im = (float *)malloc(Rx_M * Rx_M* sizeof(float));
    // Compute Noise subspace
    complex_matrix_conjugate_transpose_multiplication(vet_noise_re, vet_noise_im, Pn_re, Pn_im, Rx_M,  Rx_M - len_t_angle);
    gettimeofday(&time_Pn_end, NULL);
    //printf("----------Pn------------\n");
    //print_complex_matrix(Pn_re, Pn_im, Rx_M, Rx_M);
    
    //---------------------------------------------------------------
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst *search_const = (SearchConst*)malloc(sizeof(SearchConst));
    search_const->Rx_M = Rx_M;
    search_const->d = d;
    search_const->kc = kc;
    search_const->Pn_re = Pn_re;
    search_const->Pn_im = Pn_im;
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float search_step_theta[6] = {30, 10, 3, 1, 0.1};
    // parameter setting
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta 
    search_start_theta[0] = -60;
    int search_len_theta = 5;
    float *search_theta_deg = (float *)malloc(search_len_theta * sizeof(float));
    float *search_theta_rad = (float *)malloc(search_len_theta * sizeof(float));
    for (int i = 0; i < search_len_theta; ++i){ 
        search_theta_deg[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad[i] = search_theta_deg[i] * PI / 180;
        //printf("search_theta_deg[%d] = %.2f\n", i, search_theta_deg[i]);
    }
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB = (float *)malloc(search_len_theta * sizeof(float));
    calculate_spatial_spectrum_PS(search_const, 
                                  search_len_theta, 
                                  search_theta_rad, 
                                  S_MUSIC_dB,
                                  search_step_theta[0]);
    gettimeofday(&time_Spatial_Spectrum_end, NULL);
    //printf("---\n");
    
    // find Max and position
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MUSIC_dB, position_theta, search_len_theta, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 1
    gettimeofday(&time_Spatial_Spectrum1_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune1 = NULL;
    float *search_theta_rad_prune1 = NULL;
    int search_len_theta_prune1;
    calculate_search_theta(search_theta_deg[position_theta[0]], &search_len_theta_prune1, &search_theta_deg_prune1, &search_theta_rad_prune1, search_step_theta[1]);
    //printf("len_dth1 = %d\n", len_dth1);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB_prune1 = (float *)malloc(search_len_theta_prune1 * sizeof(float));
    calculate_spatial_spectrum_PS(search_const, 
                                  search_len_theta_prune1,
                                  search_theta_rad_prune1,
                                  S_MUSIC_dB_prune1,
                                  search_step_theta[1]);
    gettimeofday(&time_Spatial_Spectrum1_end, NULL);
    // find Max and position
    gettimeofday(&time_Peak_Search1_start, NULL);
    int *position_theta_prune1 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MUSIC_dB_prune1, position_theta_prune1, search_len_theta_prune1, len_t_angle);
    gettimeofday(&time_Peak_Search1_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 2
    gettimeofday(&time_Spatial_Spectrum2_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune2 = NULL;
    float *search_theta_rad_prune2 = NULL;
    int search_len_theta_prune2;
    calculate_search_theta(search_theta_deg_prune1[position_theta_prune1[0]], &search_len_theta_prune2, &search_theta_deg_prune2, &search_theta_rad_prune2, search_step_theta[2]);
    //printf("len_dth2 = %d\n", len_dth2);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB_prune2 = (float *)malloc(search_len_theta_prune2 * sizeof(float));
    calculate_spatial_spectrum_PS(search_const, 
                                  search_len_theta_prune2,
                                  search_theta_rad_prune2,
                                  S_MUSIC_dB_prune2,
                                  search_step_theta[2]);
    gettimeofday(&time_Spatial_Spectrum2_end, NULL);
    // find Max and position
    gettimeofday(&time_Peak_Search2_start, NULL);
    int *position_theta_prune2 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MUSIC_dB_prune2, position_theta_prune2, search_len_theta_prune2, len_t_angle);
    gettimeofday(&time_Peak_Search2_end, NULL);
    //printf("---\n");

    //---------------------------------------------------------------
    // parameter setting 3
    gettimeofday(&time_Spatial_Spectrum3_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune3 = NULL;
    float *search_theta_rad_prune3 = NULL;
    int search_len_theta_prune3;
    calculate_search_theta_high_accuracy(search_theta_deg_prune2[position_theta_prune2[0]], &search_len_theta_prune3, &search_theta_deg_prune3, &search_theta_rad_prune3, search_step_theta[3]);
    //printf("len_dth3 = %d\n", len_dth3);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB_prune3 = (float *)malloc(search_len_theta_prune3 * sizeof(float));
    calculate_spatial_spectrum(search_const, 
                               search_len_theta_prune3, 
                               search_theta_rad_prune3, 
                               S_MUSIC_dB);
    gettimeofday(&time_Spatial_Spectrum3_end, NULL);
    // find Max and position
    gettimeofday(&time_Peak_Search3_start, NULL);
    int *position_theta_prune3 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MUSIC_dB_prune3, position_theta_prune3, search_len_theta_prune3, len_t_angle);
    gettimeofday(&time_Peak_Search3_end, NULL);
    //printf("---\n");

    
    //---------------------------------------------------------------
    // parameter setting 4
    gettimeofday(&time_Spatial_Spectrum4_start, NULL);
    // Search angle theta
    float *search_theta_deg_prune4 = NULL;
    float *search_theta_rad_prune4 = NULL;
    int search_len_theta_prune4;
    calculate_search_theta_last(search_theta_deg_prune3[position_theta_prune3[0]], &search_len_theta_prune4, &search_theta_deg_prune4, &search_theta_rad_prune4, search_step_theta[4]);
    //printf("len_dth4 = %d\n", len_dth4);
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB_prune4 = (float *)malloc(search_len_theta_prune4 * sizeof(float));
    calculate_spatial_spectrum(search_const, 
                               search_len_theta_prune4, 
                               search_theta_rad_prune4, 
                               S_MUSIC_dB_prune4);

    gettimeofday(&time_Spatial_Spectrum4_end, NULL);
    // find Max and position
    gettimeofday(&time_Peak_Search4_start, NULL);
    int *position_theta_prune4 = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_MUSIC_dB_prune4, position_theta_prune4, search_len_theta_prune4, len_t_angle);
    gettimeofday(&time_Peak_Search4_end, NULL);
    //printf("---\n");

    gettimeofday(&time_MUSIC_end, NULL);
    // ==================================================================
    // ====================== MUSIC algorithm end =======================
    // ==================================================================
    
   //---------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Eigen, time_Pn, time_Spatial_Spectrum, time_Peak_Search, time_MUSIC;      // create float parameter in order to convert (us) to (ms)
    float time_Spatial_Spectrum1, time_Spatial_Spectrum2, time_Spatial_Spectrum3, time_Spatial_Spectrum4;
    float time_Peak_Search1, time_Peak_Search2, time_Peak_Search3, time_Peak_Search4;
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


    time_MUSIC = time_MUSIC_diff.tv_sec * 1000000 + time_MUSIC_diff.tv_usec;;
    float total_time_Spatial_Spectrum, total_time_Peak_Search;
    total_time_Spatial_Spectrum = time_Spatial_Spectrum+time_Spatial_Spectrum1+time_Spatial_Spectrum2+time_Spatial_Spectrum3+time_Spatial_Spectrum4;
    total_time_Peak_Search = time_Peak_Search+time_Peak_Search1+time_Peak_Search2+time_Peak_Search3+time_Peak_Search4;

    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MUSIC DOA----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("BMGS-QR num of blocks:\t%d\n", BMGS_qr_num_blocks);
    printf("Search step theta:\t%.2f, %.2f, %.2f, %.2f, %.2f (degree)\n", 
        search_step_theta[0], search_step_theta[1], search_step_theta[2], search_step_theta[3], search_step_theta[4]);
    printf("\n");
    printf("\n\t-----Estimated results------\n");
    printf("position0 theta : \t%d\n", position_theta[0]);
    printf(RED "Theta estimation0 :\t(%.3f) (degree)\n" CLOSE, search_theta_deg[position_theta[0]]);
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

    printf(L_GREEN "Total MUSIC REAL time : \t%.3f(ms)\n" CLOSE, time_MUSIC / 1000);
    printf(L_GREEN "->Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    

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

    free(search_theta_deg);
    free(search_theta_rad);

    free(S_MUSIC_dB);

    free(position_theta);

    // Prune 1
    free(search_theta_deg_prune1);
    free(search_theta_rad_prune1);
    free(S_MUSIC_dB_prune1);
    free(position_theta_prune1);

    // Prune 2
    free(search_theta_deg_prune2);
    free(search_theta_rad_prune2);
    free(S_MUSIC_dB_prune2);
    free(position_theta_prune2);

    // Prune 3
    free(search_theta_deg_prune3);
    free(search_theta_rad_prune3);
    free(S_MUSIC_dB_prune3);
    free(position_theta_prune3);

    
    // Prune 4
    free(search_theta_deg_prune4);
    free(search_theta_rad_prune4);
    free(S_MUSIC_dB_prune4);
    free(position_theta_prune4);
    
}

int main()
{
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff;
    //-------------------------------------------------------------------
    // Physical Parameters initialization
    PhysicalParameters phys;
    phys.fc = 28e9;
    phys.c = 3e8;
    phys.lambda = (float)phys.c / (float)phys.fc;
    phys.kc = 2 * PI / phys.lambda;
    phys.snr = 10;

    int M = 64;
    float angle_theta[100] = { 25.5, 45.2, 10.0, 80.0, 70, 45, 58}; // elevation
    // Tx Parameters initialization
    TxParameters tx;
    tx.Tx_M_x = M;
    tx.Tx_beamwidth = 0.1;
    tx.Tx_beamwidth_samples = 100;
    tx.angle_theta = angle_theta;
    tx.number_angle = 2;

    // Rx Parameters initialization
    RxParameters rx;
    rx.Rx_M_x = M;
    rx.d = phys.lambda * 0.5f;
    rx.nd = 1024;
    rx.qr_iter = 2;
    rx.BMGS_qr_num_blocks = 8;
    //-------------------------------------------------------------------
    gettimeofday(&time_MUSIC_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);
    //=================== MUSIC Algorithm =================================
    doa2d_music_bmgsqr_prunesearch_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_MUSIC_end, NULL);
    timersub(&time_MUSIC_end, &time_MUSIC_start, &time_MUSIC_diff);
    float time_MUSIC;
    time_MUSIC = time_MUSIC_diff.tv_sec * 1000000 + time_MUSIC_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total MUSIC time : \t\t%.3f(ms)\n" CLOSE, time_MUSIC / 1000); //Contains generated signals

    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}

 