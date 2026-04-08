// 2D DOA Estimation - MUSIC Algorithm with BMGS QR Decomposition
// Mixed Precision (float32 / int32) Original Search Version
//--------------------
#define PI acos(-1)
#define AVX 16 
#define Q_SHIFT 13
//--------------------
#include <immintrin.h>
#include "../common/q_format_config.h"
#include "../common/complex_matrix_ops.h"
#include "../common/generate_signal.h"
#include "../common/eigen_qr.h"
#include "../common/lu_decomp.h"
#include "../common/spatial_spectrum.h"
#include "../common/doa_parameters.h"
//--------------------
// int32
#include "../common/complex_matrix_ops_int32.h"
#include "../common/spatial_spectrum_int32.h"
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
//----------------------------------------------------------------
void doa2d_music_bmgsqr_cpu_float_int32_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MUSIC_start, time_MUSIC_end, time_MUSIC_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
    struct timeval time_Eigen_start, time_Eigen_end, time_Eigen_diff;
    struct timeval time_Pn_start, time_Pn_end, time_Pn_diff;
    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;
    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;
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
        //printf("(%f,%f)\n", R_xx_re[i], R_xx_im[i]isip);
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
    // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, Rx_M, Rx_M);
    
    // float to int32_t conversion
    int32_t *Pn_re_int32 = (int32_t*)malloc(Rx_M * Rx_M* sizeof(int32_t));
    int32_t *Pn_im_int32 = (int32_t *)malloc(Rx_M * Rx_M* sizeof(int32_t));
    // Convert float to int32_t
    float_matrix_to_q_format(Pn_re_int32, Pn_im_int32, Pn_re, Pn_im, Rx_M, Rx_M);

    //---------------------------------------------------------------
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst_int32 *search_const_int32 = (SearchConst_int32*)malloc(sizeof(SearchConst_int32));
    search_const_int32->Rx_M = Rx_M;
    search_const_int32->d = d;
    search_const_int32->kc = kc;
    search_const_int32->Pn_re = Pn_re_int32;
    search_const_int32->Pn_im = Pn_im_int32;
    float search_step_theta[6] = {0.1};
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta 
    search_start_theta[0] = -60; // Starting angle of search
    int search_len_theta = 1201;
    float *search_theta_deg = (float *)malloc(search_len_theta * sizeof(float));
    float *search_theta_rad = (float *)malloc(search_len_theta * sizeof(float));
    // Search angle
    for (int i = 0; i < search_len_theta; ++i){ 
        search_theta_deg[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad[i] = search_theta_deg[i] * PI / 180;
    }

    
    // Calculate Spatial Spectrum and Peak Search
    float *S_MUSIC_dB = (float *)malloc(search_len_theta * sizeof(float));

    calculate_spatial_spectrum_int32(search_const_int32, 
                                     search_len_theta, 
                                     search_theta_rad, 
                                     S_MUSIC_dB);
    gettimeofday(&time_Spatial_Spectrum_end, NULL);
    
    // find Max and position
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta = (int *)malloc(len_t_angle * sizeof(int));
    position_theta[0]=0;
    find_spatial_spectrum_peaks(S_MUSIC_dB, position_theta, search_len_theta, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    //printf("---\n");

    gettimeofday(&time_MUSIC_end, NULL);
    // ==================================================================
    // ====================== MUSIC algorithm end =======================
    // ==================================================================
    
    //---------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Eigen, time_Pn, time_Spatial_Spectrum, time_Peak_Search, time_MUSIC;      // create float parameter in order to convert (us) to (ms)
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Eigen_end, &time_Eigen_start, &time_Eigen_diff);           // calculate Eigen
    timersub(&time_Pn_end, &time_Pn_start, &time_Pn_diff);                    // calculate Pn
    timersub(&time_Spatial_Spectrum_end, &time_Spatial_Spectrum_start, &time_Spatial_Spectrum_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search_end, &time_Peak_Search_start, &time_Peak_Search_diff);                    // calculate Peak Search
    timersub(&time_MUSIC_end, &time_MUSIC_start, &time_MUSIC_diff);                    // calculate MUSIC

    // Compute time
    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Eigen = time_Eigen_diff.tv_sec * 1000000 + time_Eigen_diff.tv_usec;
    time_Pn = time_Pn_diff.tv_sec * 1000000 + time_Pn_diff.tv_usec;
    time_Spatial_Spectrum = time_Spatial_Spectrum_diff.tv_sec * 1000000 + time_Spatial_Spectrum_diff.tv_usec;
    time_Peak_Search = time_Peak_Search_diff.tv_sec * 1000000 + time_Peak_Search_diff.tv_usec;
    time_MUSIC = time_MUSIC_diff.tv_sec * 1000000 + time_MUSIC_diff.tv_usec;

    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MUSIC DOA----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("BMGS-QR num of blocks:\t%d\n", BMGS_qr_num_blocks);
    printf("Search start theta:\t%.2f\t(degree)\n", search_start_theta[0]);
    printf("Search step theta:\t%.2f\t(degree)\n", search_step_theta[0]);

    printf("\n\t-----Estimated results------\n");
    printf("position theta : \t%d\n", position_theta[0]);
    printf(RED "Theta estimation :\t(%.3f) (degree)\n" CLOSE, search_theta_deg[position_theta[0]]);
    //printf("Max_theta :\t\t%f(dB)\n", max_temp);
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
    printf("Total Spatial Spectrum time: \t%.3f(ms)\n", time_Spatial_Spectrum / 1000);
    printf("Total Peak Search time: \t%.3f(ms)\n", time_Peak_Search / 1000);
    printf(L_GREEN "Total MUSIC REAL time : \t%.3f(ms)\n" CLOSE, time_MUSIC / 1000);
    printf(L_GREEN "->Total multiplication time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //-------------------------------------------------------------------
    // save_Spectrum_to_csv("../float_DOA_3D/MUSIC_spectrum_dB/S_MUSIC_dB.csv", S_MUSIC_dB, search_len_theta);
    //-------------------------------------------------------------------
    // free memory
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
    free(search_theta_deg);
    free(search_theta_rad);
    free(S_MUSIC_dB);
    free(position_theta);
    free(search_start_theta);
    free(search_const_int32);
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
    doa2d_music_bmgsqr_cpu_float_int32_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

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


