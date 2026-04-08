// 2D DOA Estimation - ML Algorithm (float)
// Original Search
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
void doa2d_ml_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx) {
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_ML_start, time_ML_end, time_ML_diff;
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff;
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

    // ==================================================================
    // ====================== ML algorithm start ========================
    // ==================================================================
    gettimeofday(&time_ML_start, NULL);
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
    //------------------------ Search Start -------------------------
    //---------------------------------------------------------------
    SearchConst *search_const = (SearchConst*)malloc(sizeof(SearchConst));
    search_const->Rx_M = Rx_M;
    search_const->d = d;
    search_const->kc = kc;
    search_const->Pn_re = R_xx_re;
    search_const->Pn_im = R_xx_im;
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float search_step_theta[1] = {0.1};
    // parameter setting
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta
    search_start_theta[0] = -60;
    int search_len_theta = 1201;
    float *search_theta_deg = (float *)malloc(search_len_theta * sizeof(float));
    float *search_theta_rad = (float *)malloc(search_len_theta * sizeof(float));
    for (int i = 0; i < search_len_theta; ++i){ 
        search_theta_deg[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad[i] = search_theta_deg[i] * PI / 180;
        //printf("search_theta_deg[%d] = %.2f\n", i, search_theta_deg[i]);
    }

    // Calculate Spatial Spectrum and Peak Search
    float *S_ML_dB = (float *)malloc(search_len_theta * sizeof(float));
    calculate_spatial_spectrum_ML(search_const, 
                                    search_len_theta, 
                                    search_theta_rad, 
                                    S_ML_dB);
    gettimeofday(&time_Spatial_Spectrum_end, NULL);                           
    // find Max and position
    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks(S_ML_dB, position_theta, search_len_theta, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    gettimeofday(&time_ML_end, NULL);
    


    // ==================================================================
    // ======================= ML algorithm end =========================
    // ==================================================================
    float time_Rxx, time_ML, time_Spatial_Spectrum ,time_Peak_Search;
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Spatial_Spectrum_end, &time_Spatial_Spectrum_start, &time_Spatial_Spectrum_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search_end, &time_Peak_Search_start, &time_Peak_Search_diff);                    // calculate Peak Search
    timersub(&time_ML_end, &time_ML_start, &time_ML_diff);                    // calculate ML

    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Spatial_Spectrum = time_Spatial_Spectrum_diff.tv_sec * 1000000 + time_Spatial_Spectrum_diff.tv_usec;
    time_Peak_Search = time_Peak_Search_diff.tv_sec * 1000000 + time_Peak_Search_diff.tv_usec;
    time_ML = time_ML_diff.tv_sec * 1000000 + time_ML_diff.tv_usec;

    // print parameter
    printf("-----------------------------------------\n");
    printf("-----------------ML DOA------------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("Search start theta:\t%.2f\t(degree)\n", search_start_theta[0]);
    printf("Search step theta:\t%.2f\t(degree)\n", search_step_theta[0]);

    printf("\n\t-----Estimated results------\n");
    printf("position0 theta : \t%d\n", position_theta[0]);
    printf(RED "Theta estimation :\t(%.3f) (degree)\n" CLOSE, search_theta_deg[position_theta[0]]);
    //printf("Max_theta0 :\t\t%f(dB)\n", max_temp);
    
    printf("\n");
    printf("Total search count: \t%d\n", search_count);
    printf("\n\t------------Time------------\n");
    printf("Total Rxx time: \t\t\t%.3f(ms)\n", time_Rxx / 1000);
    printf("\t  -------DOA start-------\n");
    //printf("Total Generalized Inverse REAL time :\t%.3f(ms)\n" CLOSE, total_time_Generalized_Inverse / 1000);
    //printf("Total Orthogonal Projection REAL time :\t%.3f(ms)\n" CLOSE, total_time_Orthogonal_Projection / 1000);
    printf("Total Spatial Spectrum time: \t%.3f(ms)\n", time_Spatial_Spectrum / 1000);
    printf("Total Peak Search time: \t\t%.3f(ms)\n", time_Peak_Search / 1000);
    printf(L_GREEN "Total ML REAL time :\t\t\t%.3f(ms)\n" CLOSE, time_ML / 1000);
    printf(L_GREEN "->Total multiplication time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    
    save_Spectrum_to_csv("data/1D_ML_dB.csv", S_ML_dB, search_len_theta);
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

    int M = 64;
    float angle_theta[100] = { 16, 16, 10.0, 80.0, 70, 45, 58}; // elevation
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
    gettimeofday(&time_ML_start, NULL);
    //=================== generate Rx signal ==============================
    float *Rx_sig_re = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    float *Rx_sig_im = (float *)calloc(rx.Rx_M_x  * rx.nd + AVX*sizeof(float), sizeof(float));
    generate_Rx_signal(Rx_sig_re, Rx_sig_im, phys, rx, tx);
    //=================== ML Algorithm =================================
    doa2d_ml_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

    gettimeofday(&time_ML_end, NULL);
    timersub(&time_ML_end, &time_ML_start, &time_ML_diff);
    float time_ML;
    time_ML = time_ML_diff.tv_sec * 1000000 + time_ML_diff.tv_usec;
    printf("--------------------------------------\n");
    printf(L_GREEN "Total ML time : \t\t%.3f(ms)\n" CLOSE, time_ML / 1000); //Contains generated signals

    //=====================================================================
    free(Rx_sig_re);
    free(Rx_sig_im);
}
 
