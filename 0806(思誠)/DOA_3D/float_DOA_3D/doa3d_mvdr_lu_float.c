// 3D DOA Estimation - MVDR Algorithm with LU Decomposition (float)
// Original Search
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

void doa3d_mvdr_lu_cpu_float_test(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx)
{
    //-------------------------------------------------------------------
    // Time parameter initialize
    struct timeval time_MVDR_start, time_MVDR_end, time_MVDR_diff; // time initial
    struct timeval time_Rxx_start, time_Rxx_end, time_Rxx_diff; // time initial
    struct timeval time_Rxx_inv_start, time_Rxx_inv_end, time_Rxx_inv_diff;          // time initial (inverse covariance)
    struct timeval time_Spatial_Spectrum_start, time_Spatial_Spectrum_end, time_Spatial_Spectrum_diff;          // time initial
    struct timeval time_Peak_Search_start, time_Peak_Search_end, time_Peak_Search_diff;          // time initial
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
    // printf("----------Pn------------\n");
    // print_complex_matrix(Pn_re, Pn_im, Rx_M, Rx_M);

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
    float search_step_theta[6] = {0.1};
    float search_step_phi[6] = {0.1};
    float *search_start_theta = (float *)malloc(4 * sizeof(float));
    float *search_start_phi = (float *)malloc(4 * sizeof(float));
    //---------------------------------------------------------------
    gettimeofday(&time_Spatial_Spectrum_start, NULL);
    // Search angle theta 
    search_start_theta[0] = 0; // 搜索的起始角度
    int search_len_theta = 901;
    float *search_theta_deg = (float *)malloc(search_len_theta * sizeof(float));
    float *search_theta_rad = (float *)malloc(search_len_theta * sizeof(float));
    // Search angle
    for (int i = 0; i < search_len_theta; ++i){ 
        search_theta_deg[i] = search_start_theta[0] + search_step_theta[0] * i;
        search_theta_rad[i] = search_theta_deg[i] * PI / 180;
    }

    // Search angle phi
    search_start_phi[0] = -60; // 搜索的起始角度
    int search_len_phi = 1201;
    float *search_phi_deg = (float *)malloc(search_len_phi * sizeof(float));
    float *search_phi_rad = (float *)malloc(search_len_phi * sizeof(float));
    // Search angle
    for (int i = 0; i < search_len_phi; ++i){ 
        search_phi_deg[i] = search_start_phi[0] + search_step_phi[0] * i;
        search_phi_rad[i] = search_phi_deg[i] * PI / 180;
    }

    // Calculate Spatial Spectrum and Peak Search
    float *S_MVDR_dB = (float *)malloc(search_len_theta*search_len_phi * sizeof(float));
    calculate_spatial_spectrum_3D(
        search_const, 
        search_len_theta, 
        search_len_phi,               
        search_theta_rad, 
        search_phi_rad,
        S_MVDR_dB
    );
    gettimeofday(&time_Spatial_Spectrum_end, NULL);

    // find Max and position

    gettimeofday(&time_Peak_Search_start, NULL);
    int *position_theta = (int *)malloc(len_t_angle * sizeof(int));
    int *position_phi = (int *)malloc(len_t_angle * sizeof(int));
    find_spatial_spectrum_peaks_3D(S_MVDR_dB, position_theta, position_phi, search_len_theta, search_len_phi, len_t_angle);
    gettimeofday(&time_Peak_Search_end, NULL);
    //printf("---\n");

    gettimeofday(&time_MVDR_end, NULL);
    // ==================================================================
    // ====================== MVDR algorithm end ========================
    // ==================================================================

    //-------------------------------------------------------------------
    // timersub function
    float time_Rxx, time_Rxx_inv, time_Spatial_Spectrum, time_Peak_Search,time_MVDR;                           // create float parameter in order to convert (us) to (ms)
    timersub(&time_Rxx_end, &time_Rxx_start, &time_Rxx_diff);
    timersub(&time_Rxx_inv_end, &time_Rxx_inv_start, &time_Rxx_inv_diff);                    // calculate Pn
    timersub(&time_Spatial_Spectrum_end, &time_Spatial_Spectrum_start, &time_Spatial_Spectrum_diff);           // calculate Spatial Spectrum
    timersub(&time_Peak_Search_end, &time_Peak_Search_start, &time_Peak_Search_diff);                    // calculate Peak Search
    timersub(&time_MVDR_end, &time_MVDR_start, &time_MVDR_diff);                    // calculate MVDR

    // Compute time
    time_Rxx = time_Rxx_diff.tv_sec * 1000000 + time_Rxx_diff.tv_usec;
    time_Rxx_inv = time_Rxx_inv_diff.tv_sec * 1000000 + time_Rxx_inv_diff.tv_usec;
    time_Spatial_Spectrum = time_Spatial_Spectrum_diff.tv_sec * 1000000 + time_Spatial_Spectrum_diff.tv_usec;
    time_Peak_Search = time_Peak_Search_diff.tv_sec * 1000000 + time_Peak_Search_diff.tv_usec;;
    time_MVDR = time_MVDR_diff.tv_sec * 1000000 + time_MVDR_diff.tv_usec;;

    // print parameter
    printf("\n");
    printf("-----------------------------------------\n");
    printf("----------------MVDR DOA-----------------\n");
    printf("-----------------------------------------\n");
    printf("\n\t-------DOA parameter--------\n");
    printf("Search start theta:\t%.2f\t(degree)\n", search_start_theta[0]);
    printf("Search step theta:\t%.2f\t(degree)\n", search_step_theta[0]);
    printf("Search start phi:\t%.2f\t(degree)\n", search_start_phi[0]);
    printf("Search step phi:\t%.2f\t(degree)\n", search_step_phi[0]);
    
    printf("\n\t-----Estimated results------\n");
    printf("position theta : \t%d\n", position_theta[0]);
    printf("position phi : \t\t%d\n", position_phi[0]); 
    printf(RED "Theta estimation :\t(%.3f, %.3f) (degree)\n" CLOSE, search_theta_deg[position_theta[0]], search_phi_deg[position_phi[0]]);
    //printf("Max_theta :\t\t%f(dB)\n", max_temp);

    printf("\n\t------------Time------------\n");
    printf("\t  -------DOA start-------\n");
    printf("Total Rxx time: \t\t%.3f(ms)\n", time_Rxx / 1000);
    printf("Total Rxx LU inv time: \t\t%.3f(ms)\n", time_Rxx_inv / 1000);
    printf("Total Spatial Spectrum time: \t%.3f(ms)\n", time_Spatial_Spectrum / 1000);
    printf("Total Peak Search time: \t%.3f(ms)\n", time_Peak_Search / 1000);
    printf(L_GREEN "Total MVDR REAL time : \t\t%.3f(ms)\n" CLOSE, time_MVDR / 1000);
    printf(L_GREEN "->Total multiplication time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);

    //-------------------------------------------------------------------
    save_Spectrum_to_csv("../float_DOA_3D/MUSIC_spectrum_dB/S_MUSIC_dB.csv", S_MVDR_dB, search_len_theta*search_len_phi);
    //-------------------------------------------------------------------
    // free memory
    free(R_xx_re);
    free(R_xx_im);
    free(Pn_re);
    free(Pn_im);
    free(S_MVDR_dB);
    free(search_theta_deg);
    free(search_theta_rad);
    free(search_phi_deg);
    free(search_phi_rad);
    free(search_start_theta);
    free(search_start_phi);
    free(position_theta);
    free(position_phi);
    free(search_const);

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

    int M = 8;
    float angle_theta[100] = {45, 15, 15, 60.0, 70, 45, 58}; // elevation
    float angle_phi[100]    = {0, 16, 14, 45.0, 0, 45, 58}; // azimuth
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
    doa3d_mvdr_lu_cpu_float_test(Rx_sig_re, Rx_sig_im, phys, rx, tx);

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
 