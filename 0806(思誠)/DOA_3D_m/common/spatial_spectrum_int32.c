#define PI acos(-1)
#define BLOCK_SIZE 16
#define PRINT_RESULT 1
#define PLOT_RESULT 0

//--------------------
#define AVX 16            
#define M_Antenna 64
#define ND 512
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "complex_matrix_ops_int32.h"
#include "multi_beam_weight.h"
#include "spatial_spectrum.h"
#include "spatial_spectrum_int32.h"
#include "lu_decomp.h"
#include "q_format_config.h"
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



float time_spatial_spectrum_value = 0.0;
float time_complex_div = 0.0;
void spatial_spectrum_value_int32(int32_t *a_vector_re, int32_t *a_vector_im, int32_t *Pn_re, int32_t *Pn_im, int M, float *result_re, float *result_im)
{
    struct timeval time_spatial_spectrum_value_start, time_spatial_spectrum_value_end, time_spatial_spectrum_value_diff; // time initial
    struct timeval time_complex_div_start, time_complex_div_end, time_complex_div_diff; // time initial
    //---------------------------------------------------------------
    int32_t *Pn_a_vector_temp_re = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *Pn_a_vector_temp_im = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *compute_spatial_spectrum_value_temp_re = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *compute_spatial_spectrum_value_temp_im = (int32_t *)malloc(M * sizeof(int32_t));
    float *compute_spatial_spectrum_value_temp_re_float = (float *)malloc(sizeof(float));
    float *compute_spatial_spectrum_value_temp_im_float = (float *)malloc(sizeof(float));
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    gettimeofday(&time_spatial_spectrum_value_start, NULL);
    //---------------------------------------------------------------
    complex_matrix_multiplication_int32(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1);
    // printf("Pn_a_vector_temp\n");
    // print_complex_matrix_int32(Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, 1);

    //---------------------------------------------------------------
    complex_matrix_conjugate_transpose_int32(a_vector_re, a_vector_im, M, 1);
    // printf("a_vector_temp\n");
    // print_complex_matrix_int32(a_vector_re, a_vector_im, M, 1);

    //---------------------------------------------------------------
    complex_matrix_multiplication_int32(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, M, 1);
    // printf("compute_spatial_spectrum_value_temp\n");
    // print_complex_matrix_int32(compute_spatial_spectrum_value_temp_re, compute_spatial_spectrum_value_temp_im, 1, 1);

    //---------------------------------------------------------------
    // gettimeofday(&time_spatial_spectrum_value_end, NULL);
    // timersub(&time_spatial_spectrum_value_end, &time_spatial_spectrum_value_start, &time_spatial_spectrum_value_diff);
    // time_spatial_spectrum_value += time_spatial_spectrum_value_diff.tv_sec * 1000000 + time_spatial_spectrum_value_diff.tv_usec;
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    //---------------------------------------------------------------
    // int32_t to float
    // compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0] / (float)(1 << Q_SHIFT);
    // compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0] / (float)(1 << Q_SHIFT);
    compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0] / Q_SCALE;
    compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0] / Q_SCALE;
    // compute_spatial_spectrum_value_temp_re_float[0] = (float)compute_spatial_spectrum_value_temp_re[0];
    // compute_spatial_spectrum_value_temp_im_float[0] = (float)compute_spatial_spectrum_value_temp_im[0];
    // printf("compute_spatial_spectrum_value_temp_float = (%f,%f)\n", compute_spatial_spectrum_value_temp_re_float[0], compute_spatial_spectrum_value_temp_im_float[0]);

    //---------------------------------------------------------------
    
    // gettimeofday(&time_complex_div_start, NULL);

    cpp_division2(1, 0, &compute_spatial_spectrum_value_temp_re_float[0], &compute_spatial_spectrum_value_temp_im_float[0], result_re, result_im);
    
    // gettimeofday(&time_complex_div_end, NULL);
    // timersub(&time_complex_div_end, &time_complex_div_start, &time_complex_div_diff);
    time_complex_div += time_complex_div_diff.tv_sec * 1000000 + time_complex_div_diff.tv_usec;
    // printf("music = (%.5f,%.5f)\n", *result_re, *result_re);
    free(Pn_a_vector_temp_re);
    free(Pn_a_vector_temp_im);
    free(compute_spatial_spectrum_value_temp_re);
    free(compute_spatial_spectrum_value_temp_im);
    free(compute_spatial_spectrum_value_temp_re_float);
    free(compute_spatial_spectrum_value_temp_im_float);
}

float time_cpp_exp2_to_int = 0.0f;
void calculate_spatial_spectrum_3D_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB
)
{
    struct timeval time_cpp_exp2_to_int_start, time_cpp_exp2_to_int_end, time_cpp_exp2_to_int_diff; // time initial


    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    int32_t *a_vector_re_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *a_vector_im_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
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
                }
            }
            // --------------------------------------
            // gettimeofday(&time_cpp_exp2_to_int_start, NULL);
            float_to_q_int32_avx512(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M);
            // gettimeofday(&time_cpp_exp2_to_int_end, NULL);
            // timersub(&time_cpp_exp2_to_int_end, &time_cpp_exp2_to_int_start, &time_cpp_exp2_to_int_diff);
            // time_cpp_exp2_to_int += time_cpp_exp2_to_int_diff.tv_sec * 1000000 + time_cpp_exp2_to_int_diff.tv_usec;

            // --------------------------------------
            spatial_spectrum_value_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            // printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    // q_int32_to_float_avx512(S_spatial_spectrum_re, S_spatial_spectrum_im, S_spatial_spectrum_re_int32, S_spatial_spectrum_im_int32, search_len_theta*search_len_phi);
    
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            S_spatial_spectrum_im[elevation*search_len_phi + azimuth] = 0.0;
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
        }
    }


    // printf("time_cpp_exp2_to_int = %.3f(us)\n", time_cpp_exp2_to_int);
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    // printf("time_spatial_spectrum_value_avg = %.3f(us)\n", time_spatial_spectrum_value/(search_len_theta*search_len_phi));
    // printf("time_complex_div = %.3f(us)\n", time_complex_div);
    
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(a_vector_re_int32);free(a_vector_im_int32);

}


const int rx_m_index_int32[] = {64,256,576};
const int beam_deg_index_int32[] = {30, 10};
void calculate_spatial_spectrum_3D_multiBeam_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB,
    float search_step_theta 
)
{
    // struct timeval time_cpp_exp2_to_int_start, time_cpp_exp2_to_int_end, time_cpp_exp2_to_int_diff; // time initial
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // ---------------------------------------------------------------
    // Select different weights based on Rx_M and search_step_theta
    int shift=4;
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    int rx_m_idx = find_index(rx_m_index_int32, 4, Rx_M);
    int theta_idx = find_index(beam_deg_index_int32, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || theta_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        printf("Configuration not found\n");
    } else if ((Rx_M == 64 || Rx_M == 256) && theta_idx == 1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f;
        }
        shift = 1;
        printf("Force use 1.0 beam for Rx_M = %d, theta = 10\n", Rx_M);
    } else {
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][theta_idx], Rx_M * sizeof(float));
        printf("Find the configuration rx_m_idx = %d, theta_idx = %d\n", rx_m_idx, theta_idx);
    }
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    int32_t *a_vector_re_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *a_vector_im_int32 = (int32_t *)malloc(Rx_M * sizeof(int32_t));
    int32_t *S_spatial_spectrum_re_int32 = (int32_t *)malloc(search_len_theta*search_len_phi * sizeof(int32_t) + AVX * sizeof(int32_t));
    int32_t *S_spatial_spectrum_im_int32 = (int32_t *)malloc(search_len_theta*search_len_phi * sizeof(int32_t) + AVX * sizeof(int32_t));
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
                    // printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                    a_vector_re[j*Rx_M_y+k] = a_vector_re[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k]*shift;
                    a_vector_im[j*Rx_M_y+k] = a_vector_im[j*Rx_M_y+k]*w_multi_beam_re[j*Rx_M_y+k]*shift;
                    // printf("a_vector_multi(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                }
            }
            // --------------------------------------
            // gettimeofday(&time_cpp_exp2_to_int_start, NULL);
            // float_matrix_to_q_format(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M, 1, q_shift_DOA);
            float_to_q_int32_avx512(a_vector_re_int32, a_vector_im_int32, a_vector_re, a_vector_im, Rx_M);
            // gettimeofday(&time_cpp_exp2_to_int_end, NULL);
            // timersub(&time_cpp_exp2_to_int_end, &time_cpp_exp2_to_int_start, &time_cpp_exp2_to_int_diff);
            // time_cpp_exp2_to_int += time_cpp_exp2_to_int_diff.tv_sec * 1000000 + time_cpp_exp2_to_int_diff.tv_usec;

            // --------------------------------------
            // spatial_spectrum_value_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re_int32[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im_int32[elevation*search_len_phi + azimuth]);
            spatial_spectrum_value_int32(a_vector_re_int32, a_vector_im_int32, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            // printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    // q_int32_to_float_avx512(S_spatial_spectrum_re, S_spatial_spectrum_im, S_spatial_spectrum_re_int32, S_spatial_spectrum_im_int32, search_len_theta*search_len_phi);
    
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){//phi
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
        }
    }


    // printf("time_cpp_exp2_to_int = %.3f(us)\n", time_cpp_exp2_to_int);
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    // printf("time_spatial_spectrum_value_avg = %.3f(us)\n", time_spatial_spectrum_value/(search_len_theta*search_len_phi));
    // printf("time_complex_div = %.3f(us)\n", time_complex_div);
    
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(a_vector_re_int32);free(a_vector_im_int32);
    free(S_spatial_spectrum_re_int32);free(S_spatial_spectrum_im_int32);
}