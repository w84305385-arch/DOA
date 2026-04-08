//--------------------
#define PI acos(-1)
#define AVX 16            
//--------------------
#include <immintrin.h>
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "multi_beam_weight.h"
#include "spatial_spectrum.h"
#include "lu_decomp.h"

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


// ================================
// ======= spatial spectrum =======
// ================================
// float time_spatial_spectrum_value_float = 0.0;
// float time_complex_div_float = 0.0;
void spatial_spectrum_value(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *result_re, float *result_im)
{
    // struct timeval time_spatial_spectrum_value_start, time_spatial_spectrum_value_end, time_spatial_spectrum_value_diff; // time initial
    // struct timeval time_complex_div_start, time_complex_div_end, time_complex_div_diff; // time initial
    //---------------------------------------------------------------
    float *Pn_a_vector_temp_re = (float *)malloc(M * sizeof(float) + AVX * sizeof(float));
    float *Pn_a_vector_temp_im = (float *)malloc(M * sizeof(float) + AVX * sizeof(float));
    float *spatial_spectrum_value_temp_re = (float *)malloc(M * sizeof(float) + AVX * sizeof(float));
    float *spatial_spectrum_value_temp_im = (float *)malloc(M * sizeof(float) + AVX * sizeof(float));
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    // gettimeofday(&time_spatial_spectrum_value_start, NULL);

    complex_matrix_multiplication(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector_re, a_vector_im, M, 1);
    complex_matrix_multiplication(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, spatial_spectrum_value_temp_re, spatial_spectrum_value_temp_im, 1, M, 1);

    // gettimeofday(&time_spatial_spectrum_value_end, NULL);
    // timersub(&time_spatial_spectrum_value_end, &time_spatial_spectrum_value_start, &time_spatial_spectrum_value_diff);
    // time_spatial_spectrum_value_float += time_spatial_spectrum_value_diff.tv_sec * 1000000 + time_spatial_spectrum_value_diff.tv_usec;

    //---------------------------------------------------------------
    // gettimeofday(&time_complex_div_start, NULL);
    cpp_division2(1, 0, &spatial_spectrum_value_temp_re[0], &spatial_spectrum_value_temp_im[0], result_re, result_im);
    // gettimeofday(&time_complex_div_end, NULL);
    // timersub(&time_complex_div_end, &time_complex_div_start, &time_complex_div_diff);
    // time_complex_div_float += time_complex_div_diff.tv_sec * 1000000 + time_complex_div_diff.tv_usec;
    // Convert the time difference to seconds
    //printf("spatial_spectrum_value = (%.5f,%.5f)\n", *result_re, *result_im);

    free(Pn_a_vector_temp_re);
    free(Pn_a_vector_temp_im);
    free(spatial_spectrum_value_temp_re);
    free(spatial_spectrum_value_temp_im);
    //printf("music = (%.5f,%.5f)\n", *result_re, *result_re);
}

void calculate_spatial_spectrum_3D(
    SearchConst *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB
)
{
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
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
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[i]);
        }
    }
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value_float);
    // printf("time_spatial_spectrum_value_avg = %.3f(us)\n", time_spatial_spectrum_value_float/(search_len_theta*search_len_phi));
    // printf("time_complex_div_float = %.3f(us)\n", time_complex_div_float);
    // printf("time_complex_div_float_avg = %.3f(us)\n", time_complex_div_float/(search_len_theta*search_len_phi));
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
}


// multi beam weight table index
const int rx_m_index[] = {64,256,576};
const int beam_deg_index[] = {30, 10};
int find_index(const int *array, int size, int value) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == value) return i;
    }
    return -1; // 
}
void calculate_spatial_spectrum_3D_multiBeam(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float *spatial_spectrum_value_dB, 
    float search_step_theta
){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // ---------------------------------------------------------------
    // Select different weights based on Rx_M and search_step_theta
    float *w_comebine_beam_re = (float*)calloc(Rx_M, sizeof(float));
    int rx_m_idx = find_index(rx_m_index, 4, Rx_M);
    int theta_idx = find_index(beam_deg_index, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || theta_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_comebine_beam_re[i] = 1.0f;
        }
        printf("Configuration not found\n");
    } else if ((Rx_M == 64 || Rx_M == 256) && theta_idx == 1) {
        for (int i = 0; i < Rx_M; i++) {
            w_comebine_beam_re[i] = 1.0f;
        }
        printf("Force use 1.0 beam for Rx_M = %d, theta = 10\n", Rx_M);
    } else {
        memcpy(w_comebine_beam_re, w_multi_beamCont_re[rx_m_idx][theta_idx], Rx_M * sizeof(float));
        printf("Find the configuration rx_m_idx = %d, theta_idx = %d\n", rx_m_idx, theta_idx);
    }
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float) + AVX * sizeof(float));
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
                    a_vector_re[j*Rx_M_y+k] = a_vector_re[j*Rx_M_y+k]*w_comebine_beam_re[j*Rx_M_y+k];
                    a_vector_im[j*Rx_M_y+k] = a_vector_im[j*Rx_M_y+k]*w_comebine_beam_re[j*Rx_M_y+k];
                    // printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
                    
                }
            }
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("\tS_MUSIC(%f,%f), ", S_spatial_spectrum_re[i], S_spatial_spectrum_im[i]);
            spatial_spectrum_value_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_spatial_spectrum_re[elevation*search_len_phi + azimuth], &S_spatial_spectrum_im[elevation*search_len_phi + azimuth]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[elevation*search_len_phi + azimuth]);
        }
    }
    free(a_vector_re);free(a_vector_im);
    free(S_spatial_spectrum_re);free(S_spatial_spectrum_im);
    free(w_comebine_beam_re);
}

// =========== find peak ==========
void find_spatial_spectrum_peaks_3D(float *spatial_spectrum_value_dB, int *position_theta, int *position_phi, int search_len_theta, int search_len_phi, int len_t_angle) {
    search_count += search_len_theta * search_len_phi;
    float max_temp = -999999.0;
    len_t_angle = len_t_angle;
    for (int i = 0; i < search_len_theta; ++i)
    {
        for (int j = 0; j < search_len_phi; ++j)
        {

            if (spatial_spectrum_value_dB[i*search_len_phi + j] > max_temp)
            {
                max_temp = spatial_spectrum_value_dB[i*search_len_phi + j];
                position_theta[0] = i;
                position_phi[0] = j;
            }
        }
    }
}



// ===== Calculate search angle ====
void calculate_search_theta_3D(float search_theta_deg_prev, int *search_len_theta_current, 
                            float **search_theta_deg_current, float **search_theta_rad_current, 
                            float search_step_theta_current) {
    float search_start_theta_current;
    //printf("search_step_theta_current = %.2f\n", search_step_theta_current);
    if (search_theta_deg_prev == 0) {
        search_start_theta_current = 0.0;
        *search_len_theta_current = 3;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else if ((90 - search_theta_deg_prev) <= 2 * search_step_theta_current) {
        search_start_theta_current = 90 - 2 * search_step_theta_current;
        *search_len_theta_current = 3;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current - 1; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
        (*search_theta_deg_current)[*search_len_theta_current - 1] = (*search_theta_deg_current)[*search_len_theta_current - 2] + search_step_theta_current / 2;
        (*search_theta_rad_current)[*search_len_theta_current - 1] = (*search_theta_deg_current)[*search_len_theta_current - 1] * PI / 180;
        //printf("(*search_theta_deg_current)[%d] = %.1f\n", *search_len_theta_current - 1, (*search_theta_rad_current)[*search_len_theta_current - 1]);
    } else {
        search_start_theta_current = search_theta_deg_prev - 2 * search_step_theta_current;
        *search_len_theta_current = 5;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    }
    //printf("search_start_theta_current = %.1f\n", search_start_theta_current);
}

void calculate_search_phi_3D(float search_phi_deg_prev, int *search_len_phi_current, 
                          float **search_phi_deg_current, float **search_phi_rad_current, 
                          float search_step_phi_current) {
    float search_start_phi_current;
    if (search_phi_deg_prev == -60) {
        search_start_phi_current = -60;
        *search_len_phi_current = 3;
        *search_phi_deg_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        *search_phi_rad_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_phi_current; ++i) {
            (*search_phi_deg_current)[i] = search_start_phi_current + search_step_phi_current * i;
            (*search_phi_rad_current)[i] = (*search_phi_deg_current)[i] * PI / 180;
            //printf("(*search_phi_deg_current)[%d] = %.1f\n", i, (*search_phi_deg_current)[i]);
        }
    } else if ((60 - search_phi_deg_prev) <= 2 * search_step_phi_current) {
        search_start_phi_current = 60 - 2 * search_step_phi_current;
        *search_len_phi_current = 3;
        *search_phi_deg_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        *search_phi_rad_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_phi_current; ++i) {
            (*search_phi_deg_current)[i] = search_start_phi_current + search_step_phi_current * i;
            //printf("(*search_phi_deg_current)[%d] = %f\n", i, (*search_phi_deg_current)[i]);
            (*search_phi_rad_current)[i] = (*search_phi_deg_current)[i] * PI / 180;
            //printf("(*search_phi_deg_current)[%d] = %.1f\n", i, (*search_phi_deg_current)[i]);
        }
    } else {
        search_start_phi_current = search_phi_deg_prev - 2 * search_step_phi_current;
        *search_len_phi_current = 5;
        *search_phi_deg_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        *search_phi_rad_current = (float *)malloc(*search_len_phi_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_phi_current; ++i) {
            (*search_phi_deg_current)[i] = search_start_phi_current + search_step_phi_current * i;
            (*search_phi_rad_current)[i] = (*search_phi_deg_current)[i] * PI / 180;
            //printf("(*search_phi_deg_current)[%d] = %.1f\n", i, (*search_phi_deg_current)[i]);
            //printf("(*search_phi_deg_current)[%d] = %f\n", i, (*search_phi_deg_current)[i]);
        }
    }
    //printf("search_start_phi_current = %.1f\n", search_start_phi_current);
}


// ================================
// ====== spatial spectrum ML =====
// ================================

void calculate_spatial_spectrum_ML_3D(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float* S_ML_dB){
    int Rx_M_x = search_const->Rx_M_x;
    int Rx_M_y = search_const->Rx_M_y;
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;

    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float)+ AVX * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *a_temp_re = (float *)malloc(Rx_M * sizeof(float)+ AVX * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *S_ML_re = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX * sizeof(float));
    float *S_ML_im = (float *)malloc(search_len_theta*search_len_phi * sizeof(float)+ AVX * sizeof(float));
    
    float *theta_re = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *theta_im = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA_im = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(Rx_M * Rx_M * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float)+ AVX * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float)+ AVX * sizeof(float));
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2;

    //float total_time_Generalized_Inverse = 0;
    //float total_time_Orthogonal_Projection = 0;
    //-------------------------------------------------------------------
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        for (int azimuth = 0; azimuth < search_len_phi; ++azimuth){
            for (int j = 0; j < Rx_M_x; ++j){ // steering vector
                for (int k = 0; k < Rx_M_y; ++k){ // steering vector
                    cpp_exp2_3D(&a_vector_re[j*Rx_M_y+k], &a_vector_im[j*Rx_M_y+k], search_theta_rad, search_phi_rad, d, kc, elevation, azimuth, j - Rx_M_x_delta, k - Rx_M_y_delta);
                    //printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);       
                }
            }

            memcpy(a_temp_re,a_vector_re,(Rx_M * 1 * sizeof(float)));
            memcpy(a_temp_im,a_vector_im,(Rx_M * 1 * sizeof(float)));

            //gettimeofday(&time_Generalized_Inverse_start, NULL);
            //a_vector_conjugate_transpose A^H
            complex_matrix_conjugate_transpose(a_temp_re, a_temp_im, 1, Rx_M);   
            //A^H*A =1*1  
            complex_matrix_multiplication( a_temp_re, a_temp_im, a_vector_re, a_vector_im, AH_mulA_re, AH_mulA_im, 1, Rx_M, 1);
            //[inv(A^H*A)] =1*1   
            matrix_inverse_LU(AH_mulA_re, AH_mulA_im, AH_mulA_inv_re, AH_mulA_inv_im, 1);

            //gettimeofday(&time_Generalized_Inverse_end, NULL);
            // 計算 Generalized Inverse Matrix 的執行時間
            //total_time_Generalized_Inverse += (time_Generalized_Inverse_end.tv_sec - time_Generalized_Inverse_start.tv_sec) * 1000000
                                            //+ (time_Generalized_Inverse_end.tv_usec - time_Generalized_Inverse_start.tv_usec);

            //gettimeofday(&time_Orthogonal_Projection_start, NULL);
            // [inv(A^H*A)]*A^H = 1*M
            complex_matrix_multiplication(AH_mulA_inv_re, AH_mulA_inv_im, a_temp_re, a_temp_im, AH_mulA2_re, AH_mulA2_im, 1, 1, Rx_M);
            // P_A = A*[inv(A^H*A)]*A^H = M*M  
            complex_matrix_multiplication(a_vector_re, a_vector_im, AH_mulA2_re, AH_mulA2_im, AH_mulA3_re, AH_mulA3_im, Rx_M, 1, Rx_M);
            // P_A*R = M*M Pn = Rxx
            complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, search_const->Pn_re, search_const->Pn_im, theta_re, theta_im, Rx_M, Rx_M, Rx_M);
            // trace[P_A*R] 
            trace(theta_re, theta_im, S_ML_re, S_ML_im, Rx_M, Rx_M, elevation*search_len_phi + azimuth);
            S_ML_dB[elevation*search_len_phi + azimuth] = cpp_20log_abs(&S_ML_re[elevation*search_len_phi + azimuth], &S_ML_im[elevation*search_len_phi + azimuth]);
            //gettimeofday(&time_Orthogonal_Projection_end, NULL);
            // 計算 Orthogonal Projection Matrix 的執行時間
            //total_time_Orthogonal_Projection += (time_Orthogonal_Projection_end.tv_sec - time_Orthogonal_Projection_start.tv_sec) * 1000000
                                           // + (time_Orthogonal_Projection_end.tv_usec - time_Orthogonal_Projection_start.tv_usec);
        }
    }
    // Free dynamically allocated memory
    free(a_vector_re);
    free(a_vector_im);
    free(a_temp_re);
    free(a_temp_im);
    free(S_ML_re);
    free(S_ML_im);

    free(theta_re);
    free(theta_im);

    free(AH_mulA_re);
    free(AH_mulA_im);
    free(AH_mulA2_re);
    free(AH_mulA2_im);
    free(AH_mulA3_re);
    free(AH_mulA3_im);
    free(AH_mulA_inv_re);
    free(AH_mulA_inv_im);
}

// ================================
// ========== save data ===========
// ================================

void save_Spectrum_to_csv(const char* filename, float* spatial_spectrum_value_dB, int len_dth) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Failed to open file\n");
        return;
    }

    for (int i = 0; i < len_dth; ++i) {
        fprintf(fp, "%.4f\n", spatial_spectrum_value_dB[i]);
    }

    fclose(fp);
    printf("Spectrum values successfully written to CSV file !\n");
}


