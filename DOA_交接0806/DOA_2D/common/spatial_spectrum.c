//--------------------
#define PI acos(-1)
#define AVX 16            
//--------------------
#include <immintrin.h>
#include "math_func.h"
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
float total_spatial_spectrum_value_time_c = 0.0;
void spatial_spectrum_value(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *result_re, float *result_im)
{
    // struct timeval time_spatial_spectrum_value_start, time_spatial_spectrum_value_end, time_spatial_spectrum_value_diff; // time initial
    // float time_spatial_spectrum_value = 0.0;
    //---------------------------------------------------------------
    float *Pn_a_vector_temp_re = (float *)malloc(M * sizeof(float));
    float *Pn_a_vector_temp_im = (float *)malloc(M * sizeof(float));
    float *spatial_spectrum_value_temp_re = (float *)malloc(M * sizeof(float));
    float *spatial_spectrum_value_temp_im = (float *)malloc(M * sizeof(float));
    // printf("temp = (%f,%f)\n", *temp_re, *temp_im);
    //---------------------------------------------------------------
    // gettimeofday(&time_spatial_spectrum_value_start, NULL);
    complex_matrix_multiplication(Pn_re, Pn_im, a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector_re, a_vector_im, M, 1);
    complex_matrix_multiplication(a_vector_re, a_vector_im, Pn_a_vector_temp_re, Pn_a_vector_temp_im, spatial_spectrum_value_temp_re, spatial_spectrum_value_temp_im, 1, M, 1);
    // 乘法時間計算
    // gettimeofday(&time_spatial_spectrum_value_end, NULL);
    // timersub(&time_spatial_spectrum_value_end, &time_spatial_spectrum_value_start, &time_spatial_spectrum_value_diff);
    // total_spatial_spectrum_value_time_c += time_spatial_spectrum_value_diff.tv_sec * 1000000 + time_spatial_spectrum_value_diff.tv_usec;
    // time_spatial_spectrum_value = time_spatial_spectrum_value_diff.tv_sec * 1000000 + time_spatial_spectrum_value_diff.tv_usec;
    // printf("time_spatial_spectrum_value = %.3f(us)\n", time_spatial_spectrum_value);
    cpp_division2(1, 0, &spatial_spectrum_value_temp_re[0], &spatial_spectrum_value_temp_im[0], result_re, result_im);

    // printf("result = (%.5f,%.5f)\n", *result_re, *result_re);
}

void calculate_spatial_spectrum(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* spatial_spectrum_value_dB){
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;

    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float));
    float *spatial_spectrum_value_re = (float *)malloc(search_len_theta * sizeof(float));
    float *spatial_spectrum_value_im = (float *)malloc(search_len_theta * sizeof(float));
    float Rx_M_delta = ((float)Rx_M-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta

            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M; ++j){ // steering vector
                cpp_exp2(&a_vector_re[j], &a_vector_im[j], search_theta_rad, d, kc, elevation, j - Rx_M_delta);
                //printf("a_vector(%f,%f)\n", a_vector_re[j*Rx_M_y+k], a_vector_im[j*Rx_M_y+k]);
            }
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &spatial_spectrum_value_re[elevation], &spatial_spectrum_value_im[elevation]);
            // printf("spatial_spectrum_value(%d) = (%f,%f), ", elevation, spatial_spectrum_value_re[elevation], spatial_spectrum_value_im[elevation]);
            // printf("\tspatial_spectrum_value(%f,%f), ", spatial_spectrum_value_re[i], spatial_spectrum_value_im[i]);
            spatial_spectrum_value_dB[elevation] = cpp_20log_abs(&spatial_spectrum_value_re[elevation], &spatial_spectrum_value_im[elevation]);
            // printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[elevation]);

    }
    // printf("--------------------------------------\n");
    // printf("Total spatial spectrum value time : \t%.3f(us)\n", total_spatial_spectrum_value_time_c);
    // printf("Ave spatial spectrum value time : \t%.3f(us)\n", total_spatial_spectrum_value_time_c/121);
    free(a_vector_re);free(a_vector_im);
    free(spatial_spectrum_value_re);free(spatial_spectrum_value_im);
}


// combine beam weight table index
const int rx_m_index[] = {64, 32, 16};
const int theta_index[] = {30, 10};
int find_index(const int *array, int size, int value) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == value) return i;
    }
    return -1; // 
}
void calculate_spatial_spectrum_PS(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* spatial_spectrum_value_dB, float search_step_theta){
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;

    // 依據Rx_M,search_step_theta 選擇不同的權重
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    // 找到索引
    int rx_m_idx = find_index(rx_m_index, 3, Rx_M);
    int theta_idx = find_index(theta_index, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || theta_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f; // 設置為 1.0
        }
        printf("找不到配置\n");
    }else{
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][theta_idx], Rx_M * sizeof(float));
        printf("找到配置 rx_m_idx = %d, theta_idx = %d\n", rx_m_idx, theta_idx);
    }
    //print_complex_matrix(w_multi_beam_re, w_multi_beam_im, Rx_M, 1);
    //-------------------------------------------------------------------
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float));
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float));
    float *spatial_spectrum_value_re = (float *)malloc(search_len_theta * sizeof(float));
    float *spatial_spectrum_value_im = (float *)malloc(search_len_theta * sizeof(float));
    float Rx_M_delta = ((float)Rx_M-1)/2;
    // calculate Spatial Spectrum
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta

            //printf("azimuth, elevation = %d, %d\n", azimuth, elevation);
            // can be paralleled to compute spatial_spectrum_value_dB
            for (int j = 0; j < Rx_M; ++j){ // steering vector
                cpp_exp2(&a_vector_re[j], &a_vector_im[j], search_theta_rad, d, kc, elevation, j - Rx_M_delta);
                //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
                a_vector_re[j] = a_vector_re[j]*w_multi_beam_re[j];
                a_vector_im[j] = a_vector_im[j]*w_multi_beam_re[j];
                
            }
            //printf("\n");
            //printf("a_vector_re combine\n");
            //print_complex_matrix(a_vector_re, a_vector_im, Rx_M, 1);
            spatial_spectrum_value(a_vector_re, a_vector_im, search_const->Pn_re, search_const->Pn_im, Rx_M, &spatial_spectrum_value_re[elevation], &spatial_spectrum_value_im[elevation]);
            //printf("\tspatial_spectrum_value(%f,%f), ", spatial_spectrum_value_re[i], spatial_spectrum_value_im[i]);
            spatial_spectrum_value_dB[elevation] = cpp_20log_abs(&spatial_spectrum_value_re[elevation], &spatial_spectrum_value_im[elevation]);
            //printf("spatial_spectrum_value_dB = %.4f\n", spatial_spectrum_value_dB[elevation]);

    }
    free(a_vector_re);free(a_vector_im);
    free(spatial_spectrum_value_re);free(spatial_spectrum_value_im);
}

// =========== find peak ==========
void find_spatial_spectrum_peaks(float *spatial_spectrum_value_dB, int *position_theta, int search_len_theta, int len_t_angle) {
    float max_temp = spatial_spectrum_value_dB[0];
    position_theta[0] = 0;
    // printf("spatial_spectrum_value = \n");
    for (int i = 0; i < search_len_theta; ++i)
    {
        // printf("%.3f\n",spatial_spectrum_value_dB[i]);
        if (spatial_spectrum_value_dB[i] > max_temp)
        {
            max_temp = spatial_spectrum_value_dB[i];
            //printf("max_temp = %.4f\n", max_temp);
            position_theta[0] = i;
        }
    }
    // printf("\n");
    search_count += search_len_theta;
}

// ===== Calculate search angle ====
void calculate_search_theta(float search_theta_deg_prev, int *search_len_theta_current, 
                            float **search_theta_deg_current, float **search_theta_rad_current, 
                            float search_step_theta_current) {
    float search_start_theta_current;
    if (search_theta_deg_prev == -60) {
        search_start_theta_current = -60.0;
        *search_len_theta_current = 3;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else if ((60 - search_theta_deg_prev) <= 2 * search_step_theta_current) {
        search_start_theta_current = 60 - 2 * search_step_theta_current;
        *search_len_theta_current = 3;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
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

    //printf("search_len_theta_current = %d\n", *search_len_theta_current);
}

void calculate_search_theta_high_accuracy(float search_theta_deg_prev, int *search_len_theta_current, 
                            float **search_theta_deg_current, float **search_theta_rad_current, 
                            float search_step_theta_current) {
    float search_start_theta_current;
    if (search_theta_deg_prev == -60) {
        search_start_theta_current = -60.0;
        *search_len_theta_current = 4;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else if ((60 - search_theta_deg_prev) <= 3 * search_step_theta_current) {
        search_start_theta_current = 60 - 3 * search_step_theta_current;
        *search_len_theta_current = 4;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else {
        search_start_theta_current = search_theta_deg_prev - 1.5 * search_step_theta_current;
        *search_len_theta_current = 4;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    }

    //printf("search_len_theta_current = %d\n", *search_len_theta_current);
}

void calculate_search_theta_last(float search_theta_deg_prev, int *search_len_theta_current, 
                            float **search_theta_deg_current, float **search_theta_rad_current, 
                            float search_step_theta_current) {
    float search_start_theta_current;
    if (search_theta_deg_prev == -60) {
        search_start_theta_current = -60.0;
        *search_len_theta_current = 6;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else if ((60 - search_theta_deg_prev) <= 5 * search_step_theta_current) {
        search_start_theta_current = 60 - 5 * search_step_theta_current;
        *search_len_theta_current = 11;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    } else {
        search_start_theta_current = search_theta_deg_prev - 5 * search_step_theta_current;
        *search_len_theta_current = 11;
        *search_theta_deg_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        *search_theta_rad_current = (float *)malloc(*search_len_theta_current * sizeof(float));
        // Search angle
        for (int i = 0; i < *search_len_theta_current; ++i) {
            (*search_theta_deg_current)[i] = search_start_theta_current + search_step_theta_current * i;
            (*search_theta_rad_current)[i] = (*search_theta_deg_current)[i] * PI / 180;
            //printf("(*search_theta_deg_current)[%d] = %.1f\n", i, (*search_theta_deg_current)[i]);
        }
    }

    //printf("search_len_theta_current = %d\n", *search_len_theta_current);
}


// ================================
// ====== spatial spectrum ML =====
// ================================

void calculate_spatial_spectrum_ML(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* S_ML_dB){
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    //print_complex_matrix(w_multi_beam_re, w_multi_beam_im, Rx_M, 1);
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float));
    float *a_temp_re = (float *)malloc(Rx_M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(Rx_M * sizeof(float));
    float *S_ML_re = (float *)malloc(search_len_theta * sizeof(float));
    float *S_ML_im = (float *)malloc(search_len_theta * sizeof(float));
    
    float *theta_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *theta_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * Rx_M * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * Rx_M * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    float Rx_M_delta = ((float)Rx_M-1)/2;
    // 累積時間變數，單位為微秒
    float total_time_Generalized_Inverse = 0;
    float total_time_Orthogonal_Projection = 0;
    //-------------------------------------------------------------------
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        
            for (int j = 0; j < Rx_M; ++j){ // steering vector
                cpp_exp2(&a_vector_re[j], &a_vector_im[j], search_theta_rad, d, kc, elevation, j - Rx_M_delta);
                //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
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
            // P_A*R = M*M 
            complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, search_const->Pn_re, search_const->Pn_im, theta_re, theta_im, Rx_M, Rx_M, Rx_M);
            // trace[P_A*R] 
            trace(theta_re, theta_im, S_ML_re, S_ML_im, Rx_M, Rx_M, elevation);
            S_ML_dB[elevation] = cpp_20log_abs(&S_ML_re[elevation], &S_ML_im[elevation]);
            //gettimeofday(&time_Orthogonal_Projection_end, NULL);
            // 計算 Orthogonal Projection Matrix 的執行時間
            //total_time_Orthogonal_Projection += (time_Orthogonal_Projection_end.tv_sec - time_Orthogonal_Projection_start.tv_sec) * 1000000
                                           // + (time_Orthogonal_Projection_end.tv_usec - time_Orthogonal_Projection_start.tv_usec);
        
    }
    // Free dynamically allocated memory
    free(a_vector_re);free(a_vector_im);
    free(a_temp_re);free(a_temp_im);
    free(S_ML_re);free(S_ML_im);

    free(theta_re);free(theta_im);

    free(AH_mulA_re);free(AH_mulA_im);
    free(AH_mulA2_re);free(AH_mulA2_im);
    free(AH_mulA3_re);free(AH_mulA3_im);
    free(AH_mulA_inv_re);free(AH_mulA_inv_im);
}

void calculate_spatial_spectrum_ML_PS(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* S_ML_dB, float search_step_theta){
    int Rx_M = search_const->Rx_M;
    float d = search_const->d;
    float kc = search_const->kc;
    // 依據Rx_M,search_step_theta 選擇不同的權重
    float *w_multi_beam_re = (float*)calloc(Rx_M, sizeof(float));
    float *w_multi_beam_im = (float*)calloc(Rx_M, sizeof(float));
    // 找到索引
    int rx_m_idx = find_index(rx_m_index, 3, Rx_M);
    int theta_idx = find_index(theta_index, 2, search_step_theta);//search_step_theta
    if (rx_m_idx == -1 || theta_idx == -1) {
        for (int i = 0; i < Rx_M; i++) {
            w_multi_beam_re[i] = 1.0f; // 設置為 1.0
        }
        //printf("Configuration not found\n");
    }else{
        memcpy(w_multi_beam_re, w_multi_beamCont_re[rx_m_idx][theta_idx], Rx_M * sizeof(float));
        //printf("Found the configuration rx_m_idx = %d, theta_idx = %d\n", rx_m_idx, theta_idx);
    }
    //print_complex_matrix(w_multi_beam_re, w_multi_beam_im, Rx_M, 1);
    float *a_vector_re = (float *)malloc(Rx_M * sizeof(float));  //a_vector = M*1
    float *a_vector_im = (float *)malloc(Rx_M * sizeof(float));
    float *a_temp_re = (float *)malloc(Rx_M * sizeof(float));    //a_vector^H = 1*M
    float *a_temp_im = (float *)malloc(Rx_M * sizeof(float));
    float *S_ML_re = (float *)malloc(search_len_theta * sizeof(float));
    float *S_ML_im = (float *)malloc(search_len_theta * sizeof(float));
    
    float *theta_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *theta_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    //-------------------------------------------------------//
    float *AH_mulA_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA2_re = (float *)malloc(1 * Rx_M * sizeof(float));
    float *AH_mulA2_im = (float *)malloc(1 * Rx_M * sizeof(float));
    float *AH_mulA3_re = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA3_im = (float *)malloc(Rx_M * Rx_M * sizeof(float));
    float *AH_mulA_inv_re = (float *)malloc(1 * 1 * sizeof(float));
    float *AH_mulA_inv_im = (float *)malloc(1 * 1 * sizeof(float));
    float Rx_M_delta = ((float)Rx_M-1)/2;
    // 累積時間變數，單位為微秒
    float total_time_Generalized_Inverse = 0;
    float total_time_Orthogonal_Projection = 0;
    //-------------------------------------------------------------------
    for (int elevation = 0; elevation < search_len_theta; ++elevation){//theta
        
            for (int j = 0; j < Rx_M; ++j){ // steering vector
                cpp_exp2(&a_vector_re[j], &a_vector_im[j], search_theta_rad, d, kc, elevation, j - Rx_M_delta);
                //printf("a_vector(%f,%f)\n", a_vector_re[j], a_vector_im[j]);
                a_vector_re[j] = a_vector_re[j]*w_multi_beam_re[j];
                a_vector_im[j] = a_vector_im[j]*w_multi_beam_re[j];
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
            // P_A*R = M*M 
            complex_matrix_multiplication(AH_mulA3_re, AH_mulA3_im, search_const->Pn_re, search_const->Pn_im, theta_re, theta_im, Rx_M, Rx_M, Rx_M);
            // trace[P_A*R] 
            trace(theta_re, theta_im, S_ML_re, S_ML_im, Rx_M, Rx_M, elevation);
            S_ML_dB[elevation] = cpp_20log_abs(&S_ML_re[elevation], &S_ML_im[elevation]);
            //gettimeofday(&time_Orthogonal_Projection_end, NULL);
            // 計算 Orthogonal Projection Matrix 的執行時間
            //total_time_Orthogonal_Projection += (time_Orthogonal_Projection_end.tv_sec - time_Orthogonal_Projection_start.tv_sec) * 1000000
                                           // + (time_Orthogonal_Projection_end.tv_usec - time_Orthogonal_Projection_start.tv_usec);
        
    }
    // Free dynamically allocated memory
    free(w_multi_beam_re);free(w_multi_beam_im);
    free(a_vector_re);free(a_vector_im);
    free(a_temp_re);free(a_temp_im);
    free(S_ML_re);free(S_ML_im);

    free(theta_re);free(theta_im);

    free(AH_mulA_re);free(AH_mulA_im);
    free(AH_mulA2_re);free(AH_mulA2_im);
    free(AH_mulA3_re);free(AH_mulA3_im);
    free(AH_mulA_inv_re);free(AH_mulA_inv_im);
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


