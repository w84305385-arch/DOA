#include <stdint.h>
#ifndef SPATIAL_SPECTRUM_INT32_H
#define SPATIAL_SPECTRUM_INT32_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;
extern int search_count;

int q_shift_DOA;
typedef struct
{
    int Rx_M_x;
    int Rx_M_y;
    int Rx_M;
    float d;
    float kc;
    int32_t *Pn_re;
    int32_t *Pn_im;
} SearchConst_int32;


void spatial_spectrum_value_int32(int32_t *a_vector_re, int32_t *a_vector_im, int32_t *Pn_re, int32_t *Pn_im, int M, float *result_re, float *result_im);


void calculate_spatial_spectrum_3D_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB    
);
void calculate_spatial_spectrum_3D_multiBeam_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB,
    float search_step_theta 
);
#endif