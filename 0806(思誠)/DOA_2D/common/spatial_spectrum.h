#ifndef SPATIAL_SPECTRUM_H
#define SPATIAL_SPECTRUM_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;
extern int search_count;

typedef struct
{
    int Rx_M_x;
    int Rx_M_y;
    int Rx_M;
    float d;
    float kc;
    float *Pn_re;
    float *Pn_im;
} SearchConst;

// spatial spectrum
void spatial_spectrum_value(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *result_re, float *result_im);

void calculate_spatial_spectrum(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* spatial_spectrum_value_dB);

int find_index(const int *array, int size, int value);

void calculate_spatial_spectrum_PS(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* spatial_spectrum_value_dB, float search_step_theta);

void find_spatial_spectrum_peaks(float *spatial_spectrum_value_dB, int *position_theta, int search_len_theta, int len_t_angle);
void find_spatial_spectrum_peaks_multi(
    float *spatial_spectrum_value_dB,
    int *position_theta,
    int *position_phi,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_deg,
    float *search_phi_deg,
    int *peak_number
);



void calculate_search_theta(
    float search_theta_deg_prev, 
    int *search_len_theta_current, 
    float **search_theta_deg_current, 
    float **search_theta_rad_current, 
    float search_step_theta_current
);
void calculate_search_theta_high_accuracy(float search_theta_deg_prev, int *search_len_theta_current, 
    float **search_theta_deg_current, float **search_theta_rad_current, 
    float search_step_theta_current);
void calculate_search_theta_last(float search_theta_deg_prev, int *search_len_theta_current, 
    float **search_theta_deg_current, float **search_theta_rad_current, 
    float search_step_theta_current);

void calculate_search_phi_3D(
    float search_phi_deg_prev, 
    int *search_len_phi_current, 
    float **search_phi_deg_current, 
    float **search_phi_rad_current, 
    float search_step_phi_current
);
void calculate_search_theta_last_3D(
    float search_theta_deg_prev, 
    int *search_len_theta_current, 
    float **search_theta_deg_current, 
    float **search_theta_rad_current, 
    float search_step_theta_current);
void calculate_search_phi_last_3D(
    float search_phi_deg_prev, 
    int *search_len_phi_current, 
    float **search_phi_deg_current, 
    float **search_phi_rad_current, 
    float search_step_phi_current
);

// spatial spectrum ML
void calculate_spatial_spectrum_ML(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* S_ML_dB);
void calculate_spatial_spectrum_ML_PS(SearchConst *search_const, int search_len_theta, float *search_theta_rad, float* S_ML_dB, float search_step_theta);
// save data
void save_Spectrum_to_csv(const char* filename, float* spatial_spectrum_value_dB, int len_dth);

#endif