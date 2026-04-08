#ifndef TASK_H
#define TASK_H
typedef struct
{
    // ------------------------------------------------------------
    // Input Parameters
    // const data 
    SearchConst *search_const_input_data;

    // prev data
    float theta_deg_prev;
    float phi_deg_prev;

    // current data
    float search_step_theta;
    float search_step_phi;
    float PAPR_threshold;
    float search_threshold;

    // ------------------------------------------------------------
    // Output Results
    float *result_theta_deg;
    float *result_phi_deg;
    float *result_dB_value;
    int detected_peak_count;
} SearchThreadData;
typedef struct {
    SearchThreadData base;

    /* —— 這以下是「只給 MVDR 用」的新欄位 —— */
    int          use_subarray;          // 0=整陣列，1=子陣列加總
    int          N_SUBARRAY;           // 子陣列數量
    SearchConst *search_const_sub_tpl; // 4×4 SearchConst 原型
    float      **Pn_re_sub;            // [N_SUBARRAY] → 16×16 反矩陣
    float      **Pn_im_sub;
} SearchThreadData_MVDR;
typedef struct{
    float theta_deg;
    float phi_deg;
    float dB_value;
} search_peune;
#endif

#ifndef M_SPATIAL_SPECTRUM_H
#define M_SPATIAL_SPECTRUM_H
// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;
extern int flag_ind;
extern int basic_val;
extern int search_count;

extern int num_row_blocks;
extern const int rx_m_index[];
extern const int beam_deg_index[];
pid_t gettid();

void prune_and_search_worker_PAPR(void* arg);
void prune_and_search_worker_PAPR_MVDR(void *arg);
void prune_and_search_worker_PAPR_MVDR_sub(void *arg);
void prune_and_search_worker_max(void* arg);

void m_calculate_spatial_spectrum_3D_multiBeam(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float *spatial_spectrum_value_dB, 
    float search_step_theta
);

void m_calculate_spatial_spectrum_3D(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,  
    float *spatial_spectrum_value_dB
);

void m_compute_spatial_spectrum_value_row_block_parallel(float *a_vector_re, float *a_vector_im, float *Pn_re, float *Pn_im, int M, float *music_Real, float *music_Imag);

void m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,
    float *spatial_spectrum_value_dB, 
    float search_step_theta
);

void m_calculate_spatial_spectrum_3D_row_block_parallel(
    SearchConst *search_const,
    int search_len_theta, 
    int search_len_phi, 
    float *search_theta_rad, 
    float *search_phi_rad,  
    float *spatial_spectrum_value_dB
);

void m_find_spatial_spectrum_peaks_3D_PAPR(
    float *spatial_spectrum_value_dB,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_deg,
    float *search_phi_deg,
    float PAPR_threshold,
    // output
    float *result_theta_deg,
    float *result_phi_deg,
    float *result_dB_value,
    int *detected_peak_count
);

void m_find_spatial_spectrum_peaks_3D_max(
    float *spatial_spectrum_value_dB,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_deg,
    float *search_phi_deg,
    // output
    float *result_theta_deg,
    float *result_phi_deg,
    float *result_dB_value,
    int *detected_peak_count
);

void select_top_peaks(
    search_peune* input_peaks, int input_count,
    search_peune* result_peaks, int len_t_angle
);


void free_SearchThreadData(SearchThreadData *td);
#endif



