
#ifndef TASK_INT32_H
#define TASK_INT32_H
#include "spatial_spectrum_int32.h"
typedef struct
{
    // ------------------------------------------------------------
    // Input Parameters
    // const data 
    SearchConst_int32 *search_const_input_data;

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
} SearchThreadData_int32;
typedef struct {
    SearchThreadData_int32 base;

    /* —— 這以下是「只給 MVDR 用」的新欄位 —— */
    int          use_subarray;          // 0=整陣列，1=子陣列加總
    int          N_SUBARRAY;           // 子陣列數量
    SearchConst_int32 *search_const_sub_tpl; // 4×4 SearchConst 原型
    int32_t     **Pn_re_sub;            // [N_SUBARRAY] → 16×16 反矩陣
    int32_t      **Pn_im_sub;
} SearchThreadData_MVDR_int32;

#endif

#ifndef M_SPATIAL_SPECTRUM_INT32_H
#define M_SPATIAL_SPECTRUM_INT32_H
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

void prune_and_search_worker_PAPR_int32(void* arg);
void prune_and_search_worker_PAPR_MVDR_int32(void *arg);
void prune_and_search_worker_PAPR_MVDR_sub_int32(void *arg);
void prune_and_search_worker_max_int32(void* arg);


void m_compute_spatial_spectrum_value_row_block_parallel_int32(int32_t *a_vector_re, int32_t *a_vector_im, int32_t *Pn_re, int32_t *Pn_im, int M, float *result_re, float *result_im);

void m_calculate_spatial_spectrum_3D_multiBeam_row_block_parallel_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB,
    float search_step_theta 
);

void m_calculate_spatial_spectrum_3D_row_block_parallel_int32(
    SearchConst_int32 *search_const,
    int search_len_theta,
    int search_len_phi,
    float *search_theta_rad,
    float *search_phi_rad,
    float* spatial_spectrum_value_dB
);




void free_SearchThreadData_int32(SearchThreadData_int32 *td);
#endif



