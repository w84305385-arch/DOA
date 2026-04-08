#ifndef GENERATE_SIGNAL_3D_H
#define GENERATE_SIGNAL_3D_H
#include "doa_parameters.h"
// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;


// generate signal
void generate_Rx_incident_sig_3D(float* total_incident_sig_re,float*  total_incident_sig_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, int nd, float t_theta_deg, float t_phi_deg, float  Tx_beamwidth, float  Tx_beamwidth_samples);
void generate_Tx_signal_3D(float* Tx_sig_re, float* Tx_sig_im, int Tx_M_x, int Tx_M_y, float d, float kc, int nd, float t_theta_deg, float t_phi_deg);
void generate_Directional_Channel_3D(float* H_re, float* H_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, float t_theta_deg, float t_phi_deg, float Tx_beamwidth, int Tx_beamwidth_samples);
void generate_Directional_Channel_3D_1(float* H_re, float* H_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, float t_theta_deg, float t_phi_deg, int Tx_beamwidth_samples);
void generate_incident_signal_3D(float* Tx_sig_re, float* Tx_sig_im, float* H_re, float* H_im, float* Coherent_sig_re, float* Coherent_sig_im, int Rx_M, int Tx_M, int nd);
void generate_Rx_signal(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx);

// generate signal for original version
void generate_Rx_signal_original(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx);
#endif