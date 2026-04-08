//--------------------
#define PI acos(-1)
#define DEG2RAD (PI/180.0)
#define AVX 16            
//--------------------
#include <immintrin.h>
#include "generate_signal_3D.h"
#include "math_func_3D.h"
#include "complex_matrix_ops.h"
#include "doa_parameters.h"
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
// ======== generate signal =======
// ================================

void generate_Rx_incident_sig_3D(float* total_incident_sig_re,float*  total_incident_sig_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, int nd, float t_theta_deg, float t_phi_deg, float  Tx_beamwidth,float  Tx_beamwidth_samples){
    int Tx_M = Tx_M_x*Tx_M_y;
    int Rx_M = Rx_M_x*Rx_M_y;
    //---------------------------------------------------------------
    // generate Tx signal
    float *Tx_sig_re = (float *)malloc(nd * Tx_M * sizeof(float)+ AVX*sizeof(float));
    float *Tx_sig_im = (float *)malloc(nd * Tx_M * sizeof(float)+ AVX*sizeof(float));
    generate_Tx_signal_3D(Tx_sig_re, Tx_sig_im, Tx_M_x, Tx_M_y, d, kc, nd, t_theta_deg, t_phi_deg);//t_theta_deg, t_phi_deg

    //printf("---------------Tx_sig---------------\n");
    //print_complex_matrix(Tx_sig_re, Tx_sig_im, nd, Tx_M);
    //printf("t_theta_deg = %.2f\n", t_theta_deg);
    //printf("t_phi_deg = %.2f\n", t_phi_deg);
    // generate beamwidth directional channel
    float *H_re = (float *)calloc(Rx_M * Tx_M , sizeof(float)+ AVX*sizeof(float));
    float *H_im = (float *)calloc(Rx_M * Tx_M , sizeof(float)+ AVX*sizeof(float));
    if(Tx_beamwidth_samples == 1){
        generate_Directional_Channel_3D_1(H_re, H_im, Tx_M_x, Tx_M_y, Rx_M_x, Rx_M_y, d, kc, t_theta_deg, t_phi_deg, Tx_beamwidth_samples);
    }else{
        generate_Directional_Channel_3D(H_re, H_im, Tx_M_x, Tx_M_y, Rx_M_x, Rx_M_y, d, kc, t_theta_deg, t_phi_deg, Tx_beamwidth, Tx_beamwidth_samples);
    }

    //printf("---------------H---------------\n");
    //print_complex_matrix(H_re, H_im, Rx_M, Tx_M);


    // ==================================================================
    // ========================= gNB Rx model ===========================
    // ==================================================================
    //---------------------------------------------------------------
    // generate incident signal
    float *incident_sig_re = (float *)malloc(nd * Rx_M * sizeof(float)+ AVX*sizeof(float));
    float *incident_sig_im = (float *)malloc(nd * Rx_M * sizeof(float)+ AVX*sizeof(float));
    generate_incident_signal_3D(Tx_sig_re, Tx_sig_im, H_re, H_im, incident_sig_re, incident_sig_im, Rx_M, Tx_M, nd);
    
    for(int j=0; j<Rx_M*nd; j++){
        total_incident_sig_re[j] += incident_sig_re[j];
        total_incident_sig_im[j] += incident_sig_im[j];
        //printf("total_incident_sig_re[%d] = %.2f\n", j, total_incident_sig_re[j]);
    }
    
    
    free(Tx_sig_re);
    free(Tx_sig_im);
    free(H_re);
    free(H_im);
    free(incident_sig_re);
    free(incident_sig_im);
}

void generate_Tx_signal_3D(float* Tx_sig_re, float* Tx_sig_im, int Tx_M_x, int Tx_M_y, float d, float kc, int nd, float t_theta_deg, float t_phi_deg){
    //---------------------------------------------------------------
    float *tx_a_re = (float *)malloc((Tx_M_x * Tx_M_y) * 1 * sizeof(float)+ AVX*sizeof(float));
    float *tx_a_im = (float *)malloc((Tx_M_x * Tx_M_y) * 1 * sizeof(float)+ AVX*sizeof(float));
    //---------------------------------------------------------------
    //printf("Tx_M_x = %d\n", Tx_M_x);
    float Tx_M_x_delta = ((float)Tx_M_x-1)/2;
    float Tx_M_y_delta = ((float)Tx_M_y-1)/2;
    //printf("Tx_M_x_delta = %f\n", Tx_M_x_delta);
    //Tx_M_x_delta = 0;
    //Tx_M_y_delta = 0;
    //printf("t_theta_deg = %.2f, t_phi_deg = %.2f\n", t_theta_deg, t_phi_deg);
    for (int i = 0; i < Tx_M_x; ++i){   
        for(int j = 0; j < Tx_M_y; ++j){    
            //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
            //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
            //printf("i - Tx_M_x_delta = %.3f, j - Tx_M_y_delta = %.3f\n", i - Tx_M_x_delta, j - Tx_M_y_delta);
            cpp_exp_3D(&tx_a_re[i*Tx_M_y + j], &tx_a_im[i*Tx_M_y + j], t_theta_deg, t_phi_deg, d, kc, i - Tx_M_x_delta, j - Tx_M_y_delta);
            //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
            //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
        }
    }
    //print_complex_matrix(tx_a_re, tx_a_im, Tx_M , len_t_theta);
    float *rand_sig_re = (float *)malloc(1 * nd  * sizeof(float)+ AVX*sizeof(float));
    float *rand_sig_im = (float *)malloc(1 * nd * sizeof(float)+ AVX*sizeof(float));
    //---------------------------------------------------------------
    for (int i = 0; i < nd; ++i){
        cpp_t_sig(&rand_sig_re[i], &rand_sig_im[i]);
        //printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
    }

    
    // complex_matrix_multiplication(rand_sig_re, rand_sig_im, tx_a_re, tx_a_im, Tx_sig_re, Tx_sig_im, nd, 1, Tx_M_x * Tx_M_y);
    complex_matrix_multiplication(tx_a_re, tx_a_im, rand_sig_re, rand_sig_im, Tx_sig_re, Tx_sig_im, Tx_M_x * Tx_M_y, 1, nd);

    free(tx_a_re);
    free(tx_a_im);
    free(rand_sig_re);
    free(rand_sig_im);
}



void gen_angles(double t_theta_deg, double t_phi_deg, double Tx_beamwidth, int radius_samples, int angle_samples, float *theta_samples, float *phi_samples)
{
    // Generate θ/φ angle samples (in degrees) around the target direction
    double radius_step_deg = Tx_beamwidth / 2.0 / radius_samples; /* Angular radius step per layer (degrees) */
    double angle_step_rad  = 2.0 * PI / angle_samples;             /* Angular step per sector (radian) */

    int idx = 0;

    // r = 0: Center point (target direction)
    theta_samples[idx] = t_theta_deg;
    phi_samples[idx]   = t_phi_deg;
    // printf("r=%d, a=%2d → θ = %7.3f°,  φ = %7.3f°\n",
    //        0, 0, theta_samples[idx], phi_samples[idx]);
    idx++;

    // r = 1 to radius_samples: Concentric circular layers
    for (int r = 1; r <= radius_samples; ++r) {

        double delta_deg = radius_step_deg * r;   // Radius of the current layer

        for (int a = 0; a < angle_samples; ++a) {

            double ang_rad = a * angle_step_rad;  // Azimuthal angle (in radians)

            theta_samples[idx] = t_theta_deg + delta_deg * cos(ang_rad);
            phi_samples[idx]   = t_phi_deg   + delta_deg * sin(ang_rad);

            // printf("r=%d, a=%2d → θ = %7.3f°,  φ = %7.3f°\n",
            //        r, a, theta_samples[idx], phi_samples[idx]);

            idx++;
        }
    }
}
void generate_Directional_Channel_3D(float* H_re, float* H_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, float t_theta_deg, float t_phi_deg, float Tx_beamwidth, int Tx_beamwidth_samples){
    
    float radius_samples = 5;
    int angle_samples = 20;
    // float Tx_beamwidth_radius_step = (Tx_beamwidth/2)/radius_samples; //radius
    // float Tx_beamwidth_angle_step = (2*PI)/(float)angle_samples;
    Tx_beamwidth_samples = radius_samples*angle_samples+1;
    float Tx_M_x_delta = ((float)Tx_M_x-1)/2 ;
    float Tx_M_y_delta = ((float)Tx_M_y-1)/2 ;
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2 ;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2 ;
    float *theta_samples = (float *)malloc(Tx_beamwidth_samples * sizeof(float)+ AVX*sizeof(float));
    float *phi_samples = (float *)malloc(Tx_beamwidth_samples * sizeof(float)+ AVX*sizeof(float));

    // generating angle: 3D 
    gen_angles(t_theta_deg, t_phi_deg, Tx_beamwidth, radius_samples, angle_samples, theta_samples, phi_samples);

    
    float *tx_a_re = (float *)malloc((Tx_M_x*Tx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *tx_a_im = (float *)malloc((Tx_M_x*Tx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_re = (float *)malloc((Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_im = (float *)malloc((Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_re = (float *)malloc((Tx_M_x*Tx_M_y)*(Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_im = (float *)malloc((Tx_M_x*Tx_M_y)*(Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float)); 
    //printf("kc = %.4f\n", kc);
    for(int sample=0; sample<Tx_beamwidth_samples; sample++){
        // printf("theta_samples[%d] = %.2f, phi_samples[%d] = %.2f\n", sample, theta_samples[sample], sample, phi_samples[sample]);
        //printf("tx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Tx_M_x; ++i){   // theta
            for(int j = 0; j < Tx_M_y; ++j){    // phi
                //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp_3D(&tx_a_re[(i*Tx_M_y + j)], &tx_a_im[(i*Tx_M_y + j)], theta_samples[sample], phi_samples[sample], d, kc, i - Tx_M_x_delta, j - Tx_M_y_delta);
                //printf("%f + %fi\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
            }
        }
        // printf("rx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Rx_M_x; ++i){   // theta
            for(int j = 0; j < Rx_M_y; ++j){    // phi
                // printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                // printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp_3D(&rx_a_re[(i*Rx_M_y + j)], &rx_a_im[(i*Rx_M_y + j)], theta_samples[sample], phi_samples[sample], d, kc, i - Rx_M_x_delta, j - Rx_M_y_delta);
                // printf("%f + %fi\n", rx_a_re[i*Tx_M_y + j], rx_a_im[i*Tx_M_y + j]);
                // printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                // printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
            }
        }
        //matrix_transpose(tx_a_re, tx_a_im, (Tx_M_x*Tx_M_y), 1);
        complex_matrix_conjugate_transpose(tx_a_re, tx_a_im, (Tx_M_x*Tx_M_y), 1);
        complex_matrix_multiplication(rx_a_re, rx_a_im, tx_a_re, tx_a_im, H_temp_re, H_temp_im, (Rx_M_x*Rx_M_y), 1, (Tx_M_x*Tx_M_y));
        
        // printf("H_temp_re = \n");
        // print_complex_matrix(H_temp_re, H_temp_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
        // getchar();
        for(int j = 0; j<(Rx_M_x*Rx_M_y)*(Tx_M_x*Tx_M_y); j++){
            H_re[j] += H_temp_re[j];
            H_im[j] += H_temp_im[j];
        }
        //print_complex_matrix(H_re, H_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
        //getchar();
        
    }

    for(int j = 0; j<(Rx_M_x*Rx_M_y)*(Tx_M_x*Tx_M_y); j++){
        H_re[j] = H_re[j]/Tx_beamwidth_samples;
        H_im[j] = H_im[j]/Tx_beamwidth_samples;
    }
    // printf("H = \n");
    // print_complex_matrix(H_re, H_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
    free(theta_samples);
    free(phi_samples);
    free(tx_a_re);
    free(tx_a_im);
    free(rx_a_re);
    free(rx_a_im);
    free(H_temp_re);
    free(H_temp_im);
}


void generate_Directional_Channel_3D_1(float* H_re, float* H_im, int Tx_M_x, int Tx_M_y, int Rx_M_x, int Rx_M_y, float d, float kc, float t_theta_deg, float t_phi_deg, int Tx_beamwidth_samples){
    float Tx_M_x_delta = ((float)Tx_M_x-1)/2 ;
    float Tx_M_y_delta = ((float)Tx_M_y-1)/2 ;
    float Rx_M_x_delta = ((float)Rx_M_x-1)/2 ;
    float Rx_M_y_delta = ((float)Rx_M_y-1)/2 ;
    float *theta_samples = (float *)malloc(Tx_beamwidth_samples * sizeof(float)+ AVX*sizeof(float));
    float *phi_samples = (float *)malloc(Tx_beamwidth_samples * sizeof(float)+ AVX*sizeof(float));


    // generating angle: 3D 
    theta_samples[Tx_beamwidth_samples-1] = t_theta_deg;
    phi_samples[Tx_beamwidth_samples-1] = t_phi_deg;
    
    float *tx_a_re = (float *)malloc((Tx_M_x*Tx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *tx_a_im = (float *)malloc((Tx_M_x*Tx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_re = (float *)malloc((Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_im = (float *)malloc((Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_re = (float *)malloc((Tx_M_x*Tx_M_y)*(Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_im = (float *)malloc((Tx_M_x*Tx_M_y)*(Rx_M_x*Rx_M_y) * sizeof(float) + AVX*sizeof(float)); 
    //printf("kc = %.4f\n", kc);
    for(int sample=0; sample<Tx_beamwidth_samples; sample++){
        //printf("tx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Tx_M_x; ++i){   // theta
            for(int j = 0; j < Tx_M_y; ++j){    // phi
                //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp_3D(&tx_a_re[(i*Tx_M_y + j)], &tx_a_im[(i*Tx_M_y + j)], theta_samples[sample], phi_samples[sample], d, kc, i - Tx_M_x_delta, j - Tx_M_y_delta);
                //printf("%f + %fi\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
            }
        }
        //printf("rx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Rx_M_x; ++i){   // theta
            for(int j = 0; j < Rx_M_y; ++j){    // phi
                //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp_3D(&rx_a_re[(i*Rx_M_y + j)], &rx_a_im[(i*Rx_M_y + j)], theta_samples[sample], phi_samples[sample], d, kc, i - Rx_M_x_delta, j - Rx_M_y_delta);
                //printf("%f + %fi\n", rx_a_re[i*Tx_M_y + j], rx_a_im[i*Tx_M_y + j]);
                //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
            }
        }

        // matrix_transpose(tx_a_re, tx_a_im, (Tx_M_x*Tx_M_y), 1);
        complex_matrix_conjugate_transpose(tx_a_re, tx_a_im, (Tx_M_x*Tx_M_y), 1);
        complex_matrix_multiplication(rx_a_re, rx_a_im, tx_a_re, tx_a_im, H_temp_re, H_temp_im, (Rx_M_x*Rx_M_y), 1, (Tx_M_x*Tx_M_y));
        
        //getchar();
        for(int j = 0; j<(Rx_M_x*Rx_M_y)*(Tx_M_x*Tx_M_y); j++){
            H_re[j] += H_temp_re[j];
            H_im[j] += H_temp_im[j];
        }
        //print_complex_matrix(H_re, H_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
        //getchar();
        
    }

    for(int j = 0; j<(Rx_M_x*Rx_M_y)*(Tx_M_x*Tx_M_y); j++){
        H_re[j] = H_re[j]/Tx_beamwidth_samples;
        H_im[j] = H_im[j]/Tx_beamwidth_samples;
    }

    free(theta_samples);
    free(phi_samples);
    free(tx_a_re);
    free(tx_a_im);
    free(rx_a_re);
    free(rx_a_im);
    free(H_temp_re);
    free(H_temp_im);
}

void generate_incident_signal_3D(float* Tx_sig_re, float* Tx_sig_im, float* H_re, float* H_im, float* Coherent_sig_re, float* Coherent_sig_im, int Rx_M, int Tx_M, int nd){   
    // Coherent_sig(Rx_M , nd) = H*Tx_sig
    complex_matrix_multiplication(H_re, H_im, Tx_sig_re, Tx_sig_im, Coherent_sig_re, Coherent_sig_im, Rx_M, Tx_M, nd);
    
}



void generate_Rx_signal(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx){
    struct timeval time_initSig_start, time_initSig_end, time_initSig_diff;
    struct timeval time_awgn_start, time_awgn_end, time_awgn_diff;
    gettimeofday(&time_initSig_start, NULL);
    //-------------------------------------------------------------------
    // Parameter initialize
    // === Physical Parameters ===
    float kc = phys.kc;
    int SNR = phys.snr;

    // === Tx Parameters ===
    int Tx_M_x = tx.Tx_M_x;
    int Tx_M_y = tx.Tx_M_y;
    float Tx_beamwidth = tx.Tx_beamwidth;
    int Tx_beamwidth_samples = tx.Tx_beamwidth_samples;
    float *angle_theta = tx.angle_theta;
    float *angle_phi = tx.angle_phi;
    int number_angle = tx.number_angle;

    // === Rx Parameters ===
    int Rx_M_x = rx.Rx_M_x;
    int Rx_M_y = rx.Rx_M_y;       
    float d = rx.d;
    int nd = rx.nd;


    // ==================================================================
    // ========================= UE Tx model ============================
    // ==================================================================
    // Tx parameter initialize
    int len_t_angle = number_angle; // Tx Number of angle
    float *t_theta_deg = (float *)malloc(len_t_angle * sizeof(float)+ AVX*sizeof(float));
    for (int a = 0; a < len_t_angle; a++)
    {
        t_theta_deg[a] = angle_theta[a];
    }

    float *t_phi_deg = (float *)malloc(len_t_angle * sizeof(float));
    for (int a = 0; a < len_t_angle; a++)
    {
        t_phi_deg[a] = angle_phi[a];
    }
    //---------------------------------------------------------------
    //generate Rx incident signal
    float *Rx_incident_sig_re = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float), sizeof(float));
    float *Rx_incident_sig_im = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float) , sizeof(float));
    for(int signal = 0;signal<len_t_angle;signal++){
        generate_Rx_incident_sig_3D(Rx_incident_sig_re, Rx_incident_sig_im, Tx_M_x, Tx_M_y, Rx_M_x, Rx_M_y, d, kc, nd, t_theta_deg[signal], t_phi_deg[signal], Tx_beamwidth, Tx_beamwidth_samples);
    }
    //---------------------------------------------------------------
    // Normalize input signal to unit power
    float signal_power = 0.0f;
    for (int i = 0; i < Rx_M_x * Rx_M_y * nd; i++) {
        signal_power += Rx_incident_sig_re[i] * Rx_incident_sig_re[i] + Rx_incident_sig_im[i] * Rx_incident_sig_im[i];
    }
    signal_power /= (Rx_M_x * Rx_M_y * nd);
    float norm_factor = sqrt(signal_power);
    for (int i = 0; i < Rx_M_x * Rx_M_y * nd; i++) {
        Rx_incident_sig_re[i] /= norm_factor;
        Rx_incident_sig_im[i] /= norm_factor;
    }
    //export_Rx_sig_to_csv("Rx_sig.csv", Rx_incident_sig_re, Rx_incident_sig_im, Rx_M_x * Rx_M_y, nd);
    //---------------------------------------------------------------
    // AWGN
    gettimeofday(&time_awgn_start, NULL);
    cpp_awgn(Rx_incident_sig_re, Rx_incident_sig_im, Rx_sig_re, Rx_sig_im, SNR, Rx_M_x * Rx_M_y, nd); 
    gettimeofday(&time_awgn_end, NULL);
    //printf("---------------Rx_sig---------------\n");
    //print_complex_matrix(Rx_sig_re, Rx_sig_im, Rx_M_x * Rx_M_y, nd);

    gettimeofday(&time_initSig_end, NULL);

    float time_awgn;
    timersub(&time_awgn_end, &time_awgn_start, &time_awgn_diff);
    time_awgn = time_awgn_diff.tv_sec * 1000000 + time_awgn_diff.tv_usec;
    float time_initSig;
    timersub(&time_initSig_end, &time_initSig_start, &time_initSig_diff);
    time_initSig = time_initSig_diff.tv_sec * 1000000 + time_initSig_diff.tv_usec;
    printf("-----------------------------------------\n");
    printf("Total init signal time: %.3f(ms)\n", time_initSig / 1000);
    printf("-> AWGN time: \t\t%.3f(ms)\n", time_awgn / 1000);
    printf("-----------------------------------------\n");
    printf("\n\t--------Tx parameter--------\n");
    printf("Tx antenna count:\t%d(%dx%d)\n", Tx_M_x*Tx_M_y, Tx_M_x, Tx_M_y);
    printf("Tx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f, %.1f)", angle_theta[a], angle_phi[a]); 
    }
    printf("\t(degree)\n");
    
    printf("Tx beamwidth:\t\t%.1f\t\t(degree)\n", Tx_beamwidth);
    printf("Tx beamwidth samples:\t%d\n", Tx_beamwidth_samples);
    printf("\n\t--------Rx parameter--------\n");
    printf("Rx antenna count:\t%d(%dx%d)\n", Rx_M_x*Rx_M_y, Rx_M_x, Rx_M_y);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("Rx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f, %.1f)", angle_theta[a], angle_phi[a]); 
    }
    printf("\t(degree)\n");
    free(Rx_incident_sig_re);
    free(Rx_incident_sig_im);
    free(t_theta_deg);
    free(t_phi_deg);
}


void generate_Rx_signal_original(float *Rx_sig_re, float *Rx_sig_im, PhysicalParameters phys, RxParameters rx, TxParameters tx){
    struct timeval time_initSig_start, time_initSig_end, time_initSig_diff; // time initial
    gettimeofday(&time_initSig_start, NULL);
    //-------------------------------------------------------------------
    // Parameter initialize
    // === Physical Parameters ===
    float kc = phys.kc;
    int SNR = phys.snr;

    // === Tx Parameters ===
    int Tx_M_x = tx.Tx_M_x;
    int Tx_M_y = tx.Tx_M_y;
    float Tx_beamwidth = tx.Tx_beamwidth;
    int Tx_beamwidth_samples = tx.Tx_beamwidth_samples;
    float *angle_theta = tx.angle_theta;
    float *angle_phi = tx.angle_phi;
    int number_angle = tx.number_angle;

    // === Rx Parameters ===
    int Rx_M_x = rx.Rx_M_x;
    int Rx_M_y = rx.Rx_M_y;       
    float d = rx.d;
    int nd = rx.nd;


    // ==================================================================
    // ========================= UE Tx model ============================
    // ==================================================================
    // Tx parameter initialize
    int len_t_angle = number_angle; // Tx Number of angle
    float *t_theta_deg = (float *)malloc(len_t_angle * sizeof(float)+ AVX*sizeof(float));
    for (int a = 0; a < len_t_angle; a++)
    {
        t_theta_deg[a] = angle_theta[a];
    }

    float *t_phi_deg = (float *)malloc(len_t_angle * sizeof(float));
    for (int a = 0; a < len_t_angle; a++)
    {
        t_phi_deg[a] = angle_phi[a];
    }
    //---------------------------------------------------------------
    //generate Rx incident signal
    float *Rx_incident_sig_re = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float), sizeof(float));
    float *Rx_incident_sig_im = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float) , sizeof(float));
    for(int signal = 0;signal<len_t_angle;signal++){
        float *Rx_incident_sig_re_temp = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float), sizeof(float));
        float *Rx_incident_sig_im_temp = (float *)calloc(Rx_M_x * Rx_M_y * nd + AVX*sizeof(float) , sizeof(float));
        //---------------------------------------------------------------
        float *tx_a_re = (float *)malloc(Tx_M_x * Tx_M_y * sizeof(float)+ AVX*sizeof(float));
        float *tx_a_im = (float *)malloc(Tx_M_x * Tx_M_y * sizeof(float)+ AVX*sizeof(float));
        //---------------------------------------------------------------
        //printf("Tx_M_x = %d\n", Tx_M_x);
        float Tx_M_x_delta = ((float)Tx_M_x-1)/2;
        float Tx_M_y_delta = ((float)Tx_M_y-1)/2;
        //printf("Tx_M_x_delta = %f\n", Tx_M_x_delta);
        //Tx_M_x_delta = 0;
        //Tx_M_y_delta = 0;
        //printf("t_theta_deg = %.2f, t_phi_deg = %.2f\n", t_theta_deg, t_phi_deg);
        for (int i = 0; i < Tx_M_x; ++i){   
            for(int j = 0; j < Tx_M_y; ++j){    
                //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                //printf("i - Tx_M_x_delta = %.3f, j - Tx_M_y_delta = %.3f\n", i - Tx_M_x_delta, j - Tx_M_y_delta);
                cpp_exp_3D(&tx_a_re[i*Tx_M_y + j], &tx_a_im[i*Tx_M_y + j], t_theta_deg[signal], t_phi_deg[signal], d, kc, i - Tx_M_x_delta, j - Tx_M_y_delta);
                //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
            }
        }
        //print_complex_matrix(tx_a_re, tx_a_im, Tx_M , len_t_theta);
        float *rand_sig_re = (float *)malloc(nd  * sizeof(float)+ AVX*sizeof(float));
        float *rand_sig_im = (float *)malloc(nd * sizeof(float)+ AVX*sizeof(float));
        //---------------------------------------------------------------
        for (int i = 0; i < nd; ++i){
            cpp_t_sig(&rand_sig_re[i], &rand_sig_im[i]);
            //printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
        }

        
        complex_matrix_multiplication(tx_a_re, tx_a_im, rand_sig_re, rand_sig_im, Rx_incident_sig_re_temp, Rx_incident_sig_im_temp, Rx_M_x * Rx_M_y, 1, nd);

        for(int j=0; j<Rx_M_x * Rx_M_y * nd; j++){
            Rx_incident_sig_re[j] += Rx_incident_sig_re_temp[j];
            Rx_incident_sig_im[j] += Rx_incident_sig_im_temp[j];
            //printf("total_incident_sig_re[%d] = %.2f\n", j, total_incident_sig_re[j]);
        }
        free(tx_a_re);
        free(tx_a_im);
        free(rand_sig_re);
        free(rand_sig_im);
        free(Rx_incident_sig_re_temp);
        free(Rx_incident_sig_im_temp);
    }
    //---------------------------------------------------------------
    // Normalize input signal to unit power
    float signal_power = 0.0f;
    for (int i = 0; i < Rx_M_x * Rx_M_y * nd; i++) {
        signal_power += Rx_incident_sig_re[i] * Rx_incident_sig_re[i] + Rx_incident_sig_im[i] * Rx_incident_sig_im[i];
    }
    signal_power /= (Rx_M_x * Rx_M_y * nd);
    float norm_factor = sqrt(signal_power);
    for (int i = 0; i < Rx_M_x * Rx_M_y * nd; i++) {
        Rx_incident_sig_re[i] /= norm_factor;
        Rx_incident_sig_im[i] /= norm_factor;
    }
    //export_Rx_sig_to_csv("Rx_sig.csv", Rx_incident_sig_re, Rx_incident_sig_im, Rx_M_x * Rx_M_y, nd);
    //---------------------------------------------------------------
    // AWGN
    cpp_awgn(Rx_incident_sig_re, Rx_incident_sig_im, Rx_sig_re, Rx_sig_im, SNR, Rx_M_x * Rx_M_y, nd); 
    //printf("---------------Rx_sig---------------\n");
    //print_complex_matrix(Rx_sig_re, Rx_sig_im, Rx_M_x * Rx_M_y, nd);

    gettimeofday(&time_initSig_end, NULL);

    float time_initSig;
    timersub(&time_initSig_end, &time_initSig_start, &time_initSig_diff);
    time_initSig = time_initSig_diff.tv_sec * 1000000 + time_initSig_diff.tv_usec;
    printf("-----------------------------------------\n");
    printf("Total Init signal time: \t%.3f(ms)\n", time_initSig / 1000);
    printf("-----------------------------------------\n");
    printf("\n\t--------Tx parameter--------\n");
    printf("Tx antenna count:\t%d(%dx%d)\n", Tx_M_x*Tx_M_y, Tx_M_x, Tx_M_y);
    printf("Tx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f, %.1f) ", angle_theta[a], angle_phi[a]); 
    }
    printf("\n");
    printf("Tx beamwidth:\t\t%.1f (degree)\n", Tx_beamwidth);
    printf("Tx beamwidth samples:\t%d\n", Tx_beamwidth_samples);
    printf("\n\t--------Rx parameter--------\n");
    printf("Rx antenna count:\t%d(%dx%d)\n", Rx_M_x*Rx_M_y, Rx_M_x, Rx_M_y);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("Rx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f, %.1f) ", angle_theta[a], angle_phi[a]); 
    }
    printf("\n");
    free(Rx_incident_sig_re);
    free(Rx_incident_sig_im);
    free(t_theta_deg);
    free(t_phi_deg);
}