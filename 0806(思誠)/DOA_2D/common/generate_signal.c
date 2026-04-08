//--------------------
#define PI acos(-1)
#define DEG2RAD (PI/180.0)
#define AVX 16            
//--------------------
#include <immintrin.h>
#include "generate_signal.h"
#include "math_func.h"
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

void generate_Rx_incident_sig(float* total_incident_sig_re,float*  total_incident_sig_im, int Tx_M, int Rx_M, float d, float kc, int nd, float t_theta_deg,float  Tx_beamwidth_theta,float  Tx_beamwidth_samples){
    //---------------------------------------------------------------
    // 生成Tx signal ;會有很多個訊號
    float *Tx_sig_re = (float *)malloc(nd * Tx_M * sizeof(float));
    float *Tx_sig_im = (float *)malloc(nd * Tx_M * sizeof(float));
    generate_Tx_signal(Tx_sig_re, Tx_sig_im, Tx_M, d, kc, nd, t_theta_deg);//t_theta_deg
    //printf("---------------Tx_sig---------------\n");
    //print_complex_matrix(Tx_sig_re, Tx_sig_im, nd, Tx_M);

    //對每個發射角度計算通道響應並累加接收訊號
    float *H_re = (float *)calloc(Rx_M * Tx_M , sizeof(float));
    float *H_im = (float *)calloc(Rx_M * Tx_M , sizeof(float));
    generate_Directional_Channel(H_re, H_im, Tx_M, Rx_M, d, kc, t_theta_deg, Tx_beamwidth_theta, Tx_beamwidth_samples);
    //printf("---------------H---------------\n");
    //print_complex_matrix(H_re, H_im, Rx_M, Tx_M);


    // ==================================================================
    // ========================= gNB Rx model ===========================
    // ==================================================================
    //---------------------------------------------------------------
    float *incident_sig_re = (float *)malloc(nd * Rx_M * sizeof(float));
    float *incident_sig_im = (float *)malloc(nd * Rx_M * sizeof(float));

    generate_incident_signal(Tx_sig_re, Tx_sig_im, H_re, H_im, incident_sig_re, incident_sig_im, Rx_M, Tx_M, nd);

    for(int j=0; j<Rx_M*nd; j++){
        total_incident_sig_re[j] += incident_sig_re[j];
        total_incident_sig_im[j] += incident_sig_im[j];
    }
    
    free(Tx_sig_re);
    free(Tx_sig_im);
    free(H_re);
    free(H_im);
    free(incident_sig_re);
    free(incident_sig_im);
}

void generate_Tx_signal(float* Tx_sig_re, float* Tx_sig_im, int Tx_M, float d, float kc, int nd, float t_theta_deg){
    //---------------------------------------------------------------
    float *tx_a_re = (float *)malloc(Tx_M * 1 * sizeof(float));
    float *tx_a_im = (float *)malloc(Tx_M * 1 * sizeof(float));
    //---------------------------------------------------------------
    float Tx_delta = ((float)Tx_M-1)/2;
    for (int i = 0; i < Tx_M; ++i)
    {
            cpp_exp_1(&tx_a_re[i], &tx_a_im[i], t_theta_deg, d, kc, i-Tx_delta); //t_theta_deg
            //printf("\t(%f,%f)\n", tx_a_re[i * len_t_theta + j], tx_a_im[i * len_t_theta + j]);
            //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
    }
    //print_complex_matrix(tx_a_re, tx_a_im, Tx_M , len_t_theta);
    float *rand_sig_re = (float *)malloc(1* nd * sizeof(float));
    float *rand_sig_im = (float *)malloc(1* nd * sizeof(float));
    //---------------------------------------------------------------

    for (int i = 0; i < nd; ++i){
        cpp_t_sig(&rand_sig_re[i], &rand_sig_im[i]);
        //printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
    }

    complex_matrix_multiplication(tx_a_re, tx_a_im, rand_sig_re, rand_sig_im, Tx_sig_re, Tx_sig_im, Tx_M, 1, nd);

    free(tx_a_re);
    free(tx_a_im);
    free(rand_sig_re);
    free(rand_sig_im);
}

void generate_Directional_Channel(float* H_re, float* H_im, int Tx_M, int Rx_M, float d, float kc, float t_theta_deg, float Tx_beamwidth_theta, int Tx_beamwidth_samples){
    float Tx_beamwidth_step = (float)Tx_beamwidth_theta/(float)Tx_beamwidth_samples;
    Tx_beamwidth_samples++;
    float Tx_delta = ((float)Tx_M-1)/2;
    //printf("Tx_delta = %f\n", Tx_delta);
    float Rx_delta = ((float)Rx_M-1)/2;
    float *theta_samples = (float *)malloc(Tx_beamwidth_samples * sizeof(float));
    for(int i = 0; i<Tx_beamwidth_samples; i++){
        theta_samples[i] = (t_theta_deg - Tx_beamwidth_theta/2) + Tx_beamwidth_step*i;
        // printf("theta_samples = %.4f\n",theta_samples[i]);
    }

    float *tx_a_re = (float *)malloc((Tx_M) * sizeof(float) + AVX*sizeof(float));
    float *tx_a_im = (float *)malloc((Tx_M) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_re = (float *)malloc((Rx_M) * sizeof(float) + AVX*sizeof(float));
    float *rx_a_im = (float *)malloc((Rx_M) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_re = (float *)malloc((Tx_M)*(Rx_M) * sizeof(float) + AVX*sizeof(float));
    float *H_temp_im = (float *)malloc((Tx_M)*(Rx_M) * sizeof(float) + AVX*sizeof(float)); 
    
    for(int sample=0; sample<Tx_beamwidth_samples; sample++){
        // printf("theta_samples[%d] = %.2f, phi_samples[%d] = %.2f\n", sample, theta_samples[sample], sample, phi_samples[sample]);
        //printf("tx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Tx_M; ++i){   // theta
                //printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                //printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp(&tx_a_re[i], &tx_a_im[i],&theta_samples[sample] , d, kc, i-Tx_delta, 0);
                //printf("%f + %fi\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
        }
        // printf("rx_a(%.2f, %.2f) = \n", theta_samples[sample], phi_samples[sample]);
        for (int i = 0; i < Rx_M; ++i){   // theta
                // printf("(i*Tx_M_y + j) = %d\n", (i*Tx_M_y + j));
                // printf("i - Tx_M_x_delta = %f, i = %d, Tx_M_x_delta = %f\n", i - Tx_M_x_delta, i, Tx_M_x_delta);
                cpp_exp(&rx_a_re[i], &rx_a_im[i],&theta_samples[sample] , d, kc, i-Rx_delta, 0);
                // printf("%f + %fi\n", rx_a_re[i*Tx_M_y + j], rx_a_im[i*Tx_M_y + j]);
                // printf("\t(%f,%f)\n", tx_a_re[i*Tx_M_y + j], tx_a_im[i*Tx_M_y + j]);
                // printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
        }
        
        complex_matrix_conjugate_transpose(tx_a_re, tx_a_im, Tx_M, 1);
        complex_matrix_multiplication(rx_a_re, rx_a_im, tx_a_re, tx_a_im, H_temp_re, H_temp_im, Rx_M, 1, Tx_M);
        
        // printf("H_temp_re = \n");
        // print_complex_matrix(H_temp_re, H_temp_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
        // getchar();
        for(int j = 0; j<Rx_M*Tx_M; j++){
            H_re[j] += H_temp_re[j];
            H_im[j] += H_temp_im[j];
        }
        //print_complex_matrix(H_re, H_im, (Rx_M_x*Rx_M_y), (Tx_M_x*Tx_M_y));
        //getchar();
        
    }

    for(int j = 0; j<Rx_M*Tx_M; j++){
        H_re[j] = H_re[j]/Tx_beamwidth_samples;
        H_im[j] = H_im[j]/Tx_beamwidth_samples;
    }
    free(theta_samples);

    free(tx_a_re);
    free(tx_a_im);
    free(rx_a_re);
    free(rx_a_im);
    free(H_temp_re);
    free(H_temp_im);
}

void generate_incident_signal(float* Tx_sig_re, float* Tx_sig_im, float* H_re, float* H_im, float* Coherent_sig_re, float* Coherent_sig_im, int Rx_M, int Tx_M, int nd){
    // Coherent_sig(Rx_M , nd) = Tx_sig*rand_sig 還沒加入AWGN的訊號
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
    float Tx_beamwidth = tx.Tx_beamwidth;
    int Tx_beamwidth_samples = tx.Tx_beamwidth_samples;
    float *angle_theta = tx.angle_theta;
    int number_angle = tx.number_angle;

    // === Rx Parameters ===
    int Rx_M_x = rx.Rx_M_x;
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

    //---------------------------------------------------------------
    //generate Rx incident signal
    float *Rx_incident_sig_re = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float), sizeof(float));
    float *Rx_incident_sig_im = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float) , sizeof(float));
    for(int signal = 0;signal<len_t_angle;signal++){
        generate_Rx_incident_sig(Rx_incident_sig_re, Rx_incident_sig_im, Tx_M_x, Rx_M_x, d, kc, nd, t_theta_deg[signal], Tx_beamwidth, Tx_beamwidth_samples);
    }
    //---------------------------------------------------------------
    // Normalize input signal to unit power
    float signal_power = 0.0f;
    for (int i = 0; i < Rx_M_x * nd; i++) {
        signal_power += Rx_incident_sig_re[i] * Rx_incident_sig_re[i] + Rx_incident_sig_im[i] * Rx_incident_sig_im[i];
    }
    signal_power /= (Rx_M_x * nd);
    float norm_factor = sqrt(signal_power);
    for (int i = 0; i < Rx_M_x * nd; i++) {
        Rx_incident_sig_re[i] /= norm_factor;
        Rx_incident_sig_im[i] /= norm_factor;
    }
    //export_Rx_sig_to_csv("Rx_sig.csv", Rx_incident_sig_re, Rx_incident_sig_im, Rx_M_x, nd);
    //---------------------------------------------------------------
    // AWGN
    gettimeofday(&time_awgn_start, NULL);
    cpp_awgn(Rx_incident_sig_re, Rx_incident_sig_im, Rx_sig_re, Rx_sig_im, SNR, Rx_M_x, nd); 
    gettimeofday(&time_awgn_end, NULL);
    //printf("---------------Rx_sig---------------\n");
    //print_complex_matrix(Rx_sig_re, Rx_sig_im, Rx_M_x, nd);

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
    printf("Tx antenna count:\t%d\n", Tx_M_x);
    printf("Tx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f)", angle_theta[a]); 
    }
    printf("\t(degree)\n");
    
    printf("Tx beamwidth:\t\t%.1f\t\t(degree)\n", Tx_beamwidth);
    printf("Tx beamwidth samples:\t%d\n", Tx_beamwidth_samples);
    printf("\n\t--------Rx parameter--------\n");
    printf("Rx antenna count:\t%d\n", Rx_M_x);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("Rx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f)", angle_theta[a]); 
    }
    printf("\t(degree)\n");
    free(Rx_incident_sig_re);
    free(Rx_incident_sig_im);
    free(t_theta_deg);
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
    float Tx_beamwidth = tx.Tx_beamwidth;
    int Tx_beamwidth_samples = tx.Tx_beamwidth_samples;
    float *angle_theta = tx.angle_theta;
    int number_angle = tx.number_angle;

    // === Rx Parameters ===
    int Rx_M_x = rx.Rx_M_x;    
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
    //---------------------------------------------------------------
    //generate Rx incident signal
    float *Rx_incident_sig_re = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float), sizeof(float));
    float *Rx_incident_sig_im = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float) , sizeof(float));
    for(int signal = 0;signal<len_t_angle;signal++){
        float *Rx_incident_sig_re_temp = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float), sizeof(float));
        float *Rx_incident_sig_im_temp = (float *)calloc(Rx_M_x * nd + AVX*sizeof(float) , sizeof(float));
        //---------------------------------------------------------------
        float *tx_a_re = (float *)malloc(Tx_M_x * 1 * sizeof(float));
        float *tx_a_im = (float *)malloc(Tx_M_x * 1 * sizeof(float));
        //---------------------------------------------------------------
        float Tx_delta = ((float)Tx_M_x-1)/2;
        for (int i = 0; i < Tx_M_x; ++i)
        {
                cpp_exp_1(&tx_a_re[i], &tx_a_im[i], t_theta_deg[signal], d, kc, i-Tx_delta); //t_theta_deg
                //printf("\t(%f,%f)\n", tx_a_re[i * len_t_theta + j], tx_a_im[i * len_t_theta + j]);
                //printf("\tt_theta_deg[%d] = %.2f\n",j, t_theta_deg[j]);
        }
        //print_complex_matrix(tx_a_re, tx_a_im, Tx_M , len_t_theta);
        float *rand_sig_re = (float *)malloc(nd * sizeof(float));
        float *rand_sig_im = (float *)malloc(nd * sizeof(float));
        //---------------------------------------------------------------

        for (int i = 0; i < nd; ++i){
            cpp_t_sig(&rand_sig_re[i], &rand_sig_im[i]);
            //printf("\t(%f,%f)\n", t_sig_re[i * nd + j], t_sig_im[i * nd + j]);
        }
        complex_matrix_multiplication(rand_sig_re, rand_sig_im, tx_a_re, tx_a_im, Rx_incident_sig_re, Rx_incident_sig_im, nd, 1, Tx_M_x);
        for(int j=0; j<Rx_M_x * nd; j++){
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
    for (int i = 0; i < Rx_M_x * nd; i++) {
        signal_power += Rx_incident_sig_re[i] * Rx_incident_sig_re[i] + Rx_incident_sig_im[i] * Rx_incident_sig_im[i];
    }
    signal_power /= (Rx_M_x * nd);
    float norm_factor = sqrt(signal_power);
    for (int i = 0; i < Rx_M_x * nd; i++) {
        Rx_incident_sig_re[i] /= norm_factor;
        Rx_incident_sig_im[i] /= norm_factor;
    }
    //export_Rx_sig_to_csv("Rx_sig.csv", Rx_incident_sig_re, Rx_incident_sig_im, Rx_M_x, nd);
    //---------------------------------------------------------------
    // AWGN
    cpp_awgn(Rx_incident_sig_re, Rx_incident_sig_im, Rx_sig_re, Rx_sig_im, SNR, Rx_M_x, nd); 
    //printf("---------------Rx_sig---------------\n");
    //print_complex_matrix(Rx_sig_re, Rx_sig_im, Rx_M_x, nd);

    gettimeofday(&time_initSig_end, NULL);

    float time_initSig;
    timersub(&time_initSig_end, &time_initSig_start, &time_initSig_diff);
    time_initSig = time_initSig_diff.tv_sec * 1000000 + time_initSig_diff.tv_usec;
    printf("-----------------------------------------\n");
    printf("Total Init signal time: \t%.3f(ms)\n", time_initSig / 1000);
    printf("-----------------------------------------\n");
    printf("\n\t--------Tx parameter--------\n");
    printf("Tx antenna count:\t%d\n", Tx_M_x);
    printf("Tx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f) ", angle_theta[a]); 
    }
    printf("\n");
    printf("Tx beamwidth:\t\t%.1f (degree)\n", Tx_beamwidth);
    printf("Tx beamwidth samples:\t%d\n", Tx_beamwidth_samples);
    printf("\n\t--------Rx parameter--------\n");
    printf("Rx antenna count:\t%d\n", Rx_M_x);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("Rx angle: \t\t");
    for (int a = 0; a < len_t_angle; a++)
    {
        printf("(%.1f) ", angle_theta[a]); 
    }
    printf("\n");
    free(Rx_incident_sig_re);
    free(Rx_incident_sig_im);
    free(t_theta_deg);
}