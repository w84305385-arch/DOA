// C++
#define PI acos(-1)
#include "math_func_3D.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <ccomplex>
#include <complex>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "q_format_config.h"
using namespace std;

const std::complex<float> I_1(0, 1);
const std::complex<double> I_d(0, 1);

void cpp_abs(float *Real, float *Imaginary, float *result)
{
    complex<double> complex_variable = {*Real, *Imaginary};
    *result = abs(complex_variable);
    // printf("abs|%.2f,%.2f| = %.4f\n", *Real, *Imaginary, *result);
}

void cpp_sqrt(float *Real, float *Imaginary)
{
    complex<double> complex_variable = {*Real, *Imaginary};
    // cout << "Square root of ";
    // cout << complex_variable;
    // cout << " = ";
    // cout << sqrt(complex_variable) << endl;

    *Real = sqrt(complex_variable).real();
    *Imaginary = sqrt(complex_variable).imag();

    // printf("Real = %.4f, Imaginary = %.4f\n", *Real, *Imaginary);
}
void cpp_sqrt_i(int16_t *Real, int16_t *Imaginary)
{
    
    std::complex<float> complex_variable(static_cast<float>(*Real), static_cast<float>(*Imaginary));
    std::complex<float> result = std::sqrt(complex_variable);

    *Real = static_cast<int16_t>(round(result.real()));
    *Imaginary = static_cast<int16_t>(round(result.imag()));

    // printf("Real = %.4f, Imaginary = %.4f\n", *Real, *Imaginary);
}

void cpp_sqrt_i32(int *Real, int *Imaginary)
{
    // 將 Real 和 Imaginary 轉換為 float，並建立 complex 變數
    std::complex<float> complex_variable(static_cast<float>(*Real), static_cast<float>(*Imaginary));
    
    // 計算 complex 變數的平方根
    std::complex<float> result = std::sqrt(complex_variable);

    // 將結果的實部和虛部四捨五入並轉換回 int 型別
    *Real = static_cast<int>(round(result.real()));
    *Imaginary = static_cast<int>(round(result.imag()));

    // printf("Real = %.4f, Imaginary = %.4f\n", *Real, *Imaginary);
}

void cpp_division(float *Re_a, float *Im_b, float *Re_c, float *Im_d)
{
    complex<double> complex_variable_ab = {*Re_a, *Im_b};
    complex<double> complex_variable_cd = {*Re_c, *Im_d};
    // cout << "Division of ";
    // cout << complex_variable_ab;
    // cout << " /= ";
    // cout << complex_variable_cd;
    // cout << " = ";
    // cout << (complex_variable_ab / complex_variable_cd) << endl;
    complex_variable_ab /= complex_variable_cd;
    *Re_a = complex_variable_ab.real();
    *Im_b = complex_variable_ab.imag();
    // printf("Real = %.4f, Imaginary = %.4f\n", *Re_a, *Im_b);
}
void cpp_division_i(int16_t *Re_a, int16_t *Im_b, int16_t *Re_c, int16_t *Im_d)
{   
    //為了保持精度強制轉型 轉成float
    std::complex<float> complex_variable_ab(static_cast<float>(*Re_a), static_cast<float>(*Im_b));
    std::complex<float> complex_variable_cd(static_cast<float>(*Re_c), static_cast<float>(*Im_d));

    std::complex<float> result = complex_variable_ab / complex_variable_cd;
    //四捨五入 並轉型回去
    *Re_a = static_cast<int16_t>(round(result.real()));
    *Im_b = static_cast<int16_t>(round(result.imag()));
    //*Re_a = static_cast<int>(std::ceil(result.real()));
    //*Im_b = static_cast<int>(std::ceil(result.imag()));
    // printf("Real = %.4f, Imaginary = %.4f\n", *Re_a, *Im_b);
}

void cpp_division_i32(int *Re_a, int *Im_b, int *Re_c, int *Im_d)
{   
    //為了保持精度強制轉型 轉成float
    std::complex<float> complex_variable_ab(static_cast<float>(*Re_a), static_cast<float>(*Im_b));
    std::complex<float> complex_variable_cd(static_cast<float>(*Re_c), static_cast<float>(*Im_d));

    std::complex<float> result = complex_variable_ab / complex_variable_cd;
    //四捨五入 並轉型回去
    *Re_a = static_cast<int>(round(result.real()));
    *Im_b = static_cast<int>(round(result.imag()));
    //*Re_a = static_cast<int>(std::ceil(result.real()));
    //*Im_b = static_cast<int>(std::ceil(result.imag()));
    // printf("Real = %.4f, Imaginary = %.4f\n", *Re_a, *Im_b);
}

void cpp_division2(float Re_a, float Im_b, float *Re_c, float *Im_d, float *result_real, float *result_imag)
{
    complex<double> complex_variable_ab = {Re_a, Im_b};
    complex<double> complex_variable_cd = {*Re_c, *Im_d};
    complex<double> complex_variable_ef; // result store here
    // cout << "Division2 of ";
    // cout << complex_variable_ab;
    // cout << " / ";
    // cout << complex_variable_cd;
    // cout << " = ";
    // cout << (complex_variable_ab / complex_variable_cd) << endl;
    complex_variable_ef = complex_variable_ab / complex_variable_cd;
    *result_real = complex_variable_ef.real();
    *result_imag = complex_variable_ef.imag();
    // printf("Real = %.4f, Imaginary = %.4f\n", *result_real, *result_imag);
}
void cpp_division3(float *Re_a, float *Im_b, float *Re_c, float *Im_d)
{
    complex<double> complex_variable_ab = {*Re_a, *Im_b};
    complex<double> complex_variable_cd = {*Re_c, *Im_d};
    // cout << "Division3 of ";
    // cout << complex_variable_ab;
    // cout << " / ";
    // cout << complex_variable_cd;
    // cout << " = ";
    // cout << (complex_variable_ab / complex_variable_cd) << endl;

    *Re_c = (complex_variable_ab / complex_variable_cd).real();
    *Im_d = (complex_variable_ab / complex_variable_cd).imag();
    // printf("Real = %.4f, Imaginary = %.4f\n", *Re_c, *Im_d);
}


inline std::complex<double> randn(void)
{
    // Only constructed the first time it is called; each thread has a copy
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::normal_distribution<double> dist{0.0, 1.0};
    // std::cout <<  dist(rng)<< std::endl;
    return {dist(rng), dist(rng)};
    
}

void cpp_awgn(float *input_re, float *input_im, float *output_re, float *output_im, int snr, int row, int col)
{
    //printf("cpp_awgn(%d,%d)\n", row, col);
    std::complex<double> *input_signal = (complex<double> *)malloc(row * col * sizeof(complex<double>));  // Input
    std::complex<double> *output_signal = (complex<double> *)malloc(row * col * sizeof(complex<double>)); // Output

    std::complex<double> SignalPower;
    std::complex<double> NoisePower;
    std::complex<double> NoiseSigma;
    std::complex<double> Noise;

    std::complex<double> total_SignalPower = (0,0);
    std::complex<double> total_NoisePower = (0,0);
    double snr_linear = std::pow(10.0, snr / 10.0);
    // std::cout << "--- snr_linear ---" << snr_linear << std::endl;
    for (int i = 0; i < row * col; i++)
    {
        input_signal[i] = {input_re[i], input_im[i]};
        output_signal[i] = {output_re[i], output_im[i]};
        // //---------------------------------------------------------------
        SignalPower = pow(abs(input_signal[i]), 2);
        total_SignalPower += SignalPower;
        // std::cout << "--- SignalPower ---" << SignalPower << std::endl;
        NoisePower = SignalPower / snr_linear;
        // std::cout << "--- NoisePower ---" << NoisePower << std::endl;
        total_NoisePower += NoisePower;
        NoiseSigma = sqrt(NoisePower / complex<double>(2));
        Noise = NoiseSigma * (randn() + randn() * 1i);
        // std::cout << "--- Noise ---" << Noise << std::endl;
        output_signal[i] = input_signal[i] + Noise;
        //---------------------------------------------------------------
        output_re[i] = output_signal[i].real();
        output_im[i] = output_signal[i].imag();
        //---------------------------------------------------------------
        // std::cout << "---awgn output_signal---" << output_signal[i] << std::endl;
    }
    // std::cout << "--- average_SignalPower---" << total_SignalPower/complex<double>(row * col) << std::endl;
    // std::cout << "--- average_NoisePower ---" << total_NoisePower/complex<double>(row * col)<< std::endl;
    free(input_signal);
    free(output_signal);
}

void cpp_exp_3D(float *A_theta_re, float *A_theta_im, float t_theta, float t_phi, float d, float kc, float M_x, float M_y)
{
    float theta_rad = (t_theta) * PI / 180.0f; // Convert degrees to radians
    float phi_rad = (t_phi) * PI / 180.0f; // Convert degrees to radians
    float phase = kc * d * (M_x*cos(phi_rad)*cos(theta_rad) + M_y * sin(phi_rad) * cos(theta_rad));
    //std::cout << "---phase--- " << phase << std::endl;
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}


void cpp_exp2_3D(float *A_theta_re, float *A_theta_im, float *dr, float *drp, float d, float kc, int theta, int phi, float M_x, float M_y)
{
    // float *dr is rad
    // theta is elevation
    // phi is azimuth
    float phase = kc * d * (M_x*cos(drp[phi])*cos(dr[theta]) + M_y*sin(drp[phi])*cos(dr[theta]));
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}



void cpp_t_sig(float *t_sig_re, float *t_sig_im)
{
    complex<double> t_sig;
    t_sig = (randn() + randn() * I_d) / complex<double>(sqrt(2));
    // std::cout << "---t_sig---" << t_sig << std::endl;

    *t_sig_re = t_sig.real();
    *t_sig_im = t_sig.imag();
}


float cpp_20log_abs(float *S_MUSIC_re, float *S_MUSIC_im)
{
    complex<double> S_MUSIC = {*S_MUSIC_re, *S_MUSIC_im};
    // std::cout << "20log" << S_MUSIC << " = " << 20 * log10(abs(S_MUSIC)) << std::endl;
    return 20 * log10(abs(S_MUSIC));

    //return abs(S_MUSIC);
}



