// C++
#define PI acos(-1)
#include "math_func.h"
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
using namespace std;
// g++ -c  math_func.cpp -Wall -Wextra -std=c++14
// ar -rcs math_func.a math_func.o
// g++ -o test test.c math_func.a

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

std::complex<double> randn(void)
{
    std::random_device randomness_device{};
    std::mt19937 pseudorandom_generator{randomness_device()};
    auto mean = 0.0;
    auto std_dev = 1.0;
    std::normal_distribution<> distribution{mean, std_dev};
    auto sample = distribution(pseudorandom_generator);
    return (std::complex<double>)(sample);
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
    for (int i = 0; i < row * col; i++)
    {
        input_signal[i] = {input_re[i], input_im[i]};
        output_signal[i] = {output_re[i], output_im[i]};
        // //---------------------------------------------------------------
        SignalPower = pow(abs(input_signal[i]), 2);
        total_SignalPower += SignalPower;
        //std::cout << "--- SignalPower ---" << SignalPower << std::endl;
        double snr_linear = std::pow(10.0, snr / 10.0);
        //std::cout << "--- snr_linear ---" << snr_linear << std::endl;
        NoisePower = SignalPower / snr_linear;
        //std::cout << "--- NoisePower ---" << NoisePower << std::endl;
        total_NoisePower += NoisePower;
        NoiseSigma = sqrt(NoisePower / complex<double>(2));
        Noise = NoiseSigma * (randn() + randn() * 1i);
        //std::cout << "--- Noise ---" << Noise << std::endl;
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

void cpp_exp(float *A_theta_re, float *A_theta_im, float *t_theta, float d, float kc, float n, int j)
{
    // float d is deg
    float theta_rad = (*t_theta) * PI / 180.0f; // Convert degrees to radians
    float phase = kc * (n) * d * sin(theta_rad);
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}
void cpp_exp_tx(float *A_theta_re, float *A_theta_im, float *t_theta, float d, float kc, float n, int j)
{
    // float d is deg
    float theta_rad = (*t_theta) * PI / 180.0f; // Convert degrees to radians
    float phase = kc * (n) * d * sin(theta_rad);
    std::complex<float> A_theta = std::exp(I_1 * phase);// tx is +j

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}

void cpp_exp_1(float *A_theta_re, float *A_theta_im, float t_theta, float d, float kc, float n)
{
    // float d is deg
    float theta_rad = t_theta * PI / 180.0f; // Convert degrees to radians
    float phase = kc * (n) * d * sin(theta_rad);
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}
void cpp_exp_1_tx(float *A_theta_re, float *A_theta_im, float t_theta, float d, float kc, float n)
{
    // float d is deg
    float theta_rad = t_theta * PI / 180.0f; // Convert degrees to radians
    float phase = kc * (n) * d * sin(theta_rad);
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}

void cpp_exp2(float *A_theta_re, float *A_theta_im, float *dr, float d, float kc, int i, float n)
{
    // float *dr is rad
    float phase = kc * n * d * sin(dr[i]);
    std::complex<float> A_theta = std::exp(I_1 * phase);

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}
/* ------------  fixed-point utility ------------ */
#ifndef Q_SHIFT               /* gcc … -DQ_SHIFT=14 可覆寫 */
#define Q_SHIFT 14            /* 預設 Q1.14 ── 1 整數位 + 14 小數位 */
#endif

#define Q_SCALE  (1LL << Q_SHIFT)

/* float → Qx.y  (四捨五入 + 飽和保護) */
#define TO_Q(x)  ({                              \
    long long t = llroundf((x) * (float)Q_SCALE); \
    if (t >  0x7FFFFFFFLL) t =  0x7FFFFFFFLL;    \
    if (t < -0x80000000LL) t = -0x80000000LL;    \
    (int32_t)t;                                  \
})

/* Qx.y → float */
#define FROM_Q(q)   ((float)(q) / (float)Q_SCALE)
/* ----------- steering-vector element (Q1.14) ----------- */
void cpp_exp2_int32q14(int32_t *A_theta_re,
                  int32_t *A_theta_im,
                  const float *dr,   /* θ(rad) 陣列                */
                  float d,           /* 陣元間距                    */
                  float kc,          /* 2π/λ                        */
                  int   idx,         /* dr[] index (= elevation)   */
                  float n)           /* n = j - (M-1)/2            */
{
    /* 仍然先在 float 域計算相位，精度足夠又省事 */
    float phase = kc * n * d * sinf(dr[idx]);

    std::complex<float> A_theta = std::exp(I_1 * phase);
    *A_theta_re = TO_Q(A_theta.real());  /* 寫進 Q1.14 */
    *A_theta_im = TO_Q(A_theta.imag());
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
}


float cpp_rand(float max_value)
{
    std::random_device randomness_device{};
    std::mt19937 pseudorandom_generator{randomness_device()};

    // 使用均勻分佈，範圍從 -max_value 到 max_value
    std::uniform_real_distribution<float> distribution(-max_value, max_value);

    // 產生一個服從均勻分佈的隨機 float 數
    return distribution(pseudorandom_generator);
}


