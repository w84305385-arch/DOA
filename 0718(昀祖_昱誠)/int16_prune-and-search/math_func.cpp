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
    printf("cpp_awgn(%d,%d)\n", row, col);
    std::complex<double> *input_signal = (complex<double> *)malloc(row * col * sizeof(complex<double>));  // Input
    std::complex<double> *output_signal = (complex<double> *)malloc(row * col * sizeof(complex<double>)); // Output

    std::complex<double> Esym;
    std::complex<double> No;
    std::complex<double> noiseSigma;
    std::complex<double> n;
    std::complex<double> e_AWGN;
    std::default_random_engine generator;
    // std::normal_distribution<double> distribution(0.0, noiseSigma);
    for (int i = 0; i < row * col; i++)
    {
        input_signal[i] = {input_re[i], input_im[i]};
        output_signal[i] = {output_re[i], output_im[i]};
        // //---------------------------------------------------------------
        Esym += pow(abs(input_signal[i]), 2) / complex<double>(row * col);
        double snr_linear = std::pow(10.0, snr / 10.0);
        No = Esym / snr_linear;
        //No = Esym / complex<double>(snr);
        noiseSigma = sqrt(No / complex<double>(2));
        n = noiseSigma * (randn() + randn() * 1i);
        // // std::complex<double> n(distribution(generator), distribution(generator));
        output_signal[i] = input_signal[i] + n;
        e_AWGN += output_signal[i]; // debug mode
        //---------------------------------------------------------------
        output_re[i] = output_signal[i].real();
        output_im[i] = output_signal[i].imag();
        //---------------------------------------------------------------
        // output_re[i] = input_re[i];
        // output_im[i] = input_im[i];
        // std::cout << "---awgn output_signal---" << output_signal[i] << std::endl;
    }
    // std::cout << "---awgn sum ---" << e_AWGN << std::endl;
    // std::cout << "---awgn average.real ---" << e_AWGN.real() / (row * col) << std::endl;
    // std::cout << "---awgn average.imag ---" << e_AWGN.imag() / (row * col) << std::endl;
    // free(input_signal);
    // free(output_signal);
}

void cpp_exp(float *A_theta_re, float *A_theta_im, float *t_theta, float d, float kc, int i, int j)
{
    complex<float> A_theta;
    A_theta = exp(I_1 * kc * std::complex<float>(i + 1) * d * sin((*t_theta) * complex<float>(PI / 180)));
    //std::cout << "---A_theta^ ---" << A_theta;

    *A_theta_re = A_theta.real();
    *A_theta_im = A_theta.imag();
}
void cpp_exp2(float *a_vector_re, float *a_vector_im, float *dr, float d, float kc, int i, int j)
{
    complex<float> a_vector;
    a_vector = exp(I_1 * kc * (complex<float>)j * d * sin(dr[i]));
    // std::cout << "---A_theta---" << A_theta;
    *a_vector_re = a_vector.real();
    *a_vector_im = a_vector.imag();
    // cout << "---a_vector---" << a_vector << endl;
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