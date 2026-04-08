#ifndef MATH_FUNC_H
#define MATH_FUNC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>
#include <stdint.h>

// 函數聲明
void cpp_abs(float *Real, float *Imaginary, float *result);
void cpp_sqrt(float *Real, float *Imaginary);
void cpp_sqrt_i(int16_t *Real, int16_t *Imaginary);
void cpp_sqrt_i32(int *Real, int *Imaginary);
void cpp_division(float *Re_a, float *Im_b, float *Re_c, float *Im_d);
void cpp_division_i(int16_t *Re_a, int16_t *Im_b, int16_t *Re_c, int16_t *Im_d);
void cpp_division_i32(int *Re_a, int *Im_b, int *Re_c, int *Im_d);
void cpp_division2(float Re_a, float Im_b, float *Re_c, float *Im_d, float *result_real, float *result_imag);
void cpp_division3(float *Re_a, float *Im_b, float *Re_c, float *Im_d);
void cpp_awgn(float *input_re, float *input_im, float *output_re, float *output_im, int snr, int row, int col);
void cpp_exp(float *A_theta_re, float *A_theta_im, float *t_theta, float d, float kc, float i, int j);
void cpp_exp_tx(float *A_theta_re, float *A_theta_im, float *t_theta, float d, float kc, float n, int j);
void cpp_exp_1(float *A_theta_re, float *A_theta_im, float t_theta, float d, float kc, float n);
void cpp_exp_1_tx(float *A_theta_re, float *A_theta_im, float t_theta, float d, float kc, float n);
void cpp_exp2(float *a_vector_re, float *a_vector_im, float *dr, float d, float kc, int i, float j);
void cpp_t_sig(float *t_sig_re, float *t_sig_im);
float cpp_20log_abs(float *S_MUSIC_re, float *S_MUSIC_im);
float cpp_rand(float Max_value);



void cpp_exp2_int32q14(int32_t *A_theta_re,
                  int32_t *A_theta_im,
                  const float *dr,   /* θ(rad) 陣列                */
                  float d,           /* 陣元間距                    */
                  float kc,          /* 2π/λ                        */
                  int   idx,         /* dr[] index (= elevation)   */
                  float n);

#ifdef __cplusplus
}
#endif

#endif // MATH_FUNC_H
