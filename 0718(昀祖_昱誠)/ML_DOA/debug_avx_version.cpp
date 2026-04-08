// AVX512
// g++ -mavx512f -g -o debug_avx_version debug_avx_version.cpp -Wall -Wextra -std=c++14
// ./debug_avx_version
#include <immintrin.h>
// C++
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <ccomplex>
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
//   std::cout << std::setprecision(6) << "Elapsed cmm:\t\t" << (timecomplex_matrix_end - timecomplex_matrix_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
//#include "cu_function.h"

// #include "matplotlib-cpp/matplotlibcpp.h"

#define PI acos(-1)
//#define cudaCheck(ans) {cudaAssert((ans), __FILE__, __LINE__);}

#define BLOCK_SIZE 16
#define PRINT_RESULT 1
#define PLOT_RESULT 0

using namespace std::literals::complex_literals;
const std::complex<double> I_1(0, 1);
extern "C"
{
    __attribute__((aligned(64))) double mat_C[1000000] = {0.0};
}
long int total_multiply_time = 0;
// using namespace std::complex_literals;
//  namespace plt = matplotlibcpp;

// inline void cudaAssert(cudaError_t code, const char *file, int line) {
//     if(code != cudaSuccess) {
//         fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         exit(code);
//     }
// }

// // warm up gpu for time measurement
// __global__ void warmup() {}

// print complex matrix matlab
void print_complex_matrix_matlab(std::complex<double> *matA, int rowA, int colA)
{
    std::cout << "[";
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            std::cout << std::setprecision(4) << matA[i * colA + j].real() << "+" << matA[i * colA + j].imag() << "i ";
        }
        std::cout << ";" << std::endl;
    }
    std::cout << "]" << std::endl;
}

// print complex matrix
void print_complex_matrix(std::complex<double> *matA, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << std::setw(10) << matA[i * colA + j] << " ";
            // printf("\t%.0f ", matA[i * colA + j].real());
            // printf("+ %.0fi", matA[i * colA + j].imag());
            //  std::cout << std::setprecision(6) << matA[i * colA + j] << " ";
        }
        printf("\n");
    }
}

// generate random number with normal_distribution
std::complex<double> randn()
{
    std::random_device randomness_device{};
    std::mt19937 pseudorandom_generator{randomness_device()};
    auto mean = 0.0;
    auto std_dev = 1.0;
    std::normal_distribution<> distribution{mean, std_dev};
    auto sample = distribution(pseudorandom_generator);
    return (std::complex<double>)(sample);
}

// add white gaussian noise
void awgn(std::complex<double> *input_signal, std::complex<double> *output_signal, int snr, int row, int col)
{
    std::complex<double> Esym;
    std::complex<double> No;
    std::complex<double> noiseSigma;
    std::complex<double> n;
    for (int i = 0; i < row * col; i++)
    {
        Esym += pow(abs(input_signal[i]), 2) / std::complex<double>(row * col);
        No = Esym / std::complex<double>(snr);
        noiseSigma = sqrt(No / std::complex<double>(2));
        n = noiseSigma * (randn() + randn() * 1i);
        output_signal[i] = input_signal[i] + n;
        // std::cout << "---awgn output_signal---" << output_signal[i] << std::endl;
    }
}

// complex matrix addition
void complex_matrix_addition(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
        }
    }
}

// complex matrix subtraction
void complex_matrix_subtraction(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA)
{
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA[i * colA + j].real(matA[i * colA + j].real() - matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() - matB[i * colA + j].imag());
        }
    }
}

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

// complex matrix multiplication ,colB = rowA  ; rowB = colA (M=32 ~= 9.4ms)
void complex_matrix_multiplication(std::complex<double> *matA, std::complex<double> *matB, std::complex<double> *matC, int rowA, int rowB, int colB)
{
    //----------------------------------------------------------
    struct timespec start, end;
    double time_used;
    //----------------------------------------------------------
    struct timeval start_scalar, end_scalar, diff_scalar;
    memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
    __m256d simd_matA, simd_matB, simd_matC;

    clock_gettime(CLOCK_MONOTONIC, &start);
    gettimeofday(&start_scalar, NULL);
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colB; ++j)
        {
            for (int k = 0; k < rowB; ++k)
            {
                matC[i * colB + j] += matA[i * rowB + k] * matB[k * colB + j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    gettimeofday(&end_scalar, NULL);
    timersub(&end_scalar, &start_scalar, &diff_scalar);
    total_multiply_time += diff_scalar.tv_usec; // sum of diff_multiply

    struct timespec temp = diff(start, end);
    time_used = temp.tv_sec + (double)temp.tv_nsec / 1000000000.0;
    // printf(L_PURPLE "-------------------------\nTime = %.0f(ns)\n" CLOSE, time_used * 1000000000);
    // printf(L_PURPLE "\nElapsed scalar time: %ld(us)\n" CLOSE, (long int)diff_scalar.tv_usec);
}

// get complex matrix by column
void complex_matrix_get_columns(std::complex<double> *matA, std::complex<double> *matCol, int rowA, int colA, int colTarget)
{
    for (int i = 0; i < rowA; ++i)
    {
        matCol[i] = matA[i * colA + colTarget];
    }
}

// get complex matrix by row
void complex_matrix_get_rows(std::complex<double> *matA, std::complex<double> *matRow, int rowA, int colA, int rowTarget)
{
    for (int i = 0; i < colA; ++i)
    {
        matRow[i] = matA[rowTarget * colA + i];
    }
}

// complex matrix conjugate transpose (M=32 -> 0.001ms)
void complex_matrix_conjugate_transpose(std::complex<double> *matA, int rowA, int colA)
{
    float cmct_start, cmct_end;
    // cmct_start = clock();
    std::complex<double> *temp = (std::complex<double> *)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            matA[j * rowA + i].real(temp[i * colA + j].real());
            matA[j * rowA + i].imag(-temp[i * colA + j].imag());
        }
    }
    // cmct_end = clock();
    // std::cout << std::setprecision(6) << CYAN "cmct:\t\t" << (cmct_end - cmct_start) / CLOCKS_PER_SEC * 1000 << " ms" CLOSE << std::endl;
    free(temp);
}

// complex matrix conjugate transpose and multiplication
void complex_matrix_conjugate_transpose_multiplication(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA)
{
    float cmctm_start, cmctm_end;
    std::complex<double> *temp = (std::complex<double> *)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    // cmctm_start = clock();
    complex_matrix_conjugate_transpose(temp, rowA, colA);
    complex_matrix_multiplication(matA, temp, matB, rowA, colA, rowA); // colB = rowA
    // cmctm_end = clock();
    // std::cout << std::setprecision(6) << "Elapsed memcpy:\t\t" << (cmctm_end - cmctm_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

    free(temp);
}

// compute Pn: matlab co.de: (Pn=Pn+vet_noise(:,ii)*vet_noise(:,ii)';), where (ii=1:length(vet_noise(1,:)))
void compute_Pn(std::complex<double> *Pn, std::complex<double> *vet_noise, int M, int len_t_theta)
{
    std::complex<double> *vet_noise_temp = (std::complex<double> *)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *Pn_temp = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // print_complex_matrix(vet_noise, M, M - len_t_theta);

    for (int i = 0; i < M - len_t_theta; ++i)
    {
        complex_matrix_get_columns(vet_noise, vet_noise_temp, M, M - len_t_theta, i);
        complex_matrix_conjugate_transpose_multiplication(vet_noise_temp, Pn_temp, M, 1);
        complex_matrix_addition(Pn, Pn_temp, M, M);
    }
    free(vet_noise_temp);
    free(Pn_temp);
}

// compute S_MUSIC: matlab code: (S_MUSIC(i)=1/(a_vector'*Pn*a_vector))
std::complex<double> compute_S_MUSIC(std::complex<double> *a_vector, std::complex<double> *Pn, int M)
{
    std::complex<double> *Pn_a_vector_temp = (std::complex<double> *)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC_temp = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    complex_matrix_multiplication(Pn, a_vector, Pn_a_vector_temp, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector, M, 1);
    complex_matrix_multiplication(a_vector, Pn_a_vector_temp, S_MUSIC_temp, 1, M, 1);
    std::complex<double> S_MUSIC = std::complex<double>(1) / S_MUSIC_temp[0];
    // std::cout << "---S_MUSIC---" << S_MUSIC << std::endl;
    // std::cout << "---std::complex<double>(1)---" << std::complex<double>(1) << std::endl;
    free(Pn_a_vector_temp);
    free(S_MUSIC_temp);
    return S_MUSIC;
}

// QR decomposer for c code
void qr(std::complex<double> *A, std::complex<double> *Q, std::complex<double> *R, int row, int col)
{
    std::complex<double> *Q_col = (std::complex<double> *)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *vector_cur = (std::complex<double> *)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *Qvector_cur = (std::complex<double> *)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *power_cur = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    std::complex<double> *power_val = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_val = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_Qvector_cur = (std::complex<double> *)malloc(row * 1 * sizeof(std::complex<double>));
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q[i].real(1); // value 1 (unit matrix)
        R[i].real(1); // value 1 (unit matrix)
    }
    for (int i = 0; i < col; ++i)
    {
        for (int m = 0; m < row; ++m)
        {
            Q[m * col + i] = A[m * col + i];
        }
        complex_matrix_get_columns(Q, Q_col, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col, row, 1);
        memset(power_cur, 0, sizeof(std::complex<double>));
        complex_matrix_conjugate_transpose_multiplication(Q_col, power_cur, 1, row);
        power_cur[0] = sqrt(power_cur[0]);
        if (i > 0)
        {
            complex_matrix_get_columns(A, vector_cur, row, col, i);
            std::complex<double> *Q_col_proj = (std::complex<double> *)malloc(row * i * sizeof(std::complex<double>));
            std::complex<double> *proj_vector = (std::complex<double> *)malloc(i * sizeof(std::complex<double>));
            memset(proj_vector, 0, i * sizeof(std::complex<double>));
            for (int j = 0; j < i; ++j)
            {
                for (int m = 0; m < row; ++m)
                {
                    Q_col_proj[m * i + j] = Q[m * col + j];
                }
            }
            complex_matrix_conjugate_transpose(Q_col_proj, row, i);
            complex_matrix_multiplication(Q_col_proj, vector_cur, proj_vector, i, row, 1);
            complex_matrix_conjugate_transpose(Q_col_proj, i, row);
            memset(Q_col, 0, row * 1 * sizeof(std::complex<double>));
            complex_matrix_multiplication(Q_col_proj, proj_vector, Q_col, row, i, 1);
            complex_matrix_subtraction(vector_cur, Q_col, row, 1);
            for (int m = 0; m < row; ++m)
            {
                Q[m * col + i] = vector_cur[m];
            }
            for (int j = 0; j < i; ++j)
            {
                R[i + col * j] = proj_vector[j];
            }
            free(Q_col_proj);
            free(proj_vector);
        }
        complex_matrix_get_columns(Q, Q_col, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col, row, 1);
        memset(power_val, 0, sizeof(std::complex<double>));
        complex_matrix_conjugate_transpose_multiplication(Q_col, power_val, 1, row);
        power_val[0] = sqrt(power_val[0]);

        // 1e-4 = 0.0001
        if (power_val[0].real() / power_cur[0].real() < 1e-4)
        {
            R[i * row + i] = 0;
            // span again
            for (int m = 0; m < row; ++m)
            {
                Q[m * col + i] = 0;
            }
            Q[i * row + i].real(1);
            complex_matrix_get_columns(Q, vector_cur, row, col, i);
            for (int j = 0; j < i; ++j)
            {
                complex_matrix_get_columns(Q, Qvector_cur, row, col, j);
                memset(proj_val, 0, sizeof(std::complex<double>));
                complex_matrix_conjugate_transpose(Qvector_cur, row, 1);
                complex_matrix_multiplication(Qvector_cur, vector_cur, proj_val, 1, row, 1);
                complex_matrix_conjugate_transpose(Qvector_cur, 1, row);
                complex_matrix_get_columns(Q, Q_col, row, col, i);
                memset(proj_Qvector_cur, 0, row * 1 * sizeof(std::complex<double>));
                complex_matrix_multiplication(Qvector_cur, proj_val, proj_Qvector_cur, row, 1, 1);
                complex_matrix_subtraction(Q_col, proj_Qvector_cur, row, 1);
                for (int m = 0; m < row; ++m)
                {
                    Q[m * col + i] = Q_col[m];
                }
            }
            complex_matrix_get_columns(Q, Q_col, row, col, i);
            complex_matrix_conjugate_transpose(Q_col, row, 1);
            memset(power_val, 0, sizeof(std::complex<double>));
            complex_matrix_conjugate_transpose_multiplication(Q_col, power_val, 1, row);
            power_val[0] = sqrt(power_val[0]);
            complex_matrix_conjugate_transpose(Q_col, 1, row);
            for (int m = 0; m < row; ++m)
            {
                Q[m * col + i] /= power_val[0]; // Q[m * col + i] = Q[m * col + i] / power_val[0]
            }
        }
        else
        {
            R[i * row + i] = power_val[0];
            for (int m = 0; m < row; ++m)
            {
                Q[m * col + i] /= power_val[0];
            }
        }
    }
    free(Q_col);
    free(vector_cur);
    free(Qvector_cur);
    free(power_cur);
    free(power_val);
    free(proj_val);
    free(proj_Qvector_cur);
}

// compute eigen upper triangular
void eigen_upper_triangular(std::complex<double> *A, std::complex<double> *eigenvalue, std::complex<double> *eigenvector, int row, int col)
{
    std::complex<double> *vector_cur = (std::complex<double> *)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *eigen_element_cur = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    std::complex<double> *vector_cur_temp = (std::complex<double> *)malloc(sizeof(std::complex<double>));
    std::complex<double> *A_col = (std::complex<double> *)malloc(1 * col * sizeof(std::complex<double>));
    std::complex<double> diff_eigen_value = 0;
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
            {
                A[i * col + j].real(0);
                A[i * col + j].imag(0);
            }
            if (i == j)
            {
                eigenvalue[i * col + j] = A[i * col + j];
                eigenvector[i * col + j].real(1);
                // printf(PURPLE "eigenvalue[%d] = %.2f\n" CLOSE, i * col + j, eigenvalue[i * col + j]);
            }
        }
    }
    for (int i = 0; i < col; ++i)
    {
        complex_matrix_get_columns(eigenvector, vector_cur, row, col, i);
        for (int j = i - 1; j > -1; --j)
        {
            diff_eigen_value = eigenvalue[i * col + i] - eigenvalue[j * col + j];
            if (diff_eigen_value.real() < 1e-8)
                eigen_element_cur[0] = 0;
            else
            {
                complex_matrix_get_rows(A, A_col, row, col, j);
                complex_matrix_multiplication(A_col, vector_cur, eigen_element_cur, 1, row, 1);
                eigen_element_cur[0] = eigen_element_cur[0] / diff_eigen_value;
            }
            vector_cur[j] = eigen_element_cur[0];
        }
        complex_matrix_conjugate_transpose(vector_cur, row, 1);
        complex_matrix_conjugate_transpose_multiplication(vector_cur, vector_cur_temp, 1, row);
        vector_cur_temp[0] = sqrt(vector_cur_temp[0]);
        complex_matrix_conjugate_transpose(vector_cur, 1, row);
        for (int m = 0; m < row; ++m)
        {
            eigenvector[m * col + i] = vector_cur[m] / vector_cur_temp[0];
        }
    }
    free(vector_cur);
    free(eigen_element_cur);
    free(vector_cur_temp);
    free(A_col);
}

// compute complex eigenvector and eigenvalue for c code
// void eigen(A:input array, Ve:eigen vector, De:eigen value, row , col , iter)
void eigen(std::complex<double> *A, std::complex<double> *Ve, std::complex<double> *De, int row, int col, int iter)
{
    float time_QR_start, time_QR_end; // time initial
    std::complex<double> *Q = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *R = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp_clone = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_temp[i].real(1);
    }
    time_QR_start = clock();
    for (int i = 0; i < iter; ++i)
    {
        qr(A, Q, R, row, col);
        complex_matrix_multiplication(R, Q, A, row, row, col);
        complex_matrix_multiplication(Q_temp, Q, Q_temp_clone, row, row, col);
        memcpy(Q_temp, Q_temp_clone, row * col * sizeof(std::complex<double>));
    }
    time_QR_end = clock();
    std::cout << std::setprecision(6) << L_CYAN "Elapsed QR:\t\t" << (time_QR_end - time_QR_start) / CLOCKS_PER_SEC * 1000 << " ms, Iteration = " << iter << CLOSE << std::endl;

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
                A[i * col + j] = 0;
        }
    }
    std::complex<double> *YY0 = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *XX0 = (std::complex<double> *)calloc(row * col, sizeof(std::complex<double>));
    eigen_upper_triangular(A, YY0, XX0, row, col);
    memcpy(De, YY0, row * col * sizeof(std::complex<double>));
    complex_matrix_multiplication(Q_temp, XX0, Ve, row, row, col);
    free(Q);
    free(R);
    free(Q_temp);
    free(Q_temp_clone);
    free(YY0);
    free(XX0);
}

// compute the MUSIC DOA in one dimension on CPU
void MUSIC_DOA_1D_CPU(int M, int snr, int qr_iter, int angle, float *result)
{
#ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", snr);
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("Angle:\t\t%d\n", angle);
// printf("Multiple input size:\t%d\n", multi_input);
#endif
    // generate the signal
    // time initial
    float timeStart, timeEnd;
    // parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const double lemda = (double)c / (double)fc;
    std::complex<double> d(lemda * 0.5);
    std::complex<double> kc(2.0 * PI / lemda);
    const int nd = 500;
    // angle setting
    const int len_t_theta = 1;
    std::complex<double> *t_theta = (std::complex<double> *)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
// t_theta[1].real(12);
// t_theta[2].real(20);
#ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for (int i = 0; i < len_t_theta; ++i)
    {
        if (i != len_t_theta - 1)
            std::cout << t_theta[i].real() << ", ";
        else
            std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
#endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double> *)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double> *)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    for (int i = 0; i < len_t_theta; ++i)
    {
        for (int j = 0; j < nd; ++j)
        {
            t_sig[i * nd + j] = (randn() + randn() * I_1) / std::complex<double>(sqrt(2));
            // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
        }
    }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

    // receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, snr, M, nd);

    // music algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for (int i = 0; i < M * M; ++i)
        R_xx[i] /= std::complex<double>(M);
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    // eigen(R_xx, Ve, De, M, M, qr_iter);
    // printf("Ve \n");
    // print_complex_matrix(Ve,M,M);
    // printf("De \n");
    // print_complex_matrix(De,M,M);
    // return;
    // get vet_noise (M, M - len_t_theta): part of Ve (eigenvector)
    std::complex<double> *vet_noise = (std::complex<double> *)malloc(M * (M - len_t_theta) * sizeof(std::complex<double>));
    for (int i = 0; i < M; ++i)
    {
        for (int j = len_t_theta; j < M; ++j)
        {
            vet_noise[i * (M - len_t_theta) + j - len_t_theta] = Ve[i * M + j];
        }
    }
    // Pn matrix (M, M)
    std::complex<double> *Pn = (std::complex<double> *)calloc(M * M, sizeof(std::complex<double>));
    compute_Pn(Pn, vet_noise, M, len_t_theta);
    // timestamp end
    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MUSIC (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
#endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

    // array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double *)malloc(len_dth * sizeof(double));
    double *dr = (double *)malloc(len_dth * sizeof(double));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_MUSIC_dB
    std::complex<double> *a_vector = (std::complex<double> *)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC = (std::complex<double> *)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MUSIC_dB = (double *)malloc(len_dth * sizeof(double));
    // timestamp start
    timeStart = clock();
    for (int i = 0; i < len_dth; ++i)
    { // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
        S_MUSIC[i] = compute_S_MUSIC(a_vector, Pn, M);
        // compute S_MUSIC_dB
        S_MUSIC_dB[i] = 20 * log10(abs(S_MUSIC[i]));
    }
    // find Max and position
    double max_temp = S_MUSIC_dB[0];
    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        if (S_MUSIC_dB[i] > max_temp)
        {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    // print the result
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
#endif
    float error[3];
    float errorFinal = 0;
    for (int i = 0; i < 3; ++i)
    {
        error[i] = abs(dth[position] - t_theta[i].real());
        if (i == 0)
            errorFinal = error[i];
        else if (error[i] < errorFinal)
            errorFinal = error[i];
    }
    if (errorFinal > result[0])
        result[0] = errorFinal;
    else if (errorFinal == 0)
        result[1]++;

    // plot the result
    // #ifdef PLOT_RESULT
    // std::vector<double> S_MUSIC_dB_vec(S_MUSIC_dB, S_MUSIC_dB + len_dth);
    // std::vector<double> dth_vec(dth, dth + len_dth);
    // plt::plot(dth_vec, S_MUSIC_dB_vec, "blue");
    // plt::title("MUSIC DOA Estimation");
    // plt::xlabel("Theta (degree)");
    // plt::ylabel("Power Spectrum (dB)");
    // plt::xlim(dth[0], dth[len_dth - 1]);
    // plt::grid(true);
    // plt::show();
    // #endif

    // free memory
    free(t_theta);
    free(A_theta);
    free(t_sig);
    free(sig_co);
    free(x_r);
    free(R_xx);
    free(Ve);
    free(De);
    free(vet_noise);
    free(Pn);
    free(dth);
    free(dr);
    free(a_vector);
    free(S_MUSIC);
    free(S_MUSIC_dB);
}

void MUSIC_DOA_1D_CPU_test(int M, int qr_iter, int angle, float *result, int SNR)
{
    // time initial
    float timeMusic_start, timeMusic_end;                       // Total MUSIC Algorithm time
    float timeAwgn_start, timeAwgn_end;                         // AWGN white noise time
    float timecomplex_matrix_start, timecomplex_matrix_end;     // complex_matrix time
    float timeconjugate_matrix_start, timeconjugate_matrix_end; // complex_matrix_conjugate_transpose_multiplication time
    float timeStart, timeEnd;                                   // (cpu)time
    float timeEigen_start, timeEigen_end;                       // Eigen function time
    float timeCompute_start, timeCompute_end;                   // Compute_Pn time
    struct timeval start_cmctm, end_cmctm, diff_cmctm;
    timeMusic_start = clock();
#ifdef PRINT_RESULT
    printf("---------------\n");
    printf("--MUSIC DOA--\n");
    printf("---------------\n");
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);

#endif
    // generate the signal
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const double lemda = (double)c / (double)fc;
    std::complex<double> d(lemda * 0.5);       // For Real part
    std::complex<double> kc(2.0 * PI / lemda); // For Real part
    const int nd = 512;                        //
    // angle setting
    const int len_t_theta = 1;
    std::complex<double> *t_theta = (std::complex<double> *)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
// t_theta[1].real(12);
// t_theta[2].real(20);
// timeStart_1 = clock();
#ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for (int i = 0; i < len_t_theta; ++i)
    {
        if (i != len_t_theta - 1)
            std::cout << t_theta[i].real() << ", ";
        else
            std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
#endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double> *)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
            // std::cout << "---A_theta---" << A_theta[i * len_t_theta + j] << std::endl;
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double> *)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    // memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    for (int i = 0; i < len_t_theta; ++i)
    {
        for (int j = 0; j < nd; ++j)
        {
            t_sig[i * nd + j] = (randn() + randn() * I_1) / std::complex<double>(sqrt(2));
            // std::cout << "---t_sig---" << t_sig[i * nd + j] << std::endl;

            // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
        }
    }
    timecomplex_matrix_start = clock();
    // print_complex_matrix(t_sig ,1 ,nd);

    // sig_co matrix (M, nd)
    // timeEnd_1 = clock();
    std::complex<double> *sig_co = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);
    // print_complex_matrix(sig_co, M, nd);
    timecomplex_matrix_end = clock();

    // receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    timeAwgn_start = clock();
    awgn(sig_co, x_r, SNR, M, nd);
    // for (int a = 0; a < M * nd; a++)
    // {
    //     std::cout << "---awgn---" << x_r[a] << std::endl;
    // }

    timeAwgn_end = clock();
    // print_complex_matrix(x_r, M, nd );
    std::cout << std::setprecision(6) << "Elapsed AWGN:\t\t" << (timeAwgn_end - timeAwgn_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

    // music algorithm
    // R_xx matrix (M, M)

    std::complex<double> *R_xx = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    // x_r = matA; R_xx = matB;  M = rowA;  nd = colA;
    timeconjugate_matrix_start = clock();
    gettimeofday(&start_cmctm, NULL);
    printf(GREEN "rowA = %d, colA = %d\n" CLOSE, M, nd);
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    // print_complex_matrix(R_xx, M, nd);

    gettimeofday(&end_cmctm, NULL);
    timeconjugate_matrix_end = clock();

    timersub(&end_cmctm, &start_cmctm, &diff_cmctm);
    std::cout << std::setprecision(6) << "Elapsed cmctm:\t\t" << (timeconjugate_matrix_end - timeconjugate_matrix_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\ncmctm time: %ld(usec)\n", (long int)diff_cmctm.tv_usec);

    for (int i = 0; i < M * M; ++i)
    {
        R_xx[i] /= std::complex<double>(M);
    }
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock(); // Total time

    timeEigen_start = clock(); // start Eigen time
    eigen(R_xx, Ve, De, M, M, qr_iter);
    // printf("----------Ve------------\n");
    // print_complex_matrix(Ve, M, M);
    // printf("----------De------------\n");
    // print_complex_matrix(De, M, M);

    //========================== Write file ==========================
    // FILE *fptr_re;
    // FILE *fptr_im;
    // int i, j;
    // // use appropriate location if you are using MacOS or Linux
    // fptr_re = fopen("log/data_Real.txt", "w");
    // fptr_im = fopen("log/data_Imag.txt", "w");
    // if (fptr_re == NULL)
    // {
    //     printf("Error!");
    //     exit(1);
    // }
    // // printf("Data: ");
    // for (int a = 0; a < M * M; a++)
    // {
    //     if (a < 300 - 1)
    //     {
    //         fprintf(fptr_re, "%.4f,", Ve[a].real());
    //         fprintf(fptr_im, "%.4f,", Ve[a].imag());
    //         // printf("(%f,%f), ", Ve[a].real(), Ve[a].imag());
    //     }
    //     else
    //     {
    //         fprintf(fptr_re, "%.4f,", Ve[a].real());
    //         fprintf(fptr_im, "%.4f,", Ve[a].imag());
    //         // printf("(%f,%f), ", Ve[a].real(), Ve[a].imag());
    //     }
    // }
    // fclose(fptr_re);
    // fclose(fptr_im);

    //================================================================
    // get vet_noise (M, M - len_t_theta): part of Ve (eigenvector)
    timeEigen_end = clock();
    std::cout << std::setprecision(6) << "Elapsed Eigen time:\t" << (timeEigen_end - timeEigen_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

    std::complex<double> *vet_noise = (std::complex<double> *)malloc(M * (M - len_t_theta) * sizeof(std::complex<double>));
    for (int i = 0; i < M; ++i)
    {
        for (int j = len_t_theta; j < M; ++j)
        {
            vet_noise[i * (M - len_t_theta) + j - len_t_theta] = Ve[i * M + j];
            // std::cout << "--- vet_noise[i]---" << vet_noise[i * (M - len_t_theta) + j - len_t_theta] << std::endl;
        }
    }
    timeCompute_start = clock();
    //========================== Write vet_noise file ==========================
    // FILE *vet_noise_re;
    // FILE *vet_noise_im;
    // // use appropriate location if you are using MacOS or Linux
    // vet_noise_re = fopen("log/vet_noise_Real.txt", "w");
    // vet_noise_im = fopen("log/vet_noise_Imag.txt", "w");
    // if (fptr_re == NULL)
    // {
    //     printf("Error!");
    //     exit(1);
    // }
    // printf("\nWrite vet_noise file: \n");
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = len_t_theta; j < M; ++j)
    //     {
    //         fprintf(vet_noise_re, "%.4f,", vet_noise[i * (M - len_t_theta) + j - len_t_theta].real());
    //         fprintf(vet_noise_im, "%.4f,", vet_noise[i * (M - len_t_theta) + j - len_t_theta].imag());
    //         printf("(%f,%f), ", vet_noise[i * (M - len_t_theta) + j - len_t_theta].real(), vet_noise[i * (M - len_t_theta) + j - len_t_theta].imag());
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // fclose(vet_noise_re);
    // fclose(vet_noise_im);

    //================================================================
    // Pn matrix (M, M)
    std::complex<double> *Pn = (std::complex<double> *)calloc(M * M, sizeof(std::complex<double>));
    compute_Pn(Pn, vet_noise, M, len_t_theta);
    //========================== Write file ==========================
    // FILE *Pn_re;
    // FILE *Pn_im;
    // // use appropriate location if you are using MacOS or Linux
    // Pn_re = fopen("log/Pn_Real.txt", "w");
    // Pn_im = fopen("log/Pn_Imag.txt", "w");
    // if (fptr_re == NULL)
    // {
    //     printf("Error!");
    //     exit(1);
    // }
    // // printf("Data: ");
    // for (int a = 0; a < M * M; a++)
    // {
    //     if (a < M * M - 1)
    //     {
    //         fprintf(Pn_re, "%.4f,", Pn[a].real());
    //         fprintf(Pn_im, "%.4f,", Pn[a].imag());
    //         // printf("(%f,%f), ", Pn[a].real(), Pn[a].imag());
    //     }
    //     else
    //     {
    //         fprintf(Pn_re, "%.4f,", Pn[a].real());
    //         fprintf(Pn_im, "%.4f,", Pn[a].imag());
    //         // printf("(%f,%f), ", Pn[a].real(), Pn[a].imag());
    //     }
    // }
    // fclose(Pn_re);
    // fclose(Pn_im);

    //================================================================
    // printf("----------Pn------------\n");
    // print_complex_matrix(Pn, M, M);
    // timestamp end
    timeCompute_end = clock();
    std::cout << std::setprecision(6) << "Elapsed compute_Pn:\t" << (timeCompute_end - timeCompute_start) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MUSIC (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// std::cout << std::setprecision(6) << "test: \t\t" << (timeEnd_1 - timeStart_1) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
#endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

    // array pattern
    // parameter setting
    // const int len_dth = 401;
    const int len_dth = 1201;
    double *dth = (double *)malloc(len_dth * sizeof(double));
    double *dr = (double *)malloc(len_dth * sizeof(double));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        dth[i] = -60 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_MUSIC_dB
    std::complex<double> *a_vector = (std::complex<double> *)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC = (std::complex<double> *)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MUSIC_dB = (double *)malloc(len_dth * sizeof(double));

    // timestamp start
    timeStart = clock();
    for (int i = 0; i < len_dth; ++i)
    { // can be paralleled to compute S_MUSIC_dB
        for (int j = 0; j < M; ++j)
        {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
            // std::cout << "---a_vector---" << a_vector[j] << std::endl;
        }
        S_MUSIC[i] = compute_S_MUSIC(a_vector, Pn, M);
        // printf("\tS_MUSIC(%f,%f), ", S_MUSIC[i].real(), S_MUSIC[i].imag());

        // compute S_MUSIC_dB
        S_MUSIC_dB[i] = 20 * log10(abs(S_MUSIC[i]));
        // std::cout << "20log" << S_MUSIC[i] << " = " << 20 * log10(abs(S_MUSIC[i])) << std::endl;
        // printf("S_MUSIC_dB[%d] = %.4f\n", i, S_MUSIC_dB[i]);
    }
    // find Max and position
    double max_temp = S_MUSIC_dB[0];

    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        if (S_MUSIC_dB[i] > max_temp)
        {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }
    }
    printf(YELLOW "max_temp = %f(dB)\n" CLOSE, max_temp);

    printf("position : %d\n", position);
    // timestamp end
    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;

    timeMusic_end = clock(); // Total time
    std::cout << std::setprecision(6) << L_GREEN "Total MUSIC time:\t" << (timeMusic_end - timeMusic_start) / CLOCKS_PER_SEC * 1000 << " ms" CLOSE << std::endl;
    std::cout << std::setprecision(6) << L_GREEN "Total Multiply time:\t" << total_multiply_time / 1000 << " ms" CLOSE << std::endl
              << "-----------------------------------------" << std::endl
              << std::endl;
#endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if (error > result[0])
        result[0] = error;
    if (error != 0)
        result[1] += pow(error, 2);

        // plot the result
#ifdef PLOT_RESULT
// std::vector<double> S_MUSIC_dB_vec(S_MUSIC_dB, S_MUSIC_dB + len_dth);
// std::vector<double> dth_vec(dth, dth + len_dth);
// plt::plot(dth_vec, S_MUSIC_dB_vec, "blue");
// plt::title("MUSIC DOA Estimation");
// plt::xlabel("Theta (degree)");
// plt::ylabel("Power Spectrum (dB)");
// plt::xlim(dth[0], dth[len_dth - 1]);
// plt::grid(true);
// plt::show();
#endif

    // free memory
    free(t_theta);
    free(A_theta);
    free(t_sig);
    free(sig_co);
    free(x_r);
    free(R_xx);
    free(Ve);
    free(De);
    free(vet_noise);
    free(Pn);
    free(dth);
    free(dr);
    free(a_vector);
    free(S_MUSIC);
    free(S_MUSIC_dB);
}

void MVDR_DOA_1D_CPU_test(int M, int qr_iter, int angle, float *result, int snr)
{
#ifdef PRINT_RESULT
    printf("--MVDR DOA--\n");
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", snr);
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("Angle:\t\t\t%d\n", angle);
#endif
    // generate the signal
    // time initial
    float timeStart, timeEnd;
    // float timeStart_1, timeEnd_1;
    //  parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const double lemda = (double)c / (double)fc;
    std::complex<double> d(lemda * 0.5);
    std::complex<double> kc(2.0 * PI / lemda);
    const int nd = 500;
    // angle setting
    const int len_t_theta = 1;
    std::complex<double> *t_theta = (std::complex<double> *)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
// t_theta[1].real(12);
// t_theta[2].real(20);
// timeStart_1 = clock();
#ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for (int i = 0; i < len_t_theta; ++i)
    {
        if (i != len_t_theta - 1)
            std::cout << t_theta[i].real() << ", ";
        else
            std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
#endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double> *)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < len_t_theta; ++j)
        {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double> *)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    // memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    for (int i = 0; i < len_t_theta; ++i)
    {
        for (int j = 0; j < nd; ++j)
        {
            // t_sig[i * nd + j] = (randn() + randn() * I_1) / std::complex<double>(sqrt(2));
            t_sig[i * nd + j] = (randn() + randn()) / std::complex<double>(sqrt(2));
            // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
        }
    }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);
    // receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double> *)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, snr, M, nd);

    // mvdr algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    // printf("----------R_xx------------\n");
    // print_complex_matrix(R_xx,M,M);
    for (int i = 0; i < M * M; ++i)
        R_xx[i] /= std::complex<double>(M);
    // printf("----------R_xx------------\n");
    // print_complex_matrix(R_xx,M,M);
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    eigen(R_xx, Ve, De, M, M, qr_iter);
    // printf("----------Ve------------\n");
    // print_complex_matrix(Ve, M, M);
    // printf("----------De------------\n");
    // print_complex_matrix(De, M, M);
    std::complex<double> *R_xx_inv_1 = (std::complex<double> *)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *Pn = (std::complex<double> *)calloc(M * M, sizeof(std::complex<double>));
    for (int i = 0; i < M * M; i += (M + 1))
    {
        if (abs(De[i]) < 0.00000000001)
        {
            De[i].real(1000000);
            De[i].imag(0);
        }
        else
            De[i] = std::complex<double>(1) / De[i];
    }
    // printf("----------R_xx------------\n");
    // print_complex_matrix(R_xx,M,M);
    complex_matrix_multiplication(Ve, De, R_xx_inv_1, M, M, M);
    // printf("----------R_xx_inv_1------------\n");
    // print_complex_matrix(R_xx_inv_1,M,M);
    complex_matrix_conjugate_transpose(Ve, M, M);
    complex_matrix_multiplication(R_xx_inv_1, Ve, Pn, M, M, M);

    // printf("----------Pn------------\n");
    // print_complex_matrix(Pn, M, M);
    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MVDR (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
#endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

    // array pattern
    // parameter setting
    // const int len_dth = 401;
    const int len_dth = 1201;
    double *dth = (double *)malloc(len_dth * sizeof(double));
    double *dr = (double *)malloc(len_dth * sizeof(double));
    for (int i = 0; i < len_dth; ++i)
    { // do only one time, no need to be paralleled
        // dth[i] = -10 + 0.1 * i;
        dth[i] = -60 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
        // printf("dth[%d] : %f\n",i,dth[i]);
    }
    // compute S_MVDR_dB
    std::complex<double> *a_vector = (std::complex<double> *)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MVDR = (std::complex<double> *)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MVDR_dB = (double *)malloc(len_dth * sizeof(double));
    // timestamp start
    timeStart = clock();
    //---------------------------------------------------------------
    FILE *fp_excel = NULL;
    fp_excel = fopen("data/s_MVDR_dB.csv", "w");
    //---------------------------------------------------------------
    for (int i = 0; i < len_dth; ++i)
    { // can be paralleled to compute S_MVDR_dB
        for (int j = 0; j < M; ++j)
        {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
        S_MVDR[i] = compute_S_MUSIC(a_vector, Pn, M);

        // compute S_MVDR_dB
        S_MVDR_dB[i] = 20 * log10(abs(S_MVDR[i]));
        fprintf(fp_excel, "%.1f,%.4f\n", (-60 + 0.1 * i), S_MVDR_dB[i]);
    }
    // find Max and position
    double max_temp = S_MVDR_dB[0];
    int position = 0;
    for (int i = 0; i < len_dth; ++i)
    {
        if (S_MVDR_dB[i] > max_temp)
        {
            max_temp = S_MVDR_dB[i];
            position = i;
        }
    }
    printf("position : %d\n", position);
    // timestamp end
    timeEnd = clock();
#ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl
              << "-----------------------------------------" << std::endl
              << std::endl;
#endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if (error > result[0])
        result[0] = error;
    if (error != 0)
        result[1] += pow(error, 2);

        // plot the result
#ifdef PLOT_RESULT
// std::vector<double> S_MVDR_dB_vec(S_MVDR_dB, S_MVDR_dB + len_dth);
// std::vector<double> dth_vec(dth, dth + len_dth);
// plt::plot(dth_vec, S_MVDR_dB_vec, "blue");
// plt::title("MUSIC DOA Estimation");
// plt::xlabel("Theta (degree)");
// plt::ylabel("Power Spectrum (dB)");
// plt::xlim(dth[0], dth[len_dth - 1]);
// plt::grid(true);
// plt::show();
#endif

    // free memory
    free(t_theta);
    // free(A_theta);
    // free(t_sig);
    free(sig_co);
    free(x_r);
    free(R_xx);
    free(Ve);
    free(De);
    // free(vet_noise);
    free(Pn);
    free(dth);
    free(dr);
    free(a_vector);
    free(S_MVDR);
    free(S_MVDR_dB);
}

int main(void)
{
    int M = 14;
    int snr = 100000;
    int qr_iter = 10;
    float result[3] = {0};
    int angle = 50;
    int iter = 1;
    MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
    MVDR_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);

    //-----------------------------------------------------------
    // Random value
    //-----------------------------------------------------------
    // std::complex<double> A[10000] = {0.0};
    // std::complex<double> B[10000] = {0.0};

    // int rowA = 64;
    // int colA = rowA;
    // int rowB = 64;
    // int colB = 64;
    // srand(time(NULL));
    // int i, j;
    // // printf("random A = \n");
    // for (i = 0; i < rowA * colA; i++)
    // {
    //     // rand() % Max-min+1)+min
    //     A[i] = {(rand() % 31) - 15, (rand() % 31) - 15};
    //     // printf("(%.0f + %.0fi), ", A[i].real(), A[i].imag());
    // }
    // // printf("\n");
    // // printf("random B = \n");
    // for (i = 0; i < rowB * colB; i++)
    // {
    //     // rand() % Max-min+1)+min
    //     B[i] = {(rand() % 31) - 15, (rand() % 31) - 15};
    //     // printf("(%.0f + %.0fi), ", B[i].real(), B[i].imag());
    // }
    // // printf("\n");
    //-----------------------------------------------------------
    // 20
    // int rowA = 20;
    // int colA = rowA;
    // int rowB = 20;
    // int colB = 20;
    // std::complex<double> A[] = {{36, -1}, {-27, 18}, {45, -10}, {45, -27}, {24, -4}, {-7, 23}, {30, -28}, {14, 43}, {40, -10}, {-22, 23}, {-29, -20}, {8, -22}, {48, 46}, {37, -16}, {-17, -16}, {1, 19}, {45, -22}, {39, -27}, {-5, -24}, {16, -20}, {2, -28}, {36, 45}, {35, 17}, {35, 12}, {4, -23}, {-12, -30}, {-8, -7}, {44, 16}, {-19, -6}, {17, 3}, {12, -12}, {31, 8}, {45, 47}, {27, 2}, {10, 42}, {38, 26}, {-12, -1}, {-12, 39}, {-21, 38}, {45, -2}, {-22, 10}, {-28, 11}, {49, -7}, {38, 33}, {8, 31}, {37, 14}, {48, -11}, {6, 35}, {12, -10}, {24, 18}, {48, -29}, {-9, 24}, {-7, 7}, {47, 30}, {45, 29}, {-28, -21}, {36, 1}, {1, -4}, {-25, 36}, {-16, -26}, {-8, 36}, {26, -6}, {-14, -5}, {-17, 27}, {35, -25}, {24, 45}, {23, 42}, {10, 14}, {38, -13}, {14, 9}, {33, 29}, {32, -8}, {-23, 38}, {36, 26}, {49, -30}, {0, 6}, {-11, 17}, {38, 31}, {-21, 30}, {0, 24}, {40, 42}, {-15, 10}, {38, 9}, {-7, 3}, {21, -21}, {25, 33}, {-16, -19}, {24, -16}, {49, 17}, {21, 0}, {24, -5}, {30, 26}, {24, -2}, {33, 20}, {18, 7}, {17, -28}, {-11, 23}, {-20, 17}, {22, 39}, {42, 0}, {13, 23}, {8, 46}, {33, 39}, {-27, 9}, {5, 10}, {-20, -28}, {-30, 40}, {20, -22}, {7, 19}, {5, -30}, {-1, -20}, {28, 20}, {2, -10}, {16, 12}, {46, -5}, {-10, -20}, {23, -1}, {18, -25}, {28, -30}, {47, -21}, {12, -23}, {1, -18}, {-23, -1}, {9, 13}, {45, 24}, {24, 33}, {10, -26}, {-9, 2}, {22, 0}, {7, 16}, {38, 31}, {37, 32}, {-22, 27}, {16, 20}, {-22, -10}, {46, -1}, {47, 23}, {11, 39}, {-27, 25}, {-28, 28}, {15, -11}, {35, -9}, {44, -18}, {44, 38}, {-23, 27}, {-25, 24}, {-9, 10}, {47, -12}, {14, 35}, {26, 31}, {33, -21}, {10, 13}, {20, -22}, {36, -9}, {-20, 15}, {30, -18}, {23, 34}, {-18, 31}, {-4, 10}, {33, -24}, {-13, 48}, {-6, -6}, {7, -12}, {27, -17}, {-5, 47}, {40, 45}, {-4, 25}, {42, 28}, {5, 11}, {34, -11}, {-24, -9}, {-1, 28}, {-20, 23}, {17, -17}, {-30, 20}, {47, -17}, {45, 24}, {-26, 30}, {-15, 37}, {2, 9}, {42, 39}, {-14, 25}, {48, -8}, {-25, -11}, {42, -29}, {-18, -6}, {-15, 11}, {-17, 45}, {-26, -20}, {36, 48}, {39, 2}, {-3, 19}, {46, -6}, {35, -3}, {42, -11}, {-20, 12}, {2, -28}, {-16, 41}, {-25, 32}, {-23, 31}, {-23, 25}, {48, -19}, {29, -14}, {37, 42}, {46, 26}, {-2, -7}, {34, 46}, {37, -3}, {30, -19}, {-5, 47}, {12, 13}, {17, 45}, {43, 44}, {9, -27}, {40, -7}, {-8, 35}, {44, 20}, {-1, -25}, {22, 3}, {42, -22}, {-21, -25}, {25, -29}, {11, 15}, {29, -11}, {6, 11}, {37, -5}, {-2, 3}, {-7, 39}, {-2, -23}, {6, -10}, {27, -21}, {44, 8}, {27, 46}, {36, 47}, {4, -8}, {-9, -10}, {-11, -19}, {-5, 37}, {-22, 45}, {-18, -30}, {28, -5}, {40, 9}, {8, 32}, {12, -17}, {2, 17}, {31, 20}, {41, 31}, {-4, 7}, {44, 30}, {-24, 14}, {-3, -8}, {-30, 22}, {5, -30}, {-4, -8}, {8, 15}, {16, 13}, {-5, 4}, {13, 35}, {35, 29}, {-15, 32}, {-17, 14}, {28, 6}, {41, 43}, {18, -24}, {-23, -20}, {49, 44}, {-21, -5}, {-29, 33}, {34, 32}, {-26, -13}, {13, -1}, {27, 13}, {-24, -16}, {-4, 47}, {-10, -16}, {17, 0}, {30, 38}, {4, -11}, {3, 25}, {-28, -30}, {9, 31}, {-9, 0}, {21, -7}, {20, 16}, {36, -2}, {16, 31}, {-1, 36}, {-26, 27}, {13, 38}, {28, 37}, {41, 35}, {-28, 7}, {6, -13}, {33, 8}, {48, 31}, {-21, 45}, {5, 17}, {-12, 35}, {4, 42}, {-21, -6}, {36, 3}, {33, -12}, {8, -10}, {-21, -18}, {1, 46}, {-9, -18}, {5, 44}, {34, 14}, {47, -20}, {-7, -8}, {23, 23}, {31, -29}, {36, 41}, {46, -20}, {14, 6}, {17, 8}, {25, -23}, {-14, 41}, {31, 29}, {-16, -1}, {-3, 0}, {19, 49}, {-15, 21}, {-13, 29}, {-14, -23}, {37, -20}, {32, -9}, {7, 41}, {-18, 0}, {47, -12}, {30, -27}, {15, 34}, {11, -9}, {11, -9}, {44, -14}, {46, -10}, {24, 36}, {41, -24}, {37, 20}, {14, 43}, {38, 33}, {-3, -3}, {4, -23}, {13, 26}, {-21, 12}, {29, 18}, {-7, 15}, {42, -1}, {-14, -27}, {4, 21}, {24, 43}, {-19, -22}, {-6, 3}, {33, 24}, {26, 35}, {12, 33}, {-29, 30}, {-23, 3}, {-22, 36}, {39, -4}, {36, 38}, {-15, 31}, {-23, -19}, {13, 23}, {23, 27}, {-28, 12}, {-6, -15}, {-28, 4}, {-5, 3}, {-13, -21}, {23, 25}, {37, -26}, {49, -25}, {16, 37}, {-6, 16}, {30, 39}, {32, 6}, {-14, -14}, {37, 21}, {-18, 35}, {17, -15}, {5, -29}, {-5, 29}, {45, 11}, {36, -19}, {23, -7}, {-28, 46}, {1, 35}, {9, 36}, {-21, -21}, {-19, -11}, {26, 4}, {39, -27}, {26, -12}, {-27, 32}, {39, 27}, {41, 15}, {29, -29}, {33, 5}, {46, -24}};

    // std::complex<double> B[] = {{43, 40}, {7, 12}, {21, 49}, {8, 33}, {11, -2}, {35, 26}, {3, 15}, {38, 26}, {-15, 4}, {3, -18}, {-15, 0}, {34, -15}, {3, -4}, {2, -19}, {-26, 15}, {37, -14}, {25, 28}, {13, 44}, {-18, -19}, {1, 0}, {15, 47}, {-11, -26}, {37, 13}, {25, -18}, {-25, 49}, {32, 44}, {48, 9}, {-13, 22}, {36, 23}, {-9, 48}, {-28, -21}, {29, 37}, {32, 20}, {41, 41}, {13, -11}, {-27, 29}, {24, -11}, {7, -29}, {-23, 25}, {48, 34}, {42, 22}, {22, 28}, {-6, -14}, {4, 14}, {13, 42}, {16, 7}, {-25, 26}, {46, 45}, {25, 24}, {19, 45}, {-5, 1}, {13, -15}, {-21, 24}, {33, 23}, {-11, 28}, {-28, 15}, {-13, 10}, {46, -11}, {22, -29}, {28, 33}, {-26, 0}, {-10, 27}, {38, 2}, {16, 18}, {-20, -2}, {20, 46}, {32, 42}, {-1, 4}, {34, 33}, {-5, 11}, {33, 45}, {14, 13}, {9, -26}, {49, -16}, {13, 21}, {19, -10}, {38, 30}, {47, 1}, {10, 30}, {11, -5}, {-16, 0}, {-30, -11}, {-29, -25}, {5, 14}, {-11, 23}, {29, -9}, {6, 4}, {20, 4}, {37, 16}, {-11, -27}, {-9, 15}, {1, 6}, {21, 30}, {-28, -28}, {-22, -29}, {-9, 34}, {30, 22}, {-25, 26}, {-3, 48}, {6, 7}, {44, 19}, {32, 2}, {44, 42}, {-16, 44}, {47, 2}, {-12, -20}, {11, 42}, {14, 25}, {45, 23}, {-19, -21}, {15, -5}, {12, -3}, {13, 38}, {9, 37}, {-27, 14}, {41, 39}, {46, 11}, {26, -26}, {10, 6}, {9, -20}, {46, 19}, {21, 45}, {33, -4}, {-29, -25}, {19, 1}, {-13, 29}, {49, -3}, {18, 20}, {38, 26}, {-14, 29}, {-28, 23}, {-24, 21}, {-13, -7}, {36, -23}, {41, 48}, {-26, -18}, {11, -20}, {15, 15}, {34, 0}, {22, 24}, {34, -27}, {28, 3}, {-9, 1}, {-14, 17}, {49, 26}, {-7, 44}, {17, 6}, {10, -6}, {20, -27}, {-25, -28}, {16, 49}, {33, 24}, {13, -12}, {12, 12}, {-26, 43}, {32, 20}, {31, 26}, {2, -27}, {-7, 32}, {34, -5}, {-27, -20}, {-10, 24}, {-11, 48}, {-9, -2}, {-23, -27}, {41, 15}, {40, 16}, {49, 35}, {-3, -29}, {35, 10}, {-28, 23}, {-22, 18}, {37, -24}, {-5, -5}, {49, 17}, {11, -23}, {42, 26}, {16, 26}, {8, -17}, {22, 25}, {-23, -29}, {4, -30}, {24, -11}, {-2, -26}, {18, 18}, {-4, 2}, {-11, 27}, {-6, -21}, {-22, -24}, {-15, 45}, {-12, 28}, {-10, -28}, {-23, 11}, {-15, 42}, {33, 26}, {33, -30}, {29, 39}, {-1, 36}, {-23, 8}, {16, 11}, {5, 40}, {31, 27}, {16, 47}, {-6, 23}, {22, -15}, {11, 18}, {-19, -6}, {41, 4}, {-22, 24}, {48, -1}, {41, 26}, {15, 3}, {44, 23}, {1, 15}, {-29, -30}, {-17, 40}, {29, 13}, {36, -17}, {46, 28}, {-16, 46}, {-14, 43}, {0, 19}, {-15, -14}, {-18, 1}, {20, 24}, {21, -29}, {23, -10}, {9, -30}, {47, 45}, {20, -23}, {-19, 10}, {19, 7}, {-29, -23}, {-29, -28}, {-9, -23}, {6, 8}, {22, 14}, {-6, 8}, {-21, -20}, {28, 18}, {0, 35}, {40, 12}, {-28, 22}, {5, -24}, {45, 16}, {15, 27}, {16, 9}, {4, -3}, {-1, 21}, {-28, 35}, {2, 30}, {-14, 12}, {25, 35}, {16, 9}, {1, 49}, {-6, -15}, {5, 17}, {-14, 19}, {23, -30}, {28, 29}, {15, -7}, {-22, 12}, {3, 22}, {-5, 21}, {-8, 40}, {15, -28}, {40, 3}, {46, -29}, {36, 18}, {42, -18}, {-5, -24}, {34, 36}, {-20, 43}, {25, 0}, {25, 3}, {-26, -24}, {-19, -17}, {29, 38}, {15, 41}, {14, 38}, {-25, -21}, {-20, -17}, {32, 1}, {17, -27}, {-2, 49}, {-16, -2}, {-14, -23}, {43, -28}, {29, 40}, {38, 27}, {17, 0}, {44, 19}, {46, 26}, {-16, 47}, {11, -22}, {38, -9}, {35, -12}, {3, 12}, {-13, 32}, {-29, -6}, {32, 10}, {42, 34}, {-18, -3}, {29, 9}, {15, 32}, {1, 11}, {-8, 17}, {20, 20}, {-5, 18}, {27, 33}, {0, 34}, {28, -30}, {-11, 29}, {29, 18}, {-8, 6}, {44, 26}, {-30, 4}, {-16, -29}, {22, -6}, {32, 26}, {-22, 44}, {-13, 25}, {6, -17}, {39, -28}, {-5, -16}, {38, -14}, {38, -8}, {38, 45}, {-12, -2}, {16, -6}, {24, -3}, {-28, 28}, {27, 23}, {42, 22}, {19, 27}, {28, 33}, {23, 19}, {-30, -3}, {34, 17}, {30, 11}, {23, 27}, {-6, 42}, {36, 20}, {14, 16}, {-23, -19}, {-26, 12}, {-22, 12}, {-29, 26}, {3, -20}, {17, -10}, {-4, 21}, {25, 20}, {46, -19}, {40, -1}, {21, -1}, {16, 49}, {6, -4}, {45, 35}, {34, 47}, {26, -11}, {28, 20}, {-27, 48}, {7, -20}, {-15, -22}, {3, -6}, {43, 42}, {42, -30}, {31, -20}, {22, 35}, {-5, 23}, {22, -30}, {-28, -19}, {-15, 9}, {-8, -13}, {42, 16}, {20, 3}, {4, -21}, {-25, -18}, {7, 3}, {-10, -11}, {-13, 42}, {33, 9}, {-11, 9}, {6, 1}, {24, 28}, {37, -19}, {-2, -15}, {-12, 25}, {-17, -20}, {4, 36}, {49, 15}, {-29, 33}, {41, 40}, {35, -3}, {49, 27}, {42, -14}, {-22, 34}, {-20, 28}, {20, 2}, {23, -21}};

    // 10 ---------------------------------------------------------------------------------------------------
    int rowA = 32;
    int colA = 32;
    int rowB = 32;
    int colB = 32;
    std::complex<double> A[10000] = {0, 0};
    std::complex<double> B[10000] = {0, 0};
    for (int i = 0; i < rowA * rowB; i++)
    {
        A[i] = {i, -50 + i};
        B[i] = {-40 + i, 40 + i};
    }

    // // 5 ---------------------------------------------------------------------------------------------------
    // int rowA = 5;
    // int colA = rowA;
    // int rowB = 5;
    // int colB = 5;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, 5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}, {9, -4}, {11, -8}, {13, 5}, {9, 12}, {4, 13}, {-7, -8}, {5, -9}, {-13, -17}, {9, -1}};

    // 5x4 ---------------------------------------------------------------------------------------------------
    // int rowA = 5;
    // int colA = rowA;
    // int rowB = 5;
    // int colB = 4;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, 5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}, {9, -4}, {11, -8}, {13, 5}, {9, 12}};

    // 5x1
    // int rowA = 5;
    // int colA = rowA;
    // int rowB = 5;
    // int colB = 1;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}};
    // // 4 ---------------------------------------------------------------------------------------------------
    // int rowA = 4;
    // int colA = rowA;
    // int rowB = 4;
    // int colB = 4;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, -5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}};

    // // 4 matlab A0---------------------------------------------------------------------------------------------------
    // int rowA = 4;
    // int colA = rowA;
    // int rowB = 4;
    // int colB = 4;
    // std::complex<double> A[] = {{-0.445,5.445}, {4.9, -4.9}, {-0.8787, 0.8787}, {6.3036, -6.3036}, {-6.3941, 6.3941}, {13.3537, -8.3537}, {1.6668, -1.6668}, {11.9448, -11.9448}, {3.68, -3.68}, {-6.66, 6.66}, {-0.06, 5.06}, {-7.0043, 7.0043}, {3.121, -3.121}, {-5.2052, 5.2052}, {-1.413, 1.413}, {-2.848, 7.848}};

    // 3_3 ---------------------------------------------------------------------------------------------------
    // int rowA = 3;
    // int colA = rowA;
    // int rowB = 3;
    // int colB = 3;
    // std::complex<double> A[] = {{5, 9}, {3, 9}, {5, 12}, {1, 7}, {9, 8}, {7, 6}, {5, 4}, {3, 2}, {2, 3}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}, {8, 9}, {6, 7}, {4, -5}, {2, 3}, {1, 2}};

    // Gram-Schmidt
    // int rowA = 3;
    // int colA = rowA;
    // int rowB = 3;
    // int colB = 3;
    // std::complex<double> A[] = {{1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}, {5, 0}, {0, 0}, {3, 0}, {6, 0}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}, {8, 9}, {6, 7}, {4, -5}, {2, 3}, {1, 2}};

    // 3 matlab A0
    //  int rowA = 3;
    //  int colA = rowA;
    //  int rowB = 3;
    //  int colB = 3;
    //  std::complex<double> A[] = {{3, 0}, {2, 0}, {4, 0}, {2, 0}, {0, 0}, {2, 0}, {4, 0}, {2, 0}, {3, 0}};
    //  std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}, {8, 9}, {6, 7}, {4, -5}, {2, 3}, {1, 2}};

    // 3_2 ---------------------------------------------------------------------------------------------------
    // int rowA = 3;
    // int colA = rowA;
    // int rowB = 3;
    // int colB = 2;
    // std::complex<double> A[] = {{5, 9}, {3, 9}, {5, 12}, {1, 7}, {9, 8}, {7, 6}, {5, 4}, {3, 2}, {2, 3}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}, {8, 9}, {6, 7}};

    // 2 ---------------------------------------------------------------------------------------------------
    // int rowA = 2;
    // int colA = rowA;
    // int rowB = 2;
    // int colB = 2;
    // std::complex<double> A[] = {{5, 9}, {3, 9}, {5, 12}, {1, 7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}};
    // ---------------------------------------------------------------------------------------------------
    struct timeval start_Multiply, end_Multiply, diff_Multiply; // estimate matrix multiply time
    std::complex<double> *C = (std::complex<double> *)malloc(rowA * colB * sizeof(std::complex<double>));
    printf(L_BLUE "\n------------mat A--------------------------\n" CLOSE);
    // print_complex_matrix(A, rowA, colA);
    printf(L_BLUE "\n------------mat B--------------------------\n" CLOSE);
    // print_complex_matrix(B, rowB, colB);
    printf(L_BLUE "\n------------mat C--------------------------\n" CLOSE);

    gettimeofday(&start_Multiply, NULL);
    complex_matrix_multiplication(A, B, C, rowA, rowB, colB);
    gettimeofday(&end_Multiply, NULL);
    timersub(&end_Multiply, &start_Multiply, &diff_Multiply);
    // print_complex_matrix(C, rowA, colB);
    printf(L_BLUE "\n--------------------------------------\n" CLOSE);
    printf(L_GREEN "\nElapsed Origin matrix[%d x %d] time: %ld(us)\n" CLOSE, rowA, rowB, (long int)diff_Multiply.tv_usec);

    //-----------------------------------------------------------------------------------------------------------------
    // print_complex_matrix_matlab(C, row, col);
    //  std::complex<double> *Q = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //  std::complex<double> *R = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //  qr(A, Q, R, 3, 3);
    std::complex<double> *Ve = (std::complex<double> *)malloc(rowA * colA * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double> *)malloc(rowA * colA * sizeof(std::complex<double>));
    //  std::complex<double> *A_inv_1=(std::complex<double>*)malloc(row* col * sizeof(std::complex<double>));

    eigen(A, Ve, De, rowA, colA, iter);
    // printf("----------Vector------------\n");
    // print_complex_matrix(Ve, rowA, colA);
    // printf("----------Eigen------------\n");
    // print_complex_matrix(De, rowA, colA);

    // printf("----------Q------------\n");
    // print_complex_matrix(Q, row, col);
    // printf("----------R------------\n");
    // print_complex_matrix(R, row, col);

    // 	for(int i = 0; i < M * M; i += (M + 1)) {
    // 	if(abs(De[i])<0.00000000001) {
    // 		De[i].real(1000000);
    // 		De[i].imag(0);
    // 	}
    // 	else De[i]= std::complex <double> (1)/De[i];
    // }
    // printf("----------De2------------\n");
    // print_complex_matrix(De ,row,col);

    // complex_matrix_multiplication(Ve,De,A_inv_1,M,M,M);

    // free(Ve);
    // free(De);

    return 0;
}
