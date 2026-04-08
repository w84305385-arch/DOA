// AVX512
#define AVX 4
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

double matC_Real[100000] = {0.0}; // Real
double matC_Imag[100000] = {0.0}; // Imaginary
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
    double re[100000] = {0.0}; // real
    double im[100000] = {0.0}; // imaginary
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colA; ++j)
        {
            // std::cout << std::fixed << std::setprecision(6) << std::setw(27) << matA[i * colA + j] << " ";
            // printf("\t%f ", matA[i * colA + j].real());
            // printf("+ %fi", matA[i * colA + j].imag());
            re[i * colA + j] = matA[i * colA + j].real();
            im[i * colA + j] = matA[i * colA + j].imag();
            printf("\t%.0f+%.0fi, ", re[i * colA + j], im[i * colA + j]);
            // std::cout << std::setprecision(6) << matA[i * colA + j] << " ";
        }
        std::cout << std::endl;
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

// complex matrix multiplication ,colB = rowA  ; rowB = colA (M=32 ~= 9.4ms)
void complex_matrix_multiplication(std::complex<double> *matA, std::complex<double> *matB, std::complex<double> *matC, int rowA, int rowB, int colB)
{
    __attribute__((aligned(64))) double matA_re[100000] = {0.0}; // re_A
    __attribute__((aligned(64))) double matA_im[100000] = {0.0}; // im_A
    //-------------------------------------------------------------------
    __attribute__((aligned(64))) double matB_re[100000] = {0.0}; // re_B
    __attribute__((aligned(64))) double matB_im[100000] = {0.0}; // im_B
    //-------------------------------------------------------------------
    __attribute__((aligned(64))) double matC_re[100000] = {0.0}; // re_C
    __attribute__((aligned(64))) double matC_im[100000] = {0.0}; // im_C
    int colA = rowB;
    struct timeval start_multiply, end_multiply, diff_multiply;
    //-------------------------------------------------------------------
    int count = 0;
    for (int r = 0; r < rowA * rowB; r++)
    {
        matA_re[r] = matA[r].real();
        matA_im[r] = matA[r].imag();
    }
    for (int col = 0; col < colB; col++) // let matrix B transpose
    {
        for (int row = 0; row < rowB; row++)
        {
            matB_re[col * rowB + row] = matB[row * colB + col].real();
            matB_im[col * rowB + row] = matB[row * colB + col].imag();
        }
    }

    // memcpy(&matA_re[0], &matA[0], rowA * rowB * (sizeof(double) * 2));
    // memcpy(&matB_re[0], &matB[0], rowA * rowB * (sizeof(double) * 2)); // real + imaginary = 16bytes
    memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
    //----------------------------------Information---------------------------------
    printf("-------------------------------------------------\n");
    printf("matA =\t");
    for (int x = 0; x < rowA * rowB; x++) // Because [re,Im],[re2,Im2],so need be shifted 2 step
    {
        if (count % rowB == 0)
        {
            printf("\n\t");
        }
        printf("%.2f + %.2fi\t", matA_re[x], matA_im[x]);
        count++;
    }
    printf("\n-------------------------------------------------\n");
    printf("matB* =\t");

    for (int x = 0; x < rowA * rowB; x++)
    {
        if (count % rowB == 0)
        {
            printf("\n\t");
        }
        printf("%.2f + %.2fi\t", matB_re[x], matB_im[x]);
        count++;
    }
    printf("\n-------------------------------------------------\n");
    //------------------------------------------------------------------------------
    __m256d re_A, re_B, re_C; // simd 256 for matrix real part
    __m256d im_A, im_B, im_C; // simd 256 for matrix Imaginary part
    __m256d simd_add, simd_sub;
    // int count_loop_index = 0;
    // for (int i = 0; i < rowA; i++) // rowA = 32
    // {
    //     printf(" i = %d\n", i);
    //     for (int j = 0; j < colB; j+=4) // colB = 32
    //     {
    //         re_A = _mm256_set1_pd(matA_re[i * colB + j]);
    //         for (int k = 0; k < rowB; k+=4) // rowB = 512
    //         {
    //             for (int l = 0; l < colA; l++)
    //             {
    //                 printf(L_GREEN "\n----------------[%d]-----------------\n" CLOSE, count_loop_index);
    //                 re_A = _mm256_loadu_pd(&matA_re[i * rowB + k]);
    //                 im_A = _mm256_loadu_pd(&matA_im[i * rowB + k]);
    //                 re_B = _mm256_loadu_pd(&matB_re[l * colB + j]);
    //                 im_B = _mm256_loadu_pd(&matB_im[l * colB + j]);
    //                 re_C = _mm256_sub_pd(_mm256_mul_pd(re_A, re_B), _mm256_mul_pd(im_A, im_B)); // Re{ac-bd}
    //                 im_C = _mm256_add_pd(_mm256_mul_pd(re_A, im_B), _mm256_mul_pd(im_A, re_B)); // Im{ad+bc}

    //                 _mm256_storeu_pd(&matC_re[count_loop_index * rowB], re_C); // store Re value
    //                 _mm256_storeu_pd(&matC_im[count_loop_index * rowB], im_C); // store Im value

    //                 //-------------------------printf--------------------------
    //                 printf("_mm256_loadu_pd(&matA_re[%d])\n", i * rowB + k);
    //                 printf("_mm256_loadu_pd(&matA_im[%d])\n", i * rowB + k);
    //                 printf("_mm256_loadu_pd(&matB_re[%d])\n", l * colB + j);
    //                 printf("_mm256_loadu_pd(&matB_im[%d])\n", l * colB + j);
    //                 printf("_mm256_storeu_pd(&matC_re[%d])\n", count_loop_index * rowB);
    //                 printf("_mm256_storeu_pd(&matC_im[%d])\n", count_loop_index * rowB);
    //                 printf("\nmatC_re= ");
    //                 for (int x = 0; x < 20; x++)
    //                 {
    //                     printf("%.0f, ", matC_re[x]);
    //                 }
    //                 printf("\nmatC_im= ");
    //                 for (int x = 0; x < 20; x++)
    //                 {
    //                     printf("%.0f, ", matC_im[x]);
    //                 }
    //                 printf(L_GREEN "\n------------------------------------\n" CLOSE);
    //                 count_loop_index++;
    //             }
    //         }
    //     }
    // }
    //-----------------------------------------------Implemant 2 ------------------------------------------------------------
    // for (int i = 0; i < rowA; ++i) -> for (int j = 0; j < colB; ++j) -> for (int k = 0; k < rowB; ++k)
    printf("matA_re:\n");
    for (int a = 0; a < rowA * colA; a++)
    {
        printf("%.0f, ", matA_re[a]);
    }
    printf("\nmatA_im:\n");
    for (int a = 0; a < rowA * colA; a++)
    {
        printf("%.0f, ", matA_im[a]);
    }
    printf("\n");
    printf("matB_re:\n");
    for (int a = 0; a < rowB * colB; a++)
    {
        printf("%.0f, ", matB_re[a]);
    }
    printf("\nmatB_im:\n");
    for (int a = 0; a < rowB * colB; a++)
    {
        printf("%.0f, ", matB_im[a]);
    }
    printf("\n");
    int l_a = 0;
    int l_b = 0;

    gettimeofday(&start_multiply, NULL);
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            l_a = 0;
            l_b = 0;
            for (int k = 0; k < ((rowB - 1) / AVX) + 1; k++)
            {
                re_A = _mm256_loadu_pd(&matA_re[i * rowB + AVX * k]);
                im_A = _mm256_loadu_pd(&matA_im[i * rowB + AVX * k]);
                re_B = _mm256_loadu_pd(&matB_re[j * rowB + AVX * k]);
                im_B = _mm256_loadu_pd(&matB_im[j * rowB + AVX * k]);
                re_C = _mm256_sub_pd(_mm256_mul_pd(re_A, re_B), _mm256_mul_pd(im_A, im_B)); // Re{ac-bd}
                im_C = _mm256_add_pd(_mm256_mul_pd(re_A, im_B), _mm256_mul_pd(im_A, re_B)); // Im{ad+bc}

                _mm256_storeu_pd(&matC_re[(i * colB * rowB + j * rowB) + AVX * k], re_C); // store Re value
                _mm256_storeu_pd(&matC_im[(i * colB * rowB + j * rowB) + AVX * k], im_C); // store Im value

                printf("\n_mm256_loadu_pd(&matA_re[%d])\n", i * rowB + AVX * k);
                printf("_mm256_loadu_pd(&matA_im[%d])\n", i * rowB + AVX * k);
                printf("_mm256_loadu_pd(&matB_re[%d])\n", j * rowB + AVX * k);
                printf("_mm256_loadu_pd(&matB_im[%d])\n", j * rowB + AVX * k);
                printf("i = %d , j = %d\n", i, j);
                printf("_mm256_storeu_pd(&matC_re[%d])\n", (i * colB * rowB + j * rowB) + AVX * k);
                printf("_mm256_storeu_pd(&matC_im[%d])\n", (i * colB * rowB + j * rowB) + AVX * k);
                printf("matC_re= ");
                for (int x = 0; x < 20; x++)
                {
                    printf("%.0f, ", matC_re[x]);
                }
                printf("\nmatC_im= ");
                for (int x = 0; x < 20; x++)
                {
                    printf("%.0f, ", matC_im[x]);
                }
                if (colA > AVX)
                {
                    l_a = colA % AVX;
                    l_b = colA % AVX;
                }
                printf("\n\n");
            }
            printf(L_GREEN "\n===================================================\n" CLOSE);
        }
    }
    gettimeofday(&end_multiply, NULL);
    timersub(&end_multiply, &start_multiply, &diff_multiply);
    printf(L_PURPLE "\nElapsed time: %ld(us)\n" CLOSE, (long int)diff_multiply.tv_usec);
    //========================================================================================================================
    printf("\nReal= ");
    for (int x = 0; x < 20; x++)
    {
        printf("%.0f, ", matC_re[x]);
    }
    printf("\nImag= ");
    for (int x = 0; x < 20; x++)
    {
        printf("%.0f, ", matC_im[x]);
    }
    printf("\n");
    printf("rowA = %d,rowB = %d,colB = %d\n", rowA, rowB, colB);
    for (int i = 0; i < rowA * colB; i++)
    {
        for (int a = 0; a < rowB; a++)
        {
            // printf("index[%d] = %.0f,   ", i * colB + a, matC_re[i * colB + a]);
            // printf("%.2f + %.2fi  ", matC_re[i * colB + a], matC_im[i * colB + a]);
            matC_Real[i] += matC_re[i * rowB + a];
            printf("matC_Real[%d] = %.0f, ", i, matC_Real[i]);
            matC_Imag[i] += matC_im[i * rowB + a];
        }
        printf(", \n");
    }
    printf("\n");

    // memcpy(&matC[0].real(), &matC_re[0], rowA * rowB * (sizeof(double))); // copy matC_re to matC
    // memcpy(&matC[0].imag(), &matC_im[0], rowA * rowB * (sizeof(double))); // copy matC_im to matC
    count = 0;
    printf("matC =\t");
    // printf("row = %d , col= %d\n", rowA, colB);

    printf("\n");
    for (int i = 0; i < rowA; ++i)
    {
        for (int j = 0; j < colB; ++j)
        {
            // std::cout << std::fixed << std::setprecision(6) << std::setw(27) << matA[i * colA + j] << " ";
            printf("%.0f ", matC_Real[i * colB + j]);
            printf(" + %.0fi\t", matC_Imag[i * colB + j]);
            // std::cout << std::setprecision(6) << matA[i * colA + j] << " ";
        }
        printf("\n");
    }
    printf("-------------------------------------------------\n");
    for (int i = 0; i < rowA * colB; i++)
    {
        matC[i] = {matC_Real[i], matC_Imag[i]};
        // matC[i] = matC_Imag[i];
    }

    //========================================Official version===========================================================
    // memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
    // for (int i = 0; i < rowA; ++i)
    // {
    //     for (int j = 0; j < colB; ++j)
    //     {
    //         for (int k = 0; k < rowB; ++k)
    //         {
    //             matC[i * colB + j] += matA[i * rowB + k] * matB[k * colB + j];
    //         }
    //     }
    // }
    //===================================================================================================================
}

// get complex matrix by column

int main(void)
{
    int M = 4;
    int snr = 40;
    int qr_iter = 10;
    float result[3] = {0};
    int angle = 50;
    int iter = 1000;
    // MUSIC_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
    // MVDR_DOA_1D_CPU_test(M, qr_iter, angle, result, snr);
    //-----------------------------------------------------------------------------------------------------------------

    // int rowA = 10;
    // int colA = 10;
    // int rowB = 10;
    // int colB = 1;
    // std::complex<double> A[100] = {0, 0};
    // std::complex<double> B[100] = {0, 0};
    // for (int i = 0; i < rowA * rowB; i++)
    // {
    //     A[i] = {i, -50 + i};
    //     B[i] = {-40 + i, 40 + i};
    // }

    // // 5 ---------------------------------------------------------------------------------------------------
    int rowA = 5;
    int colA = rowA;
    int rowB = 5;
    int colB = 5;
    std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, 5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}, {9, -4}, {11, -8}, {13, 5}, {9, 12}, {4, 13}, {-7, -8}, {5, -9}, {-13, -17}, {9, -1}};

    // 5 test2
    // int rowA = 1;
    // int colA = 5;
    // int rowB = 5;
    // int colB = 1;
    // std::complex<double> A[] = {{5, 0}, {9, 0}, {2, 0}, {8, 0}, {-13, 0}};
    // std::complex<double> B[] = {{5, 0}, {9, 0}, {2, 0}, {8, 0}, {-13, 0}};

    // 5x4 ---------------------------------------------------------------------------------------------------
    // int rowA = 5;
    // int colA = rowA;
    // int rowB = 5;
    // int colB = 4;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, 5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}, {9, -4}, {11, -8}, {13, 5}, {9, 12}};

    // 5x1 ---------------------------------------------------------------------------------------------------
    // int rowA = 5;
    // int colA = 5;
    // int rowB = 5;
    // int colB = 1;
    // std::complex<double> A[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}};

    //  4*5 ---------------------------------------------------------------------------------------------------
    // int rowA = 4;
    // int colA = 5;
    // int rowB = 5;
    // int colB = 5;
    // std::complex<double> B[] = {{5, 9}, {17, 9}, {5, 12}, {12, 7}, {1, 7}, {9, 8}, {7, 6}, {15, -2}, {5, 4}, {3, 2}, {2, 3}, {7, 5}, {2, 6}, {11, 8}, {10, 3}, {8, -15}, {-4, 9}, {-7, 11}, {-3, 13}, {-12, -9}, {-13, -4}, {-8, -7}, {-9, 5}, {-17, 13}, {15, -7}};
    // std::complex<double> A[] = {{5, 1}, {5, 3}, {4, -3}, {8, 15}, {2, 5}, {8, 9}, {6, 7}, {7, -5}, {4, 5}, {2, 3}, {1, 2}, {15, 1}, {2, -6}, {11, 8}, {10, 3}, {12, -7}, {9, -4}, {11, -8}, {13, 5}, {9, 12}};

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

    // 3_2 ---------------------------------------------------------------------------------------------------
    // int rowA = 3;
    // int colA = rowA;
    // int rowB = 3;
    // int colB = 2;
    // std::complex<double> A[] = {{5, 9}, {3, 9}, {5, 12}, {1, 7}, {9, 8}, {7, 6}, {5, 4}, {3, 2}, {2, 3}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}, {8, 9}, {6, 7}};

    // 3_1 ---------------------------------------------------------------------------------------------------
    // int rowA = 3;
    // int colA = 2;
    // int rowB = 2;
    // int colB = 1;
    // std::complex<double> A[] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {7, 0}, {8, 0}, {9, 0}};
    // std::complex<double> B[] = {{4, 0}, {6, 0}};

    // 2 ---------------------------------------------------------------------------------------------------
    // int rowA = 2;
    // int colA = rowA;
    // int rowB = 2;
    // int colB = 2;
    // std::complex<double> A[] = {{5, 9}, {3, 9}, {5, 12}, {1, 7}};
    // std::complex<double> B[] = {{5, 1}, {5, 3}, {4, -3}, {2, 5}};
    // ---------------------------------------------------------------------------------------------------
    std::complex<double> *C = (std::complex<double> *)malloc(rowA * colB * sizeof(std::complex<double>));
    print_complex_matrix(A, rowA, colA);
    printf("\n");
    print_complex_matrix(B, rowB, colB);
    printf("\n");
    complex_matrix_multiplication(A, B, C, rowA, rowB, colB);
    print_complex_matrix(C, rowA, colB);

    // print_complex_matrix(C, row, col);
    //    print_complex_matrix_matlab(C, row, col);
    //     std::complex<double> *Q = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //     std::complex<double> *R = (std::complex<double> *)malloc(row * col * sizeof(std::complex<double>));
    //     qr(A, Q, R, 3, 3);
    std::complex<double> *Ve = (std::complex<double> *)malloc(rowA * colA * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double> *)malloc(rowA * colA * sizeof(std::complex<double>));
    //   std::complex<double> *A_inv_1=(std::complex<double>*)malloc(row* col * sizeof(std::complex<double>));
    //-----------------------------------------------------------------------------------------------------------------
    // eigen(A, Ve, De, row, col, iter);
    // printf("----------Vector------------\n");
    // print_complex_matrix(Ve, row, col);
    // printf("----------Eigen------------\n");
    // print_complex_matrix(De, row, col);

    // printf("----------Q------------\n");
    // print_complex_matrix(Q, row, col);
    // printf("----------R------------\n");
    // print_complex_matrix(R, row, col);
    //-----------------------------------------------------------------------------------------------------------------
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
