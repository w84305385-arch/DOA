// C++
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <ccomplex>
// C
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
// music
#include "music.h"
// matlabplot
// #include "matplotlib-cpp/matplotlibcpp.h"

#define PI acos(-1)
#define cudaCheck(ans) {cudaAssert((ans), __FILE__, __LINE__);}
#define BLOCK_SIZE 16
#define RADAR_SAMPLE 500
#define PRINT_RESULT
// #define PLOT_RESULT
#define CPU_VERSION
#define GPU_VERSION

// using namespace std::complex_literals;
// namespace plt = matplotlibcpp;

const std::complex<double> I_1(0, 1);

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// warm up gpu for time measurement
// __global__ void warmup() {}


// test cuda kernel
__global__ void warmup() {}


// test cuda function
// extern "C"
// void test_CUDA() {
//     warmup<<<1, 1>>>();
//     cudaDeviceSynchronize();
// }


// // print complex matrix matlab
// void print_complex_matrix_matlab(std::complex<double> *matA, int rowA, int colA) {
//     std::cout << "[";
//     for(int i = 0; i < rowA; ++i) {
//         for(int j = 0; j < colA; ++j) {
//             std::cout << std::setprecision(16) << matA[i * colA + j].real() << "+" << matA[i * colA + j].imag() << "i ";
//         }
//         std::cout << ";" << std::endl;
//     }
//     std::cout << "]" << std::endl;
// }


// // print complex matrix
// void print_complex_matrix(std::complex<double> *matA, int rowA, int colA) {
//     for(int i = 0; i < rowA; ++i) {
//         for(int j = 0; j < colA; ++j) {
//             std::cout << std::fixed << std::setprecision(6) << std::setw(27) << matA[i * colA + j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// // print complex matrix from global memory of GPU
// void print_complex_matrix_from_global(cuDoubleComplex *matA, int rowA, int colA) {
//     std::complex<double> *A = (std::complex<double>*)malloc(rowA * colA * sizeof(std::complex<double>));
//     cudaCheck(cudaMemcpy(A, matA, rowA * colA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
//     print_complex_matrix(A, rowA, colA);
//     free(A);
// }

#ifdef GPU_VERSION
// // generate random number with normal_distribution
// std::complex<double> randn() {
//     std::random_device randomness_device{};
//     std::mt19937 pseudorandom_generator{randomness_device()};
//     auto mean = 0.0;
//     auto std_dev = 1.0;
//     std::normal_distribution<> distribution{mean, std_dev};
//     auto sample = distribution(pseudorandom_generator);
//     return (std::complex<double>)(sample);
// }


// // add white gaussian noise
// void awgn(std::complex<double> *input_signal, std::complex<double> *output_signal, int snr, int row, int col) {
//     std::complex<double> Esym;
//     std::complex<double> No;
//     std::complex<double> noiseSigma;
//     std::complex<double> n;
//     for(int i = 0; i < row * col; i++) {
//         Esym += pow(abs(input_signal[i]), 2) / std::complex<double>(row * col);
//         No = Esym / std::complex<double>(snr);
//         noiseSigma = sqrt(No / std::complex<double>(2));
//         n = noiseSigma * (randn() + randn() * I_1);
//         output_signal[i] = input_signal[i] + n;
//     }
// }


// cuDoubleComplex exponantial
__device__
cuDoubleComplex cuCexp(cuDoubleComplex arg) {
    cuDoubleComplex res;
    double s, c;
    double e = exp(arg.x);
    sincos(arg.y, &s, &c);
    res.x = c * e;
    res.y = s * e;
    return res;
}


// cuDoubleComplex sqrt
__device__
cuDoubleComplex cuCsqrt(cuDoubleComplex x) {
	double radius = cuCabs(x);
	double cosA = x.x / radius;
	cuDoubleComplex out;
	out.x = sqrt(radius * (cosA + 1.0) / 2.0);
	out.y = sqrt(radius * (1.0 - cosA) / 2.0);
	// signbit should be false if x.y is negative
	if (signbit(x.y))
		out.y *= -1.0;
	return out;
}


// // complex matrix multiplication
// void complex_matrix_multiplication(std::complex<double> *matA, std::complex<double> *matB, std::complex<double> *matC, int rowA, int rowB, int colB) {
//     memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
//     for(int i = 0; i < rowA; ++i) {
//         for(int j = 0; j < colB; ++j) {
//             for(int k = 0; k < rowB; ++k) {
//                 matC[i * colB + j] += matA[i * rowB + k] * matB[k * colB + j];
//             }
//         }
//     }
// }


// device function: complex matrix multiplication
__device__
void complex_matrix_multiplication_device(cuDoubleComplex *matA, cuDoubleComplex *matB, cuDoubleComplex *matC, int rowA, int rowB, int colB) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colB; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
            for(int k = 0; k < rowB; ++k) {
                sum = cuCadd(sum, cuCmul(matA[i * rowB + k], matB[k * colB + j]));
            }
            matC[i * colB + j] = sum;
        }
    }
}

__global__ void gpu_caculate_De(cuDoubleComplex *dev_De,int M) 
{
	// int id = threadIdx.x;
	cuDoubleComplex a ;
	a = make_cuDoubleComplex(1, 0);
	for (int i = 0 ; i < M * M; i+=M+1){
		if(cuCabs(dev_De[i])<0.00000000001) {
			dev_De[i] = make_cuDoubleComplex(1000000, 0);
			// dev_De[i].real(1000000);
			// dev_De[i].imag(0);
		}
		else dev_De[i]= cuCdiv(make_cuDoubleComplex(1,0),dev_De[i]);
	}
}


__global__ void gpu_square_matrix_mult(cuDoubleComplex *d_a, cuDoubleComplex *d_b, cuDoubleComplex *d_result, int n) 
{
    __shared__ cuDoubleComplex tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ cuDoubleComplex tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    cuDoubleComplex tmp = make_cuDoubleComplex(0, 0);
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n * n) {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = make_cuDoubleComplex(0, 0);
        }
        else {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n * n) {
            tile_b[threadIdx.y][threadIdx.x] = make_cuDoubleComplex(0, 0);
        }  
        else {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
            tmp = cuCadd(tmp, cuCmul(tile_a[threadIdx.y][k], tile_b[k][threadIdx.x]));
        }
        __syncthreads();
    }
    if(row < n && col < n) {
        d_result[row * n + col] = tmp;
    }
}


// kernel function:
__global__ void gpu_matrix_mult(cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int m, int n, int k) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    if(col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum = cuCadd(sum, cuCmul(a[row * n + i], b[i * k + col]));
        }
        c[row * k + col] = sum;
    }
}


// // complex matrix conjugate transpose
// void complex_matrix_conjugate_transpose(std::complex<double> *matA, int rowA, int colA) {
//     std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
//     memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
//     for(int i = 0; i < rowA; ++i) {
//         for(int j = 0; j < colA; ++j) {
//             matA[j * rowA + i].real(temp[i * colA + j].real());
//             matA[j * rowA + i].imag(-temp[i * colA + j].imag());
//         }
//     }
//     free(temp);
// }


// device function: complex matrix conjugate transpose
__device__
void complex_matrix_conjugate_transpose_device(cuDoubleComplex *matA, int rowA, int colA) {
    cuDoubleComplex *temp = (cuDoubleComplex*)malloc(colA * rowA * sizeof(cuDoubleComplex));
    memcpy(temp, matA, (rowA * colA * sizeof(cuDoubleComplex)));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[j * rowA + i] = make_cuDoubleComplex(cuCreal(temp[i * colA + j]), -1 * cuCimag(temp[i * colA + j]));
        }
    }
    free(temp);
}

__global__ void complex_matrix_conjugate_transpose_device_kernel(cuDoubleComplex *matA, int rowA, int colA) {
    cuDoubleComplex *temp = (cuDoubleComplex*)malloc(colA * rowA * sizeof(cuDoubleComplex));
    memcpy(temp, matA, (rowA * colA * sizeof(cuDoubleComplex)));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[j * rowA + i] = make_cuDoubleComplex(cuCreal(temp[i * colA + j]), -1 * cuCimag(temp[i * colA + j]));
        }
    }
    free(temp);
}


// // complex matrix conjugate transpose and multiplication
// void complex_matrix_conjugate_transpose_multiplication(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
//     std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
//     memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
//     complex_matrix_conjugate_transpose(temp, rowA, colA);
//     complex_matrix_multiplication(matA, temp, matB, rowA, colA, rowA);
//     free(temp);
// }


// kernel function:
__global__ void compute_Pn_kenrel(cuDoubleComplex *vet_noise, cuDoubleComplex *vet_noise_temp, cuDoubleComplex *Pn, cuDoubleComplex *Pn_temp, int m, int n, int k, int target, int len_t_theta) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m) vet_noise_temp[row] = vet_noise[row * (m - len_t_theta) + target];

    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    if(col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum = cuCadd(sum, cuCmul(vet_noise_temp[row * n + i], make_cuDoubleComplex(cuCreal(vet_noise_temp[i * k + col]), -1 * cuCimag(vet_noise_temp[i * k + col]))));
        }
        // Pn_temp[row * k + col] = sum;
        Pn[row * k + col] = cuCadd(Pn[row * k + col], sum);
    }
}


void compute_Pn_enhance(cuDoubleComplex *Pn, cuDoubleComplex *vet_noise, int M, int len_t_theta,
    cuDoubleComplex *Pn_temp,
    cuDoubleComplex *vet_noise_temp,
    cuDoubleComplex *vet_noise_temp_CT) {

    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    for(int i = 0; i < M - len_t_theta; ++i) {
        compute_Pn_kenrel<<<dimGrid, dimBlock>>>(vet_noise, vet_noise_temp, Pn, Pn_temp, M, 1, M, i, len_t_theta);
    }
}


// device function: compute S_MUSIC: matlab code: (S_MUSIC(i)=1/(a_vector'*Pn*a_vector))
__device__
cuDoubleComplex compute_S_MUSIC_device(cuDoubleComplex *a_vector, cuDoubleComplex *Pn, int M) {
    cuDoubleComplex *Pn_a_vector_temp = (cuDoubleComplex*)malloc(M * sizeof(cuDoubleComplex));
    cuDoubleComplex *S_MUSIC_temp = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex));
    complex_matrix_multiplication_device(Pn, a_vector, Pn_a_vector_temp, M, M, 1);
    complex_matrix_conjugate_transpose_device(a_vector, M, 1);
    complex_matrix_multiplication_device(a_vector, Pn_a_vector_temp, S_MUSIC_temp, 1, M, 1);
    cuDoubleComplex S_MUSIC = cuCdiv(make_cuDoubleComplex(1, 0), S_MUSIC_temp[0]);
    free(Pn_a_vector_temp);
    free(S_MUSIC_temp);
    return S_MUSIC;
}

__device__
cuDoubleComplex compute_S_ML_device(cuDoubleComplex *a_vector, cuDoubleComplex *Pn, int M, cuDoubleComplex *dev_Rxx, int i) {

	cuDoubleComplex *a_vector_clone_for_conjugate_transpose = (cuDoubleComplex*)malloc(M * sizeof(cuDoubleComplex));
	memcpy(a_vector_clone_for_conjugate_transpose,a_vector,(M * sizeof(cuDoubleComplex)));
    cuDoubleComplex *ARxx = (cuDoubleComplex*)malloc(M * sizeof(cuDoubleComplex));
	cuDoubleComplex *inv = (cuDoubleComplex*)malloc(1 * 1 *sizeof(cuDoubleComplex));
	
	cuDoubleComplex *ARxx_multi_a_vector = (cuDoubleComplex*)malloc(1 * 1 *sizeof(cuDoubleComplex));
	
	complex_matrix_conjugate_transpose_device(a_vector_clone_for_conjugate_transpose, M, 1);
	complex_matrix_multiplication_device(a_vector_clone_for_conjugate_transpose, dev_Rxx, ARxx, 1, M, M);
    inv[0] = cuCdiv(make_cuDoubleComplex(1, 0), make_cuDoubleComplex(M, 0));
	
	
	complex_matrix_multiplication_device(ARxx, a_vector, ARxx_multi_a_vector, 1, M, 1);
	
	
	cuDoubleComplex S_MUSIC = cuCmul(inv[0],ARxx_multi_a_vector[0]);
	
	free(a_vector_clone_for_conjugate_transpose);
	free(inv);
	free(ARxx);
	free(ARxx_multi_a_vector);
    return S_MUSIC;
}


// kernel function: compute S_MUSIC_dB
__global__
void compute_S_dB_kernel
(double *dr, cuDoubleComplex *S_MUSIC, cuDoubleComplex *Pn, double *S_MUSIC_dB, int M, cuDoubleComplex d, cuDoubleComplex kc) {
    int i = threadIdx.x;
    cuDoubleComplex *a_vector = (cuDoubleComplex*)malloc(M * sizeof(cuDoubleComplex));
    for(int j = 0; j < M; ++j) {
        a_vector[j] = cuCexp(cuCmul(cuCmul(cuCmul(make_cuDoubleComplex(0, 1), kc), d), make_cuDoubleComplex(j * sin(dr[i]), 0)));
    }
    S_MUSIC[i] = compute_S_MUSIC_device(a_vector, Pn, M);
    // compute S_MUSIC_dB
    S_MUSIC_dB[i] = 20 * log10(cuCabs(S_MUSIC[i]));
    free(a_vector);
}

__global__
void compute_S_ML_dB_kernel
(double *dr, cuDoubleComplex *S_MUSIC, cuDoubleComplex *Pn, double *S_MUSIC_dB, int M, cuDoubleComplex d, cuDoubleComplex kc, cuDoubleComplex *dev_Rxx) {
    int i = threadIdx.x;
    cuDoubleComplex *a_vector = (cuDoubleComplex*)malloc(M * sizeof(cuDoubleComplex));
	
	for(int j = 0; j < M; ++j) {
		a_vector[j] = cuCexp(cuCmul(cuCmul(cuCmul(make_cuDoubleComplex(0, 1), kc), d), make_cuDoubleComplex(j * sin(dr[i]), 0)));
	}
	S_MUSIC[i] = compute_S_ML_device(a_vector, Pn, M, dev_Rxx,i);
    // compute S_MUSIC_dB
    // for (int i = 0; i<len_dth; i++) 
	// printf("S_MUSIC[%d] = %d \n",i,S_MUSIC[i]);
	S_MUSIC_dB[i] = 20 * log10(cuCabs(S_MUSIC[i]));
    free(a_vector);
	
}


// globle function for QR decomposition
__global__
void init_QR_kernel(cuDoubleComplex *A, cuDoubleComplex *Q, cuDoubleComplex *vector_cur, int row, int col, int target) {
    int i = threadIdx.x;
    Q[i * col + target] = A[i * col + target];
    vector_cur[i] = A[i * col + target];
}


__global__
void compute_proj_vector_kernel(cuDoubleComplex *Q, cuDoubleComplex *Q_col_proj, cuDoubleComplex *Q_col_proj_CT, cuDoubleComplex *vector_cur, cuDoubleComplex *proj_vector, int m, int n, int k) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col < m && row < n) {
        Q_col_proj[row * m + col] = Q[row * n + col];
        Q_col_proj_CT[col * n + row] = make_cuDoubleComplex(cuCreal(Q[row * n + col]), -1 * cuCimag(Q[row * n + col]));

        __syncthreads();
    }

    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    if(col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum = cuCadd(sum, cuCmul(Q_col_proj_CT[row * n + i], vector_cur[i * k + col]));
        }
        proj_vector[row * k + col] = sum;
    }
}


__global__ void update_QR_kernel(cuDoubleComplex *Q_col_proj, cuDoubleComplex *proj_vector, cuDoubleComplex *Q, cuDoubleComplex *R, cuDoubleComplex * vector_cur, int m, int n, int k) { 
    extern __shared__ cuDoubleComplex sm_Q_col[];
    
    int row = threadIdx.x; 
    int col = blockIdx.x;
    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    if(col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum = cuCadd(sum, cuCmul(Q_col_proj[row * n + i], proj_vector[i * k + col]));
        }
        sm_Q_col[row * k + col] = sum;
        __syncthreads();
    }
    Q[row * m + n] = cuCsub(vector_cur[row], sm_Q_col[row]);
    if(row < n) R[row * m + n] = proj_vector[row];
}


__global__
void get_power_cur_kernel(cuDoubleComplex *Q, cuDoubleComplex *power_cur, int row, int col, int colTarget) {
    extern __shared__ cuDoubleComplex sm_power_cur[];

    int i = threadIdx.x;
    cuDoubleComplex q = Q[i * col + colTarget];
    cuDoubleComplex q_conj = make_cuDoubleComplex(cuCreal(q), -1 * cuCimag(q));

    sm_power_cur[i] = cuCmul(q, q_conj);
    __syncthreads();

    for(int s = row / 2; s > 0; s >>= 1) {
        if(i < s) sm_power_cur[i] = cuCadd(sm_power_cur[i], sm_power_cur[i + s]);
        __syncthreads();
    }
    if(i == 0) {
        power_cur[0] = cuCsqrt(sm_power_cur[0]);
    }
}


__global__
void write_QR_kernel(cuDoubleComplex *Q, cuDoubleComplex *R, cuDoubleComplex *power_cur, int row, int col, int colTarget) {
    extern __shared__ cuDoubleComplex sm_power_val[];
    extern __shared__ cuDoubleComplex sm_proj_val[];

    int i = threadIdx.x;
    cuDoubleComplex q = Q[i * col + colTarget];
    cuDoubleComplex q_conj = make_cuDoubleComplex(cuCreal(q), -1 * cuCimag(q));

    sm_power_val[i] = cuCmul(q, q_conj);
    __syncthreads();

    for(int s = row / 2; s > 0; s >>= 1) {
        if(i < s) sm_power_val[i] = cuCadd(sm_power_val[i], sm_power_val[i + s]);
        __syncthreads();
    }
    if(i == 0) {
        sm_power_val[0] = cuCsqrt(sm_power_val[0]);
    }
    __syncthreads();

    if(cuCreal(sm_power_val[0]) / cuCreal(power_cur[0]) < 1e-4) {
        // span again
        if(i == 0) R[colTarget * row + colTarget] = make_cuDoubleComplex(0, 0);
        q = make_cuDoubleComplex(0, 0);
        if(i == colTarget) q = make_cuDoubleComplex(1, 0);
        cuDoubleComplex vector_cur = q;
        
        for(int jj = 0; jj < colTarget; ++jj) {
            cuDoubleComplex Qvector_cur = Q[i * col + jj];
            cuDoubleComplex Qvector_cur_conj = make_cuDoubleComplex(cuCreal(Qvector_cur), -1 * cuCimag(Qvector_cur));
            sm_proj_val[i] = cuCmul(Qvector_cur_conj, vector_cur);
            __syncthreads();
            for(int s = row / 2; s > 0; s >>= 1) {
                if(i < s) sm_proj_val[i] = cuCadd(sm_proj_val[i], sm_proj_val[i + s]);
                __syncthreads();
            }
            q = cuCsub(q, cuCmul(Qvector_cur, sm_proj_val[0]));
        }

        q_conj = make_cuDoubleComplex(cuCreal(q), -1 * cuCimag(q));
        sm_power_val[i] = cuCmul(q, q_conj);
        __syncthreads();
        for(int s = row / 2; s > 0; s >>= 1) {
            if(i < s) sm_power_val[i] = cuCadd(sm_power_val[i], sm_power_val[i + s]);
            __syncthreads();
        }
        if(i == 0) {
            sm_power_val[0] = cuCsqrt(sm_power_val[0]);
        }
        __syncthreads();
        Q[i * col + colTarget] = cuCdiv(q, sm_power_val[0]);
    } else {
        if(i == 0) R[colTarget * row + colTarget] = sm_power_val[0];
        Q[i * col + colTarget] = cuCdiv(q, sm_power_val[0]);
    }
}


// enhance for qr function
void qr_GPU(cuDoubleComplex *A, cuDoubleComplex *Q, cuDoubleComplex *R, cuDoubleComplex *power_cur, int row, int col, cuDoubleComplex *vector_cur, cuDoubleComplex *Q_col_proj, cuDoubleComplex *Q_col_proj_CT, cuDoubleComplex *proj_vector) { 
    for(int i = 0; i < col; ++i) {
        init_QR_kernel<<<1, row>>>(A, Q, vector_cur, row, col, i);
        get_power_cur_kernel<<<1, row, row * sizeof(cuDoubleComplex)>>>(Q, power_cur, row, col, i);
        if(i > 0) {
            unsigned int grid_rows = (row + BLOCK_SIZE - 1) / BLOCK_SIZE;
            unsigned int grid_cols = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 dimGrid(grid_cols, grid_rows);
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            compute_proj_vector_kernel<<<dimGrid, dimBlock>>>(Q, Q_col_proj, Q_col_proj_CT, vector_cur, proj_vector, i, row, 1); 

            update_QR_kernel<<<1, row, row * sizeof(cuDoubleComplex)>>>(Q_col_proj, proj_vector, Q, R, vector_cur, row, i, 1);
        }
        write_QR_kernel<<<1, row, row * sizeof(cuDoubleComplex)>>>(Q, R, power_cur, row, col, i);
    }
}


// global function for upper_triangular enhance
__global__
void initial_eigen_kernel(cuDoubleComplex *A, cuDoubleComplex *eigenvalue, cuDoubleComplex *eigenvector, int row, int col) {
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < row && j < col) {
        if(i > j) A[i * col + j] = make_cuDoubleComplex(0, 0);
        else if(i == j) {
            eigenvalue[i * col + j] = A[i * col + j];
            eigenvector[i * col + j] = make_cuDoubleComplex(1, 0);
        }
    }
}


__global__
void compute_eigne(cuDoubleComplex *A, cuDoubleComplex *eigenvector, cuDoubleComplex *eigenvalue, cuDoubleComplex *vector_cur, cuDoubleComplex *vector_cur_temp, int row, int col, int ii) {
    extern __shared__ cuDoubleComplex sm_vector_cur[];
    extern __shared__ cuDoubleComplex sm_vector_cur_temp[];
    extern __shared__ cuDoubleComplex sm_A_col[];
    extern __shared__ cuDoubleComplex sm_eigen_element_cur[];

    int i = threadIdx.x;
    // int j = blockIdx.x;
    cuDoubleComplex diff_eigenvalue;

    sm_vector_cur[i] = eigenvector[i * col + ii];
    __syncthreads();

    for(int jj = ii - 1; jj > -1; --jj) {
        diff_eigenvalue = cuCsub(eigenvalue[ii * col + ii], eigenvalue[jj * col + jj]);

        if(cuCreal(diff_eigenvalue) < 1e-8) {
            // eigen_element_cur = make_cuDoubleComplex(0, 0);
            if(i == 0) sm_vector_cur[jj] = make_cuDoubleComplex(0, 0);
            __syncthreads();
        }
        else {
            sm_A_col[i] = A[jj * col + i];
            __syncthreads();
            sm_eigen_element_cur[i] = cuCmul(sm_A_col[i], sm_vector_cur[i]);
            __syncthreads();

            for(int s = row / 2; s > 0; s >>= 1) {
                if(i < s) sm_eigen_element_cur[i] = cuCadd(sm_eigen_element_cur[i], sm_eigen_element_cur[i + s]);
                __syncthreads();
            }
            if(i == 0) sm_vector_cur[jj] = cuCdiv(sm_eigen_element_cur[0], diff_eigenvalue);
            __syncthreads();
        }
        // if(i == 0) sm_vector_cur[jj] = make_cuDoubleComplex(0, 0);
        // __syncthreads();
    }
    vector_cur[i] = sm_vector_cur[i];
    sm_vector_cur_temp[i] = cuCmul(make_cuDoubleComplex(cuCreal(sm_vector_cur[i]), -1 * cuCimag(sm_vector_cur[i])), sm_vector_cur[i]);
    __syncthreads();

    for(int s = row / 2; s > 0; s >>= 1) {
        if(i < s) sm_vector_cur_temp[i] = cuCadd(sm_vector_cur_temp[i], sm_vector_cur_temp[i + s]);
        __syncthreads();
    }
    if(i == 0) vector_cur_temp[0] = cuCsqrt(sm_vector_cur_temp[0]);
    __syncthreads();

    eigenvector[i * col + ii] = cuCdiv(vector_cur[i], vector_cur_temp[0]);
}


void eigen_upper_triangular_enhance(cuDoubleComplex *A, cuDoubleComplex *eigenvalue, cuDoubleComplex *eigenvector, int row, int col, cuDoubleComplex *vector_cur, cuDoubleComplex *vector_cur_temp) {

    for(int i = 0; i < col; ++i) {
        compute_eigne<<<1, row, row * sizeof(cuDoubleComplex)>>>(A, eigenvector, eigenvalue, vector_cur, vector_cur_temp, row, col, i);
    }
}


// enhance for cuda eigen funtion
void eigen_enhance(cuDoubleComplex *A, cuDoubleComplex *Ve, cuDoubleComplex *De, int row, int col, int iter, 
    cuDoubleComplex *Q,
    cuDoubleComplex *R,
    cuDoubleComplex *power_cur,
    cuDoubleComplex *Q_temp,
    cuDoubleComplex *Q_temp_clone,
    cuDoubleComplex *YY0,
    cuDoubleComplex *XX0,
    cuDoubleComplex *vector_cur,
    cuDoubleComplex *vector_cur_temp,
    cuDoubleComplex *Q_col_proj,
    cuDoubleComplex *Q_col_proj_CT,
    cuDoubleComplex *proj_vector) {

    unsigned int grid_rows = (row + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (col + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int i;
    for(i = 0; i < iter; ++i) {
        qr_GPU(A, Q, R, power_cur, row, col, vector_cur, Q_col_proj, Q_col_proj_CT, proj_vector);

        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(R, Q, A, row);
        if((i & 1) == 0) gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(Q_temp, Q, Q_temp_clone, row);
        else gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(Q_temp_clone, Q, Q_temp, row);
    }

    initial_eigen_kernel<<<dimGrid, dimBlock>>>(A, YY0, XX0, row, col);
    eigen_upper_triangular_enhance(A, YY0, XX0, row, col, vector_cur, vector_cur_temp);
    cudaMemcpy(De, YY0, row * col * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice); // we don't use the eigenvalue
    if((i & 1) == 0) gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(Q_temp, XX0, Ve, row);
    else gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(Q_temp_clone, XX0, Ve, row);
}


__global__
void get_vet_noise_from_Ve_Kernel(cuDoubleComplex *vet_noise, cuDoubleComplex *Ve, int M, int len_t_theta) {
    // int i = blockIdx.x;
    // int j = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M && j < M - len_t_theta) vet_noise[i * (M - len_t_theta) + j] = Ve[i * M + j + len_t_theta];
}
#endif


#ifdef CPU_VERSION
// generate random number with normal_distribution
std::complex<double> randn() {
    std::random_device randomness_device{};
    std::mt19937 pseudorandom_generator{randomness_device()};
    auto mean = 0.0;
    auto std_dev = 1.0;
    std::normal_distribution<> distribution{mean, std_dev};
    auto sample = distribution(pseudorandom_generator);
    return (std::complex<double>)(sample);
}

// add white gaussian noise
void awgn(std::complex<double> *input_signal, std::complex<double> *output_signal, int snr, int row, int col) {
    std::complex<double> Esym;
    std::complex<double> No;
    std::complex<double> noiseSigma;
    std::complex<double> n;
    for(int i = 0; i < row * col; i++) {
        Esym += pow(abs(input_signal[i]), 2) / std::complex<double>(row * col);
        No = Esym / std::complex<double>(snr);
        noiseSigma = sqrt(No / std::complex<double>(2));
        n = noiseSigma * (randn() + randn() * I_1);
        output_signal[i] = input_signal[i] + n;
    }
}


// complex matrix addition
void complex_matrix_addition(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
        }
    }
}


// complex matrix subtraction
void complex_matrix_subtraction(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[i * colA + j].real(matA[i * colA + j].real() - matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() - matB[i * colA + j].imag());
        }
    }
}


// complex matrix multiplication
void complex_matrix_multiplication(std::complex<double> *matA, std::complex<double> *matB, std::complex<double> *matC, int rowA, int rowB, int colB) {
    memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colB; ++j) {
            for(int k = 0; k < rowB; ++k) {
                matC[i * colB + j] += matA[i * rowB + k] * matB[k * colB + j];
            }
        }
    }
}


// get complex matrix by column
void complex_matrix_get_columns(std::complex<double> *matA, std::complex<double> *matCol, int rowA, int colA, int colTarget) {
    for(int i = 0; i < rowA; ++i) {
        matCol[i] = matA[i * colA + colTarget];
    }
}


// get complex matrix by row
void complex_matrix_get_rows(std::complex<double> *matA, std::complex<double> *matRow, int rowA, int colA, int rowTarget) {
    for(int i = 0; i < colA; ++i) {
        matRow[i] = matA[rowTarget * colA + i];
    }
}


// complex matrix conjugate transpose
void complex_matrix_conjugate_transpose(std::complex<double> *matA, int rowA, int colA) {
    std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[j * rowA + i].real(temp[i * colA + j].real());
            matA[j * rowA + i].imag(-temp[i * colA + j].imag());
        }
    }
    free(temp);
}


// complex matrix conjugate transpose and multiplication
void complex_matrix_conjugate_transpose_multiplication(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    complex_matrix_conjugate_transpose(temp, rowA, colA);
    complex_matrix_multiplication(matA, temp, matB, rowA, colA, rowA);
    free(temp);
}


// compute Pn: matlab code: (Pn=Pn+vet_noise(:,ii)*vet_noise(:,ii)';), where (ii=1:length(vet_noise(1,:)))
void compute_Pn(std::complex<double> *Pn, std::complex<double> *vet_noise, int M, int len_t_theta) {
    std::complex<double> *vet_noise_temp = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *Pn_temp = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    for(int i = 0; i < M - len_t_theta; ++i) {
        complex_matrix_get_columns(vet_noise, vet_noise_temp, M, M - len_t_theta, i);
        complex_matrix_conjugate_transpose_multiplication(vet_noise_temp, Pn_temp, M, 1);
        complex_matrix_addition(Pn, Pn_temp, M, M);
    }
    free(vet_noise_temp);
    free(Pn_temp);
}


// compute S_MUSIC: matlab code: (S_MUSIC(i)=1/(a_vector'*Pn*a_vector))
std::complex<double> compute_S_MUSIC(std::complex<double> *a_vector, std::complex<double> *Pn, int M) {
    std::complex<double> *Pn_a_vector_temp = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC_temp = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    complex_matrix_multiplication(Pn, a_vector, Pn_a_vector_temp, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector, M, 1);
    complex_matrix_multiplication(a_vector, Pn_a_vector_temp, S_MUSIC_temp, 1, M, 1);
    std::complex<double> S_MUSIC = std::complex<double>(1) / S_MUSIC_temp[0];
    free(Pn_a_vector_temp);
    free(S_MUSIC_temp);
    return S_MUSIC;
}


// QR decomposer for c code
void qr(std::complex<double> *A, std::complex<double> *Q, std::complex<double> *R, int row, int col) {
    std::complex<double> *Q_col = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *vector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *Qvector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *power_cur = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *power_val = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_val = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_Qvector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    for(int i = 0; i < row * col; i += (col + 1)) {
        Q[i].real(1);
        R[i].real(1);
    }
    for(int i = 0; i < col; ++i) {
        for(int m = 0; m < row; ++m) {
            Q[m * col + i] = A[m * col + i];
        }
        complex_matrix_get_columns(Q, Q_col, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col, row, 1);
        memset(power_cur, 0, sizeof(std::complex<double>));
        complex_matrix_conjugate_transpose_multiplication(Q_col, power_cur, 1, row);
        power_cur[0] = sqrt(power_cur[0]);
        if(i > 0) {
            complex_matrix_get_columns(A, vector_cur, row, col, i);
            std::complex<double> *Q_col_proj = (std::complex<double>*)malloc(row * i * sizeof(std::complex<double>));
            std::complex<double> *proj_vector = (std::complex<double>*)malloc(i * sizeof(std::complex<double>));
            memset(proj_vector, 0, i * sizeof(std::complex<double>));
            for(int j = 0; j < i; ++j) {
                for(int m = 0; m < row; ++m) {
                    Q_col_proj[m * i + j] = Q[m * col + j];
                }
            }
            complex_matrix_conjugate_transpose(Q_col_proj, row, i);
            complex_matrix_multiplication(Q_col_proj, vector_cur, proj_vector, i, row, 1);
            complex_matrix_conjugate_transpose(Q_col_proj, i, row);
            memset(Q_col, 0, row * 1 * sizeof(std::complex<double>));
            complex_matrix_multiplication(Q_col_proj, proj_vector, Q_col, row, i, 1);
            complex_matrix_subtraction(vector_cur, Q_col, row, 1);
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] = vector_cur[m];
            }
            for(int j = 0; j < i; ++j) {
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

        if(power_val[0].real() / power_cur[0].real() < 1e-4) {
            R[i * row + i] = 0;
            // span again
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] = 0;
            }
            Q[i * row + i].real(1);
            complex_matrix_get_columns(Q, vector_cur, row, col, i);
            for(int j = 0; j < i; ++j) {
                complex_matrix_get_columns(Q, Qvector_cur, row, col, j);
                memset(proj_val, 0, sizeof(std::complex<double>));
                complex_matrix_conjugate_transpose(Qvector_cur, row, 1);
                complex_matrix_multiplication(Qvector_cur, vector_cur, proj_val, 1, row, 1);
                complex_matrix_conjugate_transpose(Qvector_cur, 1, row);
                complex_matrix_get_columns(Q, Q_col, row, col, i);
                memset(proj_Qvector_cur, 0, row * 1 * sizeof(std::complex<double>));
                complex_matrix_multiplication(Qvector_cur, proj_val, proj_Qvector_cur, row, 1, 1);
                complex_matrix_subtraction(Q_col, proj_Qvector_cur, row, 1);
                for(int m = 0; m < row; ++m) {
                    Q[m * col + i] = Q_col[m];
                }
            }
            complex_matrix_get_columns(Q, Q_col, row, col, i);
            complex_matrix_conjugate_transpose(Q_col, row, 1);
            memset(power_val, 0, sizeof(std::complex<double>));
            complex_matrix_conjugate_transpose_multiplication(Q_col, power_val, 1, row);
            power_val[0] = sqrt(power_val[0]);
            complex_matrix_conjugate_transpose(Q_col, 1, row);
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] /= power_val[0];
            }
        } else {
            R[i * row + i] = power_val[0];
            for(int m = 0; m < row; ++m) {
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
void eigen_upper_triangular(std::complex<double> *A, std::complex<double> *eigenvalue, std::complex<double> *eigenvector, int row, int col) {
    std::complex<double> *vector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *eigen_element_cur = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *vector_cur_temp = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *A_col = (std::complex<double>*)malloc(1 * col * sizeof(std::complex<double>));
    std::complex<double> diff_eigen_value = 0;
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            if(i == j) {
                eigenvalue[i * col + j] = A[i * col + j];
                eigenvector[i * col + j].real(1);
            }
        }
    }
    for(int i = 0; i < col; ++i) {
        complex_matrix_get_columns(eigenvector, vector_cur, row, col, i);
        for(int j = i - 1; j > -1; --j) {
            diff_eigen_value = eigenvalue[i * col + i] - eigenvalue[j * col + j];
            if(diff_eigen_value.real() < 1e-8) eigen_element_cur[0] = 0;
            else {
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
        for(int m = 0; m < row; ++m) {
            eigenvector[m * col + i] = vector_cur[m] / vector_cur_temp[0];
        }
    }
    free(vector_cur);
    free(eigen_element_cur);
    free(vector_cur_temp);
    free(A_col);
}


// compute complex eigenvector and eigenvalue for c code
void eigen(std::complex<double> *A, std::complex<double> *Ve, std::complex<double> *De, int row, int col, int iter) {
    std::complex<double> *Q = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *R = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp_clone = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    for(int i = 0; i < row * col; i += (col + 1)) {
        Q_temp[i].real(1);
    }
    for(int i = 0; i < iter; ++i) {
        qr(A, Q, R, row, col);
        complex_matrix_multiplication(R, Q, A, row, row, col);
        complex_matrix_multiplication(Q_temp, Q, Q_temp_clone, row, row, col);
        memcpy(Q_temp, Q_temp_clone, row * col * sizeof(std::complex<double>));
    }
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            if(i > j) A[i * col + j] = 0;
        }
    }
    std::complex<double> *YY0 = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *XX0 = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
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
#endif


// compute the ML DOA in one dimension on GPU
void ML_DOA_1D_GPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
    // printf("Multiple input size:\t%d\n", multi_input);
    #endif
// generate the signal
    // time 
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// ml algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *eye = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
    for(int i = 0; i < M * M; i += (M + 1)) {
        eye[i] = std::complex<double>(1);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // CUDA
	
    cuDoubleComplex *dev_Rxx;
    cuDoubleComplex *dev_Pn;

    // cuda memory allocate
    cudaCheck(cudaMalloc((void**)&dev_Rxx, M * M * sizeof(cuDoubleComplex)));

    cudaCheck(cudaMalloc((void**)&dev_Pn, M * M * sizeof(cuDoubleComplex)));


    // warmup
    warmup<<<20, 1000>>>();
    warmup<<<20, 1000>>>();

    // timestamp start
    timeStart = clock();
    cudaCheck(cudaMemcpy(dev_Rxx, R_xx, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // eigen_kernel<<<1, 1>>>(dev_Rxx, dev_Ve, dev_De, M, M, qr_iter);



    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "ML (gpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;
    cudaDeviceSynchronize();

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    double *S_ML_dB = (double*)malloc(len_dth * sizeof(double));
    // CUDA 
    // initial
    double *dev_dr;
    cuDoubleComplex *dev_S_ML;
    double *dev_S_ML_dB;
    cudaCheck(cudaMalloc((void**)&dev_dr, len_dth * sizeof(double)));
    cudaCheck(cudaMalloc((void**)&dev_S_ML, len_dth * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_S_ML_dB, len_dth * sizeof(double)));
    // memory copy host to device
    cudaCheck(cudaMemcpy(dev_dr, dr, len_dth * sizeof(double), cudaMemcpyHostToDevice));
    // compute S_ML_dB on CUDA
    cuDoubleComplex dev_d = make_cuDoubleComplex(d.real(), 0);
    cuDoubleComplex dev_kc = make_cuDoubleComplex(kc.real(), 0);
    // cuda kernel dimension
    dim3 dimGridKernel_S_ML_dB(1, 1, 1);
    dim3 dimBlockKernel_S_ML_dB(len_dth, 1, 1);
    // timestamp start
    timeStart = clock();    // Since GPU do the ML
    // call kernel function
    compute_S_ML_dB_kernel
    <<<dimGridKernel_S_ML_dB, dimBlockKernel_S_ML_dB>>>
    (dev_dr, dev_S_ML, dev_Pn, dev_S_ML_dB, M, dev_d, dev_kc, dev_Rxx);
    cudaDeviceSynchronize();
    // memory copy device to host
    cudaCheck(cudaMemcpy(S_ML_dB, dev_S_ML_dB, len_dth * sizeof(double), cudaMemcpyDeviceToHost));
    // find Max and position
    double max_temp = S_ML_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_ML_dB[i] > max_temp) {
            max_temp = S_ML_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (gpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);



// free memory
    free(t_theta);
    free(A_theta);
    free(t_sig);
    free(sig_co);
    free(x_r);
    free(R_xx);
    free(Ve);
    free(dth);
    free(dr);
    free(S_ML_dB);
// free cuda memory
    cudaFree(dev_Rxx);//
    cudaFree(dev_Pn);

}


// compute the ML DOA in one dimension on CPU
void ML_DOA_1D_CPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// ml algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // compute eigenvector Ve (M, M)
	std::complex<double> *R_xx_inv_1 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
	std::complex<double> *R_xx_inv_2 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    
   
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "ML (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_ML_dB
    std::complex<double> *a_vector = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_ML = (std::complex<double>*)malloc(len_dth * sizeof(std::complex<double>));
    double *S_ML_dB = (double*)malloc(len_dth * sizeof(double));
	std::complex<double> *a_vector_conjugate_transpose_1 = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
	std::complex<double> *ARxx = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
	std::complex<double> *ARxx_multi_a_vector = (std::complex<double>*)malloc(1 * 1 * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    for(int i = 0; i < len_dth; ++i) { // can be paralleled to compute S_ML_dB
        for(int j = 0; j < M; ++j) {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
		//copy a_vector to a_vector_conjugate_transpose
		memcpy(a_vector_conjugate_transpose_1,a_vector,M * sizeof(std::complex<double>)); 
		// complex_matrix_conjugate_transpose for a_vector_conjugate_transpose
		complex_matrix_conjugate_transpose(a_vector_conjugate_transpose_1, 1, M);
		
		complex_matrix_multiplication(a_vector_conjugate_transpose_1, R_xx, ARxx, 1, M, M);
		complex_matrix_multiplication(ARxx, a_vector, ARxx_multi_a_vector, 1, M, 1);
		
		S_ML[i] = ARxx_multi_a_vector[0];
        // compute S_ML_dB
        S_ML_dB[i] = 20 * log10(abs(S_ML[i]));
    }
    // find Max and position
    double max_temp = S_ML_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_ML_dB[i] > max_temp) {
            max_temp = S_ML_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

// plot the result
    #ifdef PLOT_RESULT
    std::vector<double> S_ML_dB_vec(S_ML_dB, S_ML_dB + len_dth);
    std::vector<double> dth_vec(dth, dth + len_dth);
    plt::plot(dth_vec, S_ML_dB_vec, "blue");
    plt::title("ML DOA Estimation");
    plt::xlabel("Theta (degree)");
    plt::ylabel("Power Spectrum (dB)");
    plt::xlim(dth[0], dth[len_dth - 1]);
    plt::grid(true);
    plt::show();
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
    free(dth);
    free(dr);
    free(a_vector);
    free(S_ML);
    free(S_ML_dB);
}


// compute the MVDR DOA in one dimension on GPU
void MVDR_DOA_1D_GPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
    // printf("Multiple input size:\t%d\n", multi_input);
    #endif
// generate the signal
    // time 
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// mvdr algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *eye = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
    for(int i = 0; i < M * M; i += (M + 1)) {
        eye[i] = std::complex<double>(1);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // CUDA
    cuDoubleComplex *dev_Rxx;
    cuDoubleComplex *dev_Ve;
    cuDoubleComplex *dev_De;
    cuDoubleComplex *dev_Q;
    cuDoubleComplex *dev_R;
    cuDoubleComplex *dev_power_cur;
    cuDoubleComplex *dev_Q_temp;
    cuDoubleComplex *dev_Q_temp_clone;
    cuDoubleComplex *dev_YY0;
    cuDoubleComplex *dev_XX0;
    cuDoubleComplex *dev_vector_cur;
    cuDoubleComplex *dev_vector_cur_temp;
    cuDoubleComplex *dev_Q_col_proj;
    cuDoubleComplex *dev_Q_col_proj_CT;
    cuDoubleComplex *dev_proj_vector;
    cuDoubleComplex *dev_vet_noise;
    cuDoubleComplex *dev_vet_noise_temp;
    cuDoubleComplex *dev_vet_noise_temp_CT;
    cuDoubleComplex *dev_Pn;
    cuDoubleComplex *dev_Pn_temp;
	cuDoubleComplex *dev_multi_temp;
    // cuda memory allocate
    cudaCheck(cudaMalloc((void**)&dev_Rxx, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Ve, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_De, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_R, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_power_cur, sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_temp, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_temp_clone, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_YY0, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_XX0, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vector_cur, M * 1 * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vector_cur_temp, 1 * 1 * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_col_proj, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_col_proj_CT, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_proj_vector, 1 * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise, M * (M - len_t_theta) * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise_temp, M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise_temp_CT, M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Pn, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Pn_temp, M * M * sizeof(cuDoubleComplex)));
	cudaCheck(cudaMalloc((void**)&dev_multi_temp, M * M * sizeof(cuDoubleComplex)));

    // warmup
    warmup<<<20, 1000>>>();
    warmup<<<20, 1000>>>();
    cudaCheck(cudaMemcpy(dev_Q_temp, eye, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_Q, dev_Q_temp, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(dev_R, dev_Q_temp, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	
	std::complex<double> *R_xx_inv_1 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
	// std::complex<double> *R_xx_inv_2 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
	std::complex<double> *De = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    cudaCheck(cudaMemcpy(dev_Rxx, R_xx, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // eigen_kernel<<<1, 1>>>(dev_Rxx, dev_Ve, dev_De, M, M, qr_iter);
    eigen_enhance(dev_Rxx, dev_Ve, dev_De, M, M, qr_iter,
        dev_Q,
        dev_R,
        dev_power_cur,
        dev_Q_temp,
        dev_Q_temp_clone,
        dev_YY0,
        dev_XX0,
        dev_vector_cur,
        dev_vector_cur_temp,
        dev_Q_col_proj,
        dev_Q_col_proj_CT,
        dev_proj_vector);
    // cudaCheck(cudaMemcpy(Ve, dev_Ve, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
	// cudaCheck(cudaMemcpy(De, dev_De, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    // get vet_noise (M, M - len_t_theta): part of Ve (eigenvector)
    // cuda kernel dimension
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = ((M - len_t_theta) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	
	gpu_caculate_De
	<<<1,M>>>
	(dev_De,M);
	
	gpu_square_matrix_mult
	<<<dimGrid, dimBlock>>>
	(dev_Ve, dev_De, dev_multi_temp, M);

	complex_matrix_conjugate_transpose_device_kernel
	<<<1,1>>>
	(dev_Ve, M, M);

	gpu_square_matrix_mult
	<<<dimGrid,dimBlock>>>
	(dev_multi_temp, dev_Ve, dev_Pn, M);
	
	
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MVDR (gpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;
    cudaDeviceSynchronize();

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    double *S_MVDR_dB = (double*)malloc(len_dth * sizeof(double));
    // CUDA 
    // initial
    double *dev_dr;
    cuDoubleComplex *dev_S_MVDR;
    double *dev_S_MVDR_dB;
    cudaCheck(cudaMalloc((void**)&dev_dr, len_dth * sizeof(double)));
    cudaCheck(cudaMalloc((void**)&dev_S_MVDR, len_dth * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_S_MVDR_dB, len_dth * sizeof(double)));
    // memory copy host to device
    cudaCheck(cudaMemcpy(dev_dr, dr, len_dth * sizeof(double), cudaMemcpyHostToDevice));
    // compute S_MVDR_dB on CUDA
    cuDoubleComplex dev_d = make_cuDoubleComplex(d.real(), 0);
    cuDoubleComplex dev_kc = make_cuDoubleComplex(kc.real(), 0);
    // cuda kernel dimension
    dim3 dimGridKernel_S_MVDR_dB(1, 1, 1);
    dim3 dimBlockKernel_S_MVDR_dB(len_dth, 1, 1);
    // timestamp start
    timeStart = clock();    // Since GPU do the MVDR, Pn is already in global memory
    // call kernel function
    compute_S_dB_kernel
    <<<dimGridKernel_S_MVDR_dB, dimBlockKernel_S_MVDR_dB>>>
    (dev_dr, dev_S_MVDR, dev_Pn, dev_S_MVDR_dB, M, dev_d, dev_kc);
    cudaDeviceSynchronize();
    // memory copy device to host
    cudaCheck(cudaMemcpy(S_MVDR_dB, dev_S_MVDR_dB, len_dth * sizeof(double), cudaMemcpyDeviceToHost));
    // find Max and position
    double max_temp = S_MVDR_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_MVDR_dB[i] > max_temp) {
            max_temp = S_MVDR_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (gpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

// plot the result
    // #ifdef PLOT_RESULT
    // std::vector<double> S_MVDR_dB_vec(S_MVDR_dB, S_MVDR_dB + len_dth);
    // std::vector<double> dth_vec(dth, dth + len_dth);
    // plt::plot(dth_vec, S_MVDR_dB_vec, "blue");
    // plt::title("MVDR DOA Estimation");
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
    free(dth);
    free(dr);
    free(S_MVDR_dB);
// free cuda memory
    cudaFree(dev_Rxx);//
    cudaFree(dev_Ve);//
    cudaFree(dev_De);//
    cudaFree(dev_Q);//
    cudaFree(dev_R);//
    cudaFree(dev_power_cur);//
    cudaFree(dev_Q_temp);//
    cudaFree(dev_Q_temp_clone);//
    cudaFree(dev_YY0);//
    cudaFree(dev_XX0);//
    cudaFree(dev_vector_cur);//
    cudaFree(dev_vector_cur_temp);//
    cudaFree(dev_Q_col_proj);//
    cudaFree(dev_Q_col_proj_CT);//
    cudaFree(dev_proj_vector);//
    cudaFree(dev_dr);
    cudaFree(dev_S_MVDR);
    cudaFree(dev_S_MVDR_dB);
    cudaFree(dev_vet_noise);
    cudaFree(dev_vet_noise_temp);
    cudaFree(dev_vet_noise_temp_CT);
    cudaFree(dev_Pn);
    cudaFree(dev_Pn_temp);
	// cudaFree(dev_multi_temp);
}


// compute the MVDR DOA in one dimension on CPU
void MVDR_DOA_1D_CPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// mvdr algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    eigen(R_xx, Ve, De, M, M, qr_iter);
	std::complex<double> *R_xx_inv_1 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
	std::complex<double> *Pn = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
	for(int i = 0; i < M * M; i += (M + 1)) {
		if(abs(De[i])<0.00000000001) {
			De[i].real(1000000);
			De[i].imag(0);
		}
		else De[i]= std::complex <double> (1)/De[i];
	}
	
	complex_matrix_multiplication(Ve, De, R_xx_inv_1, M, M, M);
	complex_matrix_conjugate_transpose(Ve, M, M);
	complex_matrix_multiplication(R_xx_inv_1, Ve, Pn, M, M, M);

    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MVDR (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_MVDR_dB
    std::complex<double> *a_vector = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MVDR = (std::complex<double>*)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MVDR_dB = (double*)malloc(len_dth * sizeof(double));
    // timestamp start
    timeStart = clock();
    for(int i = 0; i < len_dth; ++i) { // can be paralleled to compute S_MVDR_dB
        for(int j = 0; j < M; ++j) {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
        S_MVDR[i] = compute_S_MUSIC(a_vector, Pn, M);
        // compute S_MVDR_dB
        S_MVDR_dB[i] = 20 * log10(abs(S_MVDR[i]));
    }
    // find Max and position
    double max_temp = S_MVDR_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_MVDR_dB[i] > max_temp) {
            max_temp = S_MVDR_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

// plot the result
    #ifdef PLOT_RESULT
    std::vector<double> S_MVDR_dB_vec(S_MVDR_dB, S_MVDR_dB + len_dth);
    std::vector<double> dth_vec(dth, dth + len_dth);
    plt::plot(dth_vec, S_MVDR_dB_vec, "blue");
    plt::title("MUSIC DOA Estimation");
    plt::xlabel("Theta (degree)");
    plt::ylabel("Power Spectrum (dB)");
    plt::xlim(dth[0], dth[len_dth - 1]);
    plt::grid(true);
    plt::show();
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
    // free(vet_noise);
    free(Pn);
    free(dth);
    free(dr);
    free(a_vector);
    free(S_MVDR);
    free(S_MVDR_dB);
}



// compute the MUSIC DOA in one dimension on GPU
void MUSIC_DOA_1D_GPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
    // printf("Multiple input size:\t%d\n", multi_input);
    #endif
// generate the signal
    // time 
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// music algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *eye = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
    for(int i = 0; i < M * M; i += (M + 1)) {
        eye[i] = std::complex<double>(1);
    }
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // CUDA
    cuDoubleComplex *dev_Rxx;
    cuDoubleComplex *dev_Ve;
    cuDoubleComplex *dev_De;
    cuDoubleComplex *dev_Q;
    cuDoubleComplex *dev_R;
    cuDoubleComplex *dev_power_cur;
    cuDoubleComplex *dev_Q_temp;
    cuDoubleComplex *dev_Q_temp_clone;
    cuDoubleComplex *dev_YY0;
    cuDoubleComplex *dev_XX0;
    cuDoubleComplex *dev_vector_cur;
    cuDoubleComplex *dev_vector_cur_temp;
    cuDoubleComplex *dev_Q_col_proj;
    cuDoubleComplex *dev_Q_col_proj_CT;
    cuDoubleComplex *dev_proj_vector;
    cuDoubleComplex *dev_vet_noise;
    cuDoubleComplex *dev_vet_noise_temp;
    cuDoubleComplex *dev_vet_noise_temp_CT;
    cuDoubleComplex *dev_Pn;
    cuDoubleComplex *dev_Pn_temp;
    // cuda memory allocate
    cudaCheck(cudaMalloc((void**)&dev_Rxx, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Ve, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_De, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_R, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_power_cur, sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_temp, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_temp_clone, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_YY0, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_XX0, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vector_cur, M * 1 * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vector_cur_temp, 1 * 1 * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_col_proj, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Q_col_proj_CT, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_proj_vector, 1 * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise, M * (M - len_t_theta) * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise_temp, M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_vet_noise_temp_CT, M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Pn, M * M * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_Pn_temp, M * M * sizeof(cuDoubleComplex)));

    // warmup
    warmup<<<20, 1000>>>();
    warmup<<<20, 1000>>>();
    cudaCheck(cudaMemcpy(dev_Q_temp, eye, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_Q, dev_Q_temp, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(dev_R, dev_Q_temp, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // timestamp start
    timeStart = clock();
    cudaCheck(cudaMemcpy(dev_Rxx, R_xx, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // eigen_kernel<<<1, 1>>>(dev_Rxx, dev_Ve, dev_De, M, M, qr_iter);
    eigen_enhance(dev_Rxx, dev_Ve, dev_De, M, M, qr_iter,
        dev_Q,
        dev_R,
        dev_power_cur,
        dev_Q_temp,
        dev_Q_temp_clone,
        dev_YY0,
        dev_XX0,
        dev_vector_cur,
        dev_vector_cur_temp,
        dev_Q_col_proj,
        dev_Q_col_proj_CT,
        dev_proj_vector);
    cudaCheck(cudaMemcpy(Ve, dev_Ve, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    // get vet_noise (M, M - len_t_theta): part of Ve (eigenvector)
    // cuda kernel dimension
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = ((M - len_t_theta) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    get_vet_noise_from_Ve_Kernel<<<dimGrid, dimBlock>>>(dev_vet_noise, dev_Ve, M, len_t_theta);
    // compute Pn matrix (M, M)
    compute_Pn_enhance(dev_Pn, dev_vet_noise, M, len_t_theta,
        dev_Pn_temp,
        dev_vet_noise_temp,
        dev_vet_noise_temp_CT);
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MUSIC (gpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;
    cudaDeviceSynchronize();

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    double *S_MUSIC_dB = (double*)malloc(len_dth * sizeof(double));
    // CUDA 
    // initial
    double *dev_dr;
    cuDoubleComplex *dev_S_MUSIC;
    double *dev_S_MUSIC_dB;
    cudaCheck(cudaMalloc((void**)&dev_dr, len_dth * sizeof(double)));
    cudaCheck(cudaMalloc((void**)&dev_S_MUSIC, len_dth * sizeof(cuDoubleComplex)));
    cudaCheck(cudaMalloc((void**)&dev_S_MUSIC_dB, len_dth * sizeof(double)));
    // memory copy host to device
    cudaCheck(cudaMemcpy(dev_dr, dr, len_dth * sizeof(double), cudaMemcpyHostToDevice));
    // compute S_MUSIC_dB on CUDA
    cuDoubleComplex dev_d = make_cuDoubleComplex(d.real(), 0);
    cuDoubleComplex dev_kc = make_cuDoubleComplex(kc.real(), 0);
    // cuda kernel dimension
    dim3 dimGridKernel_S_MUSIC_dB(1, 1, 1);
    dim3 dimBlockKernel_S_MUSIC_dB(len_dth, 1, 1);
    // timestamp start
    timeStart = clock();    // Since GPU do the MUSIC, Pn is already in global memory
    // call kernel function
    compute_S_dB_kernel
    <<<dimGridKernel_S_MUSIC_dB, dimBlockKernel_S_MUSIC_dB>>>
    (dev_dr, dev_S_MUSIC, dev_Pn, dev_S_MUSIC_dB, M, dev_d, dev_kc);
    cudaDeviceSynchronize();
    // memory copy device to host
    cudaCheck(cudaMemcpy(S_MUSIC_dB, dev_S_MUSIC_dB, len_dth * sizeof(double), cudaMemcpyDeviceToHost));
    // find Max and position
    double max_temp = S_MUSIC_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_MUSIC_dB[i] > max_temp) {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (gpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

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
    free(dth);
    free(dr);
    free(S_MUSIC_dB);
// free cuda memory
    cudaFree(dev_Rxx);//
    cudaFree(dev_Ve);//
    cudaFree(dev_De);//
    cudaFree(dev_Q);//
    cudaFree(dev_R);//
    cudaFree(dev_power_cur);//
    cudaFree(dev_Q_temp);//
    cudaFree(dev_Q_temp_clone);//
    cudaFree(dev_YY0);//
    cudaFree(dev_XX0);//
    cudaFree(dev_vector_cur);//
    cudaFree(dev_vector_cur_temp);//
    cudaFree(dev_Q_col_proj);//
    cudaFree(dev_Q_col_proj_CT);//
    cudaFree(dev_proj_vector);//
    cudaFree(dev_dr);
    cudaFree(dev_S_MUSIC);
    cudaFree(dev_S_MUSIC_dB);
    cudaFree(dev_vet_noise);
    cudaFree(dev_vet_noise_temp);
    cudaFree(dev_vet_noise_temp_CT);
    cudaFree(dev_Pn);
    cudaFree(dev_Pn_temp);
}


// compute the MUSIC DOA in one dimension on CPU
void MUSIC_DOA_1D_CPU(int M, int qr_iter, int angle, float *result, double *double_IQ) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", global_music_SNR);
    printf("QR iteration:\t\t%d\n", qr_iter);
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
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(angle);
    // t_theta[1].real(12);
    // t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(I_1 * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    memcpy(t_sig, double_IQ, 2 * nd * sizeof(double));
    // for(int i = 0; i < len_t_theta; ++i) {
    //     for(int j = 0; j < nd; ++j) {
    //         t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
    //         // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
    //     }
    // }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, global_music_SNR, M, nd);

// music algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    eigen(R_xx, Ve, De, M, M, qr_iter);
    // get vet_noise (M, M - len_t_theta): part of Ve (eigenvector)
    std::complex<double> *vet_noise = (std::complex<double>*)malloc(M * (M - len_t_theta) * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = len_t_theta; j < M; ++j) {
            vet_noise[i * (M - len_t_theta) + j - len_t_theta] = Ve[i * M + j];
        }
    }
    // Pn matrix (M, M)
    std::complex<double> *Pn = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
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
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_MUSIC_dB
    std::complex<double> *a_vector = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC = (std::complex<double>*)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MUSIC_dB = (double*)malloc(len_dth * sizeof(double));
    // timestamp start
    timeStart = clock();
    for(int i = 0; i < len_dth; ++i) { // can be paralleled to compute S_MUSIC_dB
        for(int j = 0; j < M; ++j) {
            a_vector[j] = exp(I_1 * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
        S_MUSIC[i] = compute_S_MUSIC(a_vector, Pn, M);
        // compute S_MUSIC_dB
        S_MUSIC_dB[i] = 20 * log10(abs(S_MUSIC[i]));
    }
    // find Max and position
    double max_temp = S_MUSIC_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_MUSIC_dB[i] > max_temp) {
            max_temp = S_MUSIC_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

// plot the result
    #ifdef PLOT_RESULT
    std::vector<double> S_MUSIC_dB_vec(S_MUSIC_dB, S_MUSIC_dB + len_dth);
    std::vector<double> dth_vec(dth, dth + len_dth);
    plt::plot(dth_vec, S_MUSIC_dB_vec, "blue");
    plt::title("MUSIC DOA Estimation");
    plt::xlabel("Theta (degree)");
    plt::ylabel("Power Spectrum (dB)");
    plt::xlim(dth[0], dth[len_dth - 1]);
    plt::grid(true);
    plt::show();
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


void print_sample(double *IQ) {
    for(int i = 0; i < 10; ++i) {
        std::cout << "[" << i << "] " << IQ[i * 2] << " " << IQ[i * 2 + 1] << std::endl;
    }
}


void convert_int16_to_double(int16_t *IQ, double *double_IQ) {
    int index = 2 * RADAR_SAMPLE;
    std::copy(IQ, IQ + RADAR_SAMPLE, double_IQ);

    // print_sample(double_IQ);
}


// music initail
extern "C"
void music_init(int subframe, uint32_t *rx_data, float *result) {
    // printf("subframe[%d]: %u",subframe, rx_data[0]);
    int16_t *IQ = (int16_t*)malloc(2 * RADAR_SAMPLE * sizeof(int16_t));
    memcpy(IQ, (int16_t*)rx_data, RADAR_SAMPLE * sizeof(int32_t));

    double *double_IQ = (double*)malloc(2 * RADAR_SAMPLE * sizeof(double));
    convert_int16_to_double(IQ, double_IQ);

// for environment setting
	int  iDeviceCount = 1;
	cudaDeviceProp sDeviceProp;
	cudaGetDeviceCount(&iDeviceCount);
    // set device
	int device = 0;
	cudaSetDevice(device);
    cudaGetDeviceProperties(&sDeviceProp, device);
    printf("\n--Environment setting--\n");
    printf("GPU:\t\t\t%s\n\n", sDeviceProp.name);

    // music_start(double_IQ);
    int M = global_music_antenna;
    int qr_iter = global_music_QR_iteration;
    int angle = global_music_angle;

    if(global_music_type == 0) {
        MUSIC_DOA_1D_CPU(M, qr_iter, angle, result, double_IQ);
    } else if (global_music_type == 1) {
        MUSIC_DOA_1D_GPU(M, qr_iter, angle, result, double_IQ);
    }
    
    free(IQ);
    free(double_IQ);
}


extern "C"
void mvdr_init(int subframe, uint32_t *rx_data, float *result) {
    // printf("subframe[%d]: %u",subframe, rx_data[0]);
    int16_t *IQ = (int16_t*)malloc(2 * RADAR_SAMPLE * sizeof(int16_t));
    memcpy(IQ, (int16_t*)rx_data, RADAR_SAMPLE * sizeof(int32_t));

    double *double_IQ = (double*)malloc(2 * RADAR_SAMPLE * sizeof(double));
    convert_int16_to_double(IQ, double_IQ);

// for environment setting
	int  iDeviceCount = 1;
	cudaDeviceProp sDeviceProp;
	cudaGetDeviceCount(&iDeviceCount);
    // set device
	int device = 0;
	cudaSetDevice(device);
    cudaGetDeviceProperties(&sDeviceProp, device);
    printf("\n--Environment setting--\n");
    printf("GPU:\t\t\t%s\n\n", sDeviceProp.name);

    // music_start(double_IQ);
    int M = global_music_antenna;
    int qr_iter = global_music_QR_iteration;
    int angle = global_music_angle;

    if(global_music_type == 0) {
        MVDR_DOA_1D_CPU(M, qr_iter, angle, result, double_IQ);
    } else if (global_music_type == 1) {
        MVDR_DOA_1D_GPU(M, qr_iter, angle, result, double_IQ);
    }
    
    free(IQ);
    free(double_IQ);
}

extern "C"
void ml_init(int subframe, uint32_t *rx_data, float *result) {
    // printf("subframe[%d]: %u",subframe, rx_data[0]);
    int16_t *IQ = (int16_t*)malloc(2 * RADAR_SAMPLE * sizeof(int16_t));
    memcpy(IQ, (int16_t*)rx_data, RADAR_SAMPLE * sizeof(int32_t));

    double *double_IQ = (double*)malloc(2 * RADAR_SAMPLE * sizeof(double));
    convert_int16_to_double(IQ, double_IQ);

// for environment setting
	int  iDeviceCount = 1;
	cudaDeviceProp sDeviceProp;
	cudaGetDeviceCount(&iDeviceCount);
    // set device
	int device = 0;
	cudaSetDevice(device);
    cudaGetDeviceProperties(&sDeviceProp, device);
    printf("\n--Environment setting--\n");
    printf("GPU:\t\t\t%s\n\n", sDeviceProp.name);

    // music_start(double_IQ);
    int M = global_music_antenna;
    int qr_iter = global_music_QR_iteration;
    int angle = global_music_angle;

    if(global_music_type == 0) {
        ML_DOA_1D_CPU(M, qr_iter, angle, result, double_IQ);
    } else if (global_music_type == 1) {
        ML_DOA_1D_GPU(M, qr_iter, angle, result, double_IQ);
    }
    
    free(IQ);
    free(double_IQ);
}





