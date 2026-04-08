#ifndef EIGEN_QR_H
#define EIGEN_QR_H

// global variable
extern float total_multiply_time;
extern float total_pre_transpose_time;
extern int multiply_times;

// BMGS QR, eigen
void qr(float *A_re, float *A_im, float *Q_re, float *Q_im, float *R_re, float *R_im, int row, int col);
void BMGS_qr(float *A_re, float *A_im, float *Q_re, float *Q_im, float *R_re, float *R_im, int row, int col, int seg, float *total_QR);
void eigen_upper_triangular(float *A_re, float *A_im, float *eigenvalue_re, float *eigenvalue_im, float *eigenvector_re, float *eigenvector_im, int row, int col);
void eigen_BMGS(float *A_re, float *A_im, float *Ve_re, float *Ve_im, float *De_re, float *De_im, int row, int col, int iter, int seg, float *BMGS_QR_time, float *QR_time);
void matrix_inverse_eigen(float *Ve_re, float *Ve_im, float *De_re, float *De_im, float *Pn_re, float *Pn_im, int Rx_M);

#endif