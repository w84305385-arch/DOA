//--------------------
#define PI acos(-1)
#define AVX 16   
//--------------------
#include <immintrin.h>
#include "math_func.h"
#include "complex_matrix_ops.h"
#include "eigen_qr.h"
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
// ======== BMGS QR, eigen ========
// ================================

void qr(float *A_re, float *A_im, float *Q_re, float *Q_im, float *R_re, float *R_im, int row, int col)
{
    // uesd memory pool
    //float *qr_memory_pool = (float *)aligned_alloc(64, (row * 1 * 6 + 1 * 2 + row * col * 12 ) * sizeof(float));
    float *qr_memory_pool = (float *)malloc((row * 1 * 6 + 1 * 2 + row * col * 12 ) * sizeof(float));
    //float *qr_memory_pool = (float *)malloc((row * 1 * 6 + 1 * 2 + row * col * 6 ) * sizeof(float));
    
    float *Q_col_temp_re = qr_memory_pool;
    float *Q_col_temp_im = Q_col_temp_re + row * 1;
    //--------------------------------------------------------------
    float *Q_col_re = Q_col_temp_im + row * 1;//
    float *Q_col_im = Q_col_re + row * 1;//
    //---------------------------------------------------------------
    float *vector_cur_re = Q_col_im + row * 1;
    float *vector_cur_im = vector_cur_re + row * 1;
    //---------------------------------------------------------------
    float *power_val_re = vector_cur_im + row * 1;//
    float *power_val_im = power_val_re + 1;//
    
    //---------------------------------------------------------------

    for (int i = 0; i < col*row; ++i)
    {
        Q_re[i] = A_re[i];
        Q_im[i] = A_im[i];
    }

    //printf("A:\n");
    //print_complex_matrix(A_re, A_im, row, col );
    //printf(YELLOW"---------\n"CLOSE);
    for (int i = 0; i < col; ++i)
    {
        //printf(YELLOW"-----i=(%d)----\n"CLOSE,i);
        //printf(L_GREEN"i=(%d)------\n" CLOSE ,i);
        //printf("一開始Q:\n");
        //print_complex_matrix(Q_re, Q_im, row, col );
        
        complex_matrix_get_columns(Q_re, Q_im, Q_col_re, Q_col_im, row, col, i);
        
        
        //printf("Q_col歸一前,v(%d)\n",i);
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        
        complex_matrix_conjugate_transpose_multiplication(Q_col_re, Q_col_im, power_val_re, power_val_im, 1, row);
        //printf("power_val開根號前1\n");
        //print_complex_matrix(power_val_re, power_val_im, 1, 1);
       
        //printf("power_val開根號前2\n");
        //print_complex_matrix(power_val_re, power_val_im, 1, 1);

        cpp_sqrt(&power_val_re[0], &power_val_im[0]);
        //printf("power_val開根號後:放到R對角線上\n");
        //print_complex_matrix(power_val_re, power_val_im, 1, 1);
        R_re[i * col + i] = power_val_re[0];
        R_im[i * col + i] = power_val_im[0]; //給R對角線
        
    
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&Q_col_re[m], &Q_col_im[m], &power_val_re[0], &power_val_im[0]); //Q_col=q(i)=v(i)除rii : row*1除長度//i=0:q0，i=1:q1...給後續計算用
            cpp_division(&Q_re[m * col + i], &Q_im[m * col + i], &power_val_re[0], &power_val_im[0]);//只為了存一開始歸一化的v(i)除rii進Q i=0時歸一第0行，i=1時歸一第1行...
        }
        
        //printf("歸一化後(調整過)q,(*a3:%d) \n",a3);
        //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
        complex_matrix_conjugate_transpose( Q_col_re, Q_col_im, row, 1);//q(i) -> q(i)^H : row*1 -> 1*row(為了後續算內積)
        //printf("除完的Q_re,Q_im(%d):\n",i);
        //print_complex_matrix(Q_re, Q_im, row, col);
        //printf(BLUE"---\n"CLOSE);
        float *Q_sub_re = power_val_im + 1;
        float *Q_sub_im = Q_sub_re + row * col;
        float *Q_col_proj_re = Q_sub_im + row * col;
        float *Q_col_proj_im = Q_col_proj_re + row * col;
        float *proj_vector_re = Q_col_proj_im + row * col;
        float *proj_vector_im = proj_vector_re + row * col;
        if(i<col-1)//i=0,1,2進////i=0,1,2,3,4,5,6進
        {
            int base1 = (col-(i+1));
            for (int m = 0; m < row; ++m)
            {
                int j = i+1;
                for (; j < col; ++j)
                {
                    Q_col_proj_re[m * base1 + j-(i+1)] = Q_re[m * col + j]; //i=0: 8*7//i=1: 4*2//i=2: 4*1//i=3: 4*0
                    Q_col_proj_im[m * base1 + j-(i+1)] = Q_im[m * col + j];
                    // printf(RED "[%d] = %.0f, " CLOSE, m * i + j, Q_col_proj_re[m * i + j]);
                }
            }

            //print_complex_matrix(Q_col_proj_re, Q_col_proj_im, row , col-(i+1));
            complex_matrix_multiplication(Q_col_re, Q_col_im, Q_col_proj_re, Q_col_proj_im, proj_vector_re, proj_vector_im, 1, row, col-(i+1)); //1*[col-(i+1)]//rij
            //printf("q*v:r:Q_col x Q_col_proj = proj_vector \n");
            //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1));
            
            for (int16_t j = i+1; j < col; ++j)
            {
                R_re[j + col * i] = proj_vector_re[j-(i+1)];///128; // 給右上 i=0: [0][1][2] i=1: [0][1] i=2: [0]
                R_im[j + col * i] = proj_vector_im[j-(i+1)];///128;
            }
            //print_complex_matrix(R_re, R_im, row, col );
            //*/
            //printf("R給右上:\n");
            //print_complex_matrix(R_re, R_im, row, col );
            complex_matrix_conjugate_transpose( Q_col_re, Q_col_im, 1, row);//q(i)^H -> q(i) : 1*row -> row*1
            //printf("Q_col(%d):\n",i);
            //print_complex_matrix(Q_col_re, Q_col_im, row, 1);
            
            //printf("proj_vector\n");
            //print_complex_matrix(proj_vector_re, proj_vector_im, 1, col-(i+1)); 
            complex_matrix_multiplication( Q_col_re, Q_col_im, proj_vector_re, proj_vector_im, Q_sub_re, Q_sub_im, row, 1, col-(i+1));// row*col-(i+1)
            //printf("Q_sub = Q_col x proj_vector\n");
            //print_complex_matrix(Q_sub_re, Q_sub_im,  row, col-(i+1));
            
            //printf("Q 減前\n");
            //print_complex_matrix(Q_re, Q_im, row, col);
            for(int j=i+1 ; j< col ; j++)//i=0;j = 1,2,3////i=1;j = ,2,3,4,5,6進
            {    
                //printf(RED"進for分別減,i=%d,j=%d\n"CLOSE,i,j);
                complex_matrix_get_columns(Q_re, Q_im, vector_cur_re, vector_cur_im, row, col, j); //i=0、j=1時會取v(1) ; i=0、j=2時會取v(2)... -> row*1
                complex_matrix_get_columns(Q_sub_re, Q_sub_im, Q_col_re, Q_col_im, row, col-(i+1), j-(i+1));
                
                //printf("vector_cur:\n");
                //print_complex_matrix(vector_cur_re, vector_cur_im,  row, 1);
                //printf("Q_col:\n");
                //print_complex_matrix(Q_col_re, Q_col_im,  row, 1);
                complex_matrix_subtraction(vector_cur_re, vector_cur_im, Q_col_re, Q_col_im, row, 1);

                //printf("減完的vector_cur\n");
                //print_complex_matrix(vector_cur_re, vector_cur_im,  row, 1);

                for (int m = 0; m < row; m++)
                {
                    Q_re[m * col + j] = vector_cur_re[m];//*a2_reciprocal;
                    Q_im[m * col + j] = vector_cur_im[m];//*a2_reciprocal;
                }
            } 
            //printf(YELLOW"---------\n"CLOSE);
        }   
    }
    //printf(RED "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
    //printf("Q最終:\n");
    //print_complex_matrix(Q_re, Q_im, row, col );   
    //printf("R最終:\n");
    //print_complex_matrix(R_re, R_im, row, col );  
    //printf("A:\n");
    //print_complex_matrix(A_re, A_im, row, col );
    //float *QxR_re = (float *)aligned_alloc(64, row * col * sizeof(float));
    //float *QxR_im = (float *)aligned_alloc(64, row * col * sizeof(float));
    //printf("Q*R:\n");
    //complex_matrix_multiplication( Q_re, Q_im, R_re, R_im, QxR_re, QxR_im, row, col, col); 
    //print_complex_matrix(QxR_re, QxR_im, row, col );
    free(qr_memory_pool);
}

void BMGS_qr(float *A_re, float *A_im, float *Q_re, float *Q_im, float *R_re, float *R_im, int row, int col, int seg, float *total_QR)
{
    int Block_col = col/seg; // 計算每個區塊的大小，確保整數區塊計數
    *total_QR = 0.0;
    float test_start, test_end;
    float *BMGS_qr_memory_pool = (float *)malloc((Block_col * row * 2  // R_seg_re and R_seg_im
        + row * Block_col * 4  // Q_seg_re, Q_seg_im, temp_re, temp_im
        + row * Block_col * 2  // Q_pre_re, Q_pre_im
        + Block_col * Block_col * (seg + 1) * (seg) // R_temp_re and R_temp_im
        ) * sizeof(float));                                                    
    // 逐一分配記憶體給各個變數
    float *R_seg_re = BMGS_qr_memory_pool;
    float *R_seg_im = R_seg_re + Block_col * Block_col;

    float *Q_seg_re = R_seg_im + Block_col * Block_col;
    float *Q_seg_im = Q_seg_re + row * Block_col;
    float *temp_re = Q_seg_im + row * Block_col;
    float *temp_im = temp_re + row * Block_col;

    float *Q_pre_re = temp_im + row * Block_col;
    float *Q_pre_im = Q_pre_re + row * Block_col;

    // 暫存所有小R
    float *R_temp_re = Q_pre_im + row * Block_col;
    float *R_temp_im = R_temp_re + Block_col * Block_col * (seg + 1) * (seg) / 2;

    

    // 為了提取資料方便，與增加記憶體連續性，所以以下的所有操作都是使用轉置後的A
    // 但會改變原始的A
    matrix_transpose(A_re, A_im, row, col); // 為後面提取block方便
    float *A_block_re; //指向A block的指標
    float *A_block_im;

    for(int k = 0; k < seg; k++) {
        // 每個區塊的大小為 row * p

        // 提取第 k 個區塊，此時的block是橫的
        // A = [Block0
        //      Block1
        //      Block2
        //      Block3]
        A_block_re = A_re + k * row * Block_col;
        A_block_im = A_im + k * row * Block_col;


        // 轉成直的
        // A_block = [ B
        //             l
        //             o
        //             c
        //             k 
        //             0 ]
        matrix_transpose(A_block_re, A_block_im, Block_col, row);

        

        // 正交化過程
        int j = 0;
        for(; j < k; j++) {

            // 提取先前的Q，此時提取的是Q'
            memcpy(Q_pre_re, Q_re + j * row * Block_col, row * Block_col * sizeof(float));
            memcpy(Q_pre_im, Q_im + j * row * Block_col, row * Block_col * sizeof(float));


            // 計算內積 R(j,k) = Q' * W 
            // 這個R也就是R12... 所以要直接存到R_temp中
            complex_matrix_multiplication(Q_pre_re, Q_pre_im, A_block_re, A_block_im, R_temp_re + (k + j) * Block_col * Block_col, R_temp_im + (k + j) * Block_col * Block_col, Block_col, row, Block_col);

            // 存到output的R中
            for(int m = 0; m < Block_col; m++){
                for(int n = 0; n < Block_col; n++){
                    R_re[k * Block_col + j * col * Block_col + col * m + n] = R_temp_re[(k + j) * Block_col * Block_col + m * Block_col + n];
                    R_im[k * Block_col + j * col * Block_col + col*m + n] = R_temp_im[(k + j) * Block_col * Block_col + m * Block_col + n];
                }
            }


            // 要轉回來，為了下一個計算
            complex_matrix_conjugate_transpose(Q_pre_re, Q_pre_im, Block_col, row);

            // 修正 A_block = A_block - Q(:,j) * R(j,k)

            complex_matrix_multiplication(Q_pre_re, Q_pre_im, R_temp_re + (k + j) * Block_col * Block_col, R_temp_im + (k + j) * Block_col * Block_col, temp_re, temp_im, row, Block_col, Block_col);

            complex_matrix_subtraction(A_block_re, A_block_im, temp_re, temp_im, row, Block_col);


        }

        // 重新轉置回來進行 QR 分解
        memset(Q_seg_re, 0, row * Block_col * sizeof(float));
        memset(Q_seg_im, 0, row * Block_col * sizeof(float));
        memset(R_seg_re, 0, Block_col * Block_col * sizeof(float));
        memset(R_seg_im, 0, Block_col * Block_col * sizeof(float));

        test_start = clock();
        qr(A_block_re, A_block_im, Q_seg_re, Q_seg_im, R_seg_re, R_seg_im, row, Block_col);
        test_end = clock();
        *total_QR += test_end - test_start;
        

        // qr後的 Q 跟 R 在後面的計算會用到，但只要存到 Q R 中，就不會變了，但如果用原本的方法提取會困難
        // 存到Q時，先將Q_seg 轉置後存入Q，並且因為同步時，第一個會先使用到共厄轉置的Q，所以這裡用共厄，
        // 記得最後也要共厄回來
        complex_matrix_conjugate_transpose(Q_seg_re, Q_seg_im, row, Block_col);
        memcpy(Q_re + k * row * Block_col, Q_seg_re, row * Block_col * sizeof(float));
        memcpy(Q_im + k * row * Block_col, Q_seg_im, row * Block_col * sizeof(float));


        // R的部分比較複雜，因為R不管怎麼存都不順，又後面計算都會用到之前的R，所以先存到R_temp中
        // R_temp = [R11 R12 R22 R13 R23 ...];
        // R = [R11 R12 R13 R14
        //        0 R22 R23 R24
        //        0   0 R33 R34
        //        0   0   0 R44]
        memcpy(R_temp_re + (k + j) * Block_col * Block_col, R_seg_re, Block_col * Block_col * sizeof(float));
        memcpy(R_temp_im + (k + j) * Block_col * Block_col, R_seg_im, Block_col * Block_col * sizeof(float));

        for(int m=0;m<Block_col;m++){
            for(int n=0;n<Block_col;n++){
                R_re[col*Block_col*k + k*Block_col + col*m+ + n] = R_seg_re[m*Block_col+n];
                R_im[col*Block_col*k + k*Block_col + col*m+ + n] = R_seg_im[m*Block_col+n];
            }
        }


    }

    //最後再處理output的Q
    // 還原Q
    complex_matrix_conjugate_transpose(Q_re, Q_im, row, col);

    
    // 釋放動態記憶體
    free(BMGS_qr_memory_pool);
}

void eigen_upper_triangular(float *A_re, float *A_im, float *eigenvalue_re, float *eigenvalue_im, float *eigenvector_re, float *eigenvector_im, int row, int col)
{

    //---------------------------------------------------------------
    float *vector_cur_re = (float *)malloc(row * 1 * sizeof(float));
    float *vector_cur_im = (float *)malloc(row * 1 * sizeof(float));
    //---------------------------------------------------------------
    float *eigen_element_cur_re = (float *)malloc(sizeof(float));
    float *eigen_element_cur_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *vector_cur_temp_re = (float *)malloc(sizeof(float));
    float *vector_cur_temp_im = (float *)malloc(sizeof(float));
    //---------------------------------------------------------------
    float *A_col_re = (float *)malloc(1 * col * sizeof(float));
    float *A_col_im = (float *)malloc(1 * col * sizeof(float));
    //---------------------------------------------------------------
    float diff_eigen_value_re = 0;
    float diff_eigen_value_im = 0;
    //---------------------------------------------------------------
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            if (i > j)
            {
                A_re[i * col + j] = 0;
                A_im[i * col + j] = 0;
            }
            if (i == j)
            {
                eigenvalue_re[i * col + j] = A_re[i * col + j];
                eigenvalue_im[i * col + j] = A_im[i * col + j];

                eigenvector_re[i * col + j] = 1;
                // printf(PURPLE "eigenvalue[%d] = %.2f\n" CLOSE, i * col + j, eigenvalue[i * col + j]);
            }
        }
    }

    for (int i = 0; i < col; ++i)
    {
        complex_matrix_get_columns(eigenvector_re, eigenvector_im, vector_cur_re, vector_cur_im, row, col, i);


        for (int j = i - 1; j > -1; --j)
        {
            diff_eigen_value_re = eigenvalue_re[i * col + i] - eigenvalue_re[j * col + j];
            diff_eigen_value_im = eigenvalue_im[i * col + i] - eigenvalue_im[j * col + j];
            if (diff_eigen_value_re < 1e-8)
            {
                eigen_element_cur_re[0] = 0;
                eigen_element_cur_im[0] = 0;
            }
            else
            {
                complex_matrix_get_rows(A_re, A_im, A_col_re, A_col_im, col, j);

                complex_matrix_multiplication(A_col_re, A_col_im, vector_cur_re, vector_cur_im, eigen_element_cur_re, eigen_element_cur_im, 1, row, 1);

                cpp_division(&eigen_element_cur_re[0], &eigen_element_cur_im[0], &diff_eigen_value_re, &diff_eigen_value_im);
            }
            vector_cur_re[j] = eigen_element_cur_re[0];
            vector_cur_im[j] = eigen_element_cur_im[0];
        }
        complex_matrix_conjugate_transpose(vector_cur_re, vector_cur_im, row, 1);

        complex_matrix_conjugate_transpose_multiplication(vector_cur_re, vector_cur_im, vector_cur_temp_re, vector_cur_temp_im, 1, row);

        cpp_sqrt(&vector_cur_temp_re[0], &vector_cur_temp_im[0]); // vector_cur_temp[0] = sqrt(vector_cur_temp[0]);
        complex_matrix_conjugate_transpose(vector_cur_re, vector_cur_im, 1, row);

        // Complex Division
        for (int m = 0; m < row; ++m)
        {
            cpp_division(&vector_cur_re[m], &vector_cur_im[m], &vector_cur_temp_re[0], &vector_cur_temp_im[0]);
            eigenvector_re[m * col + i] = vector_cur_re[m];
            eigenvector_im[m * col + i] = vector_cur_im[m];
            // printf(L_BLUE "eigenvector[%d] = %.2f\n" CLOSE, m * col + i, eigenvector_re[m * col + i]);
        }
    }
    
    free(vector_cur_re);
    free(vector_cur_im);
    free(eigen_element_cur_re);
    free(eigen_element_cur_im);
    free(vector_cur_temp_re);
    free(vector_cur_temp_im);
    free(A_col_re);
    free(A_col_im);
}

void eigen_BMGS(float *A_re, float *A_im, float *Ve_re, float *Ve_im, float *De_re, float *De_im, int row, int col, int iter, int seg, float *BMGS_QR_time, float *QR_time)
{
    struct timeval start_QR, end_QR, diff_QR;
    float time_QR = 0.0;
    float *eigen_memory_pool = (float *)malloc( (row * col * 12 ) * sizeof(float));
    memset(eigen_memory_pool, 0, (row * col * 12 ) * sizeof(float));
    
    float *Q_re = eigen_memory_pool;
    float *Q_im = Q_re + row * col;
    //---------------------------------------------------------------
    float *R_re = Q_im + row * col;
    float *R_im = R_re + row * col;
    //---------------------------------------------------------------
    float *Q_temp_re = R_im + row * col;
    float *Q_temp_im = Q_temp_re + row * col;
    //---------------------------------------------------------------
    float *Q_total_re = Q_temp_im + row * col;
    float *Q_total_im = Q_total_re + row * col;
    //---------------------------------------------------------------
    for (int i = 0; i < row * col; i += (col + 1))
    {
        Q_total_re[i] = 1.0;
    }

    gettimeofday(&start_QR, NULL);
    
    for (int i = 0; i < iter; ++i)
    {
        //------------------------------Before QR------------------------------------------
        // printf(YELLOW "\n----------------Before QR-------------------\n" CLOSE);
        //printf("A = \t\n");
        //print_complex_matrix(A_re, A_im, row, col);
        //printf("Q = \t\n");
        //print_complex_matrix(Q_re, Q_im, row, col);
        //printf("R = \t\n");
        //print_complex_matrix(R_re, R_im, row, col);
        //------------------------------------QR--------------------------------------------
        BMGS_qr(A_re, A_im, Q_re, Q_im, R_re, R_im, row, col, seg, QR_time);
        //qr(A_re, A_im, Q_re, Q_im, R_re, R_im, row, col);
        //printf(L_GREEN "Total multiply time : \t%.3f(ms)\n" CLOSE, total_multiply_time / 1000);
        //printf(YELLOW "\n----------------After QR-------------------\n" CLOSE);
        //printf("A = \t\n");
        //print_complex_matrix(A_re, A_im, row, col);
        //printf("Q = \t\n");
        //print_complex_matrix(Q_re, Q_im, row, col);
        //printf("R = \t\n");
        //print_complex_matrix(R_re, R_im, row, col);
        //printf("Q_temp_re = \t\n");
        //print_complex_matrix(Q_temp_re, Q_temp_im, row, col);
        //------------------------------------------------------------------------
        complex_matrix_multiplication(R_re, R_im, Q_re, Q_im, A_re, A_im, row, row, col);   
        //---------------------------------------------------------------
        // Q_total = Q_total * Q
        complex_matrix_multiplication(Q_total_re, Q_total_im, Q_re, Q_im, Q_temp_re, Q_temp_im, row, row, col);
        memcpy(Q_total_re, Q_temp_re, row * col * sizeof(float));
        memcpy(Q_total_im, Q_temp_im, row * col * sizeof(float));
        //---------------------------------------------------------------
    }
    //printf(YELLOW "\n----------------After QR-------------------\n" CLOSE);
    //printf("A = \t\n");
    //print_complex_matrix(A_re, A_im, row, col);
    gettimeofday(&end_QR, NULL);
    timersub(&end_QR, &start_QR, &diff_QR);
    time_QR = diff_QR.tv_sec * 1000000 + diff_QR.tv_usec;
    *BMGS_QR_time = time_QR;
    //printf(CYAN "Elapsed QR :\t\t%.3f(ms), Iteration = %d\n" CLOSE, time_QR / 1000, iter);
    //---------------------------------------------------------------
    float *YY0_re = Q_total_im + row * col;
    float *YY0_im = YY0_re + row * col;
    //---------------------------------------------------------------
    float *XX0_re = YY0_im + row * col;
    float *XX0_im = XX0_re + row * col;
    //---------------------------------------------------------------
    eigen_upper_triangular(A_re, A_im, YY0_re, YY0_im, XX0_re, XX0_im, row, col);
    memcpy(De_re, YY0_re, row * col * sizeof(float));
    memcpy(De_im, YY0_im, row * col * sizeof(float));
    //---------------------------------------------------------------
    memcpy(Ve_re, Q_total_re, row * col * sizeof(float));
    memcpy(Ve_im, Q_total_im, row * col * sizeof(float));
    //---------------------------------------------------------------
    free(eigen_memory_pool);
}

void matrix_inverse_eigen(float *Ve_re, float *Ve_im, float *De_re, float *De_im, float *Pn_re, float *Pn_im, int Rx_M){
        
    //compute_Pn(Pn_re, Pn_im, vet_noise_re, vet_noise_im, M, len_t_theta);
    float *R_xx_inv_1_re = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX*sizeof(float));
    float *R_xx_inv_1_im = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX*sizeof(float));
    float *De_result = (float *)malloc(Rx_M * Rx_M * sizeof(float) + AVX*sizeof(float));
    float temp_real = 1;
    float temp_imag = 0;
    for (int i = 0; i < Rx_M * Rx_M; i += (Rx_M + 1))
    {
        cpp_abs(&De_re[i], &De_im[i], &De_result[i]);
        if (fabs(De_result[i]) < 0.00000001)
        {
            De_re[i] = 1000000;
            De_im[i] = 0;
        }
        else
        {
            cpp_division3(&temp_real, &temp_imag, &De_re[i], &De_im[i]);
        }
    }
    complex_matrix_multiplication(Ve_re, Ve_im, De_re, De_im, R_xx_inv_1_re, R_xx_inv_1_im, Rx_M, Rx_M, Rx_M);
    complex_matrix_conjugate_transpose(Ve_re, Ve_im, Rx_M, Rx_M);
    complex_matrix_multiplication(R_xx_inv_1_re, R_xx_inv_1_im, Ve_re, Ve_im, Pn_re, Pn_im, Rx_M, Rx_M, Rx_M);
    free(R_xx_inv_1_re);
    free(R_xx_inv_1_im);
    free(De_result);
}

