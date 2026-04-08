// 生成127bit碼，用sliding window確認有沒有對上
// g++ -mavx512f -g -o pn_generate  pn_generate.c -Wall -Wextra -std=c++14 
// ./pn_generate
#include <immintrin.h>

#include <complex.h>
#include <assert.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define CODE_LENGTH 127

void print_code(int code[],int a){
    for(int i=0;i<a;i++){
        printf("%d",code[i]);
    }
    printf("\n");
}

// 檢查 pn code 是否與目標序列相同的函式
int checkpnCode(int code[], int target[]) {
    int mul=0;
    //print_code(code,CODE_LENGTH);
    //print_code(target,CODE_LENGTH);
    for (int i = 0; i < CODE_LENGTH; i++) {
        mul=mul+code[i]*target[i];
    }
    //printf("mul=%d ",mul);
    if(mul==CODE_LENGTH){
        //printf("matched\n");
    }
    else{
        //printf("not match\n");
    }

}


int main() {
    float time_code = 0.0;
    float timecode_start, timecode_end; 

    struct timeval time_code_start, time_code_end, time_code_diff;

    //int targetCode[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1 1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 }
    //int targetCode[CODE_LENGTH] = {00100,1010,1100,0011,1001,1011,1110,1000,1001,0101,1000,0111,0011,0111,1101,0001,0010,1011,0000,1110,0110,1111,1010,0010,0101,0110,0001,1100,1101,1111,0100,01}
    //int target_sequence[CODE_LENGTH]={1,-1,1,1,1};
    //int targetCode[CODE_LENGTH]={1,-1,1,1,1}; //答案
    
    int target_sequence[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };
    int targetCode[CODE_LENGTH] = {-1,-1,1,-1,-1 ,1,-1,1,-1, 1,1,-1,-1, -1,-1,1,1, 1,-1,-1,1, 1,-1,1,1, 1,1,1,-1, 1,-1,-1,-1, 1,-1,-1,1, -1,1,-1,1, 1,-1,-1,-1, -1,1,1,1, -1,-1,1,1, -1,1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,-1,1,-1, 1,-1,1,1, -1,-1,-1,-1, 1,1,1,-1, -1,1,1,-1, 1,1,1,1, 1,-1,1,-1, -1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, -1,-1,-1,1, 1,1,-1,-1, 1,1,-1,1, 1,1,1,1, -1,1,-1,-1, -1,1 };

    int temp ;
    //print_code(targetCode,CODE_LENGTH);
    //print_code(target_sequence,CODE_LENGTH);
    //timecode_start = clock();
    //gettimeofday(&time_code_start, NULL);
    checkpnCode(targetCode, target_sequence);
    //gettimeofday(&time_code_end, NULL);
    //timecode_end = clock();
    //--------------------------------------------------------
    //printf("\n");
    gettimeofday(&time_code_start, NULL);
    //timecode_start = clock();
    for(int i=0;i<127;i++){
        //printf("i=%d\n",i);
        ///*
        temp = targetCode[i%127];
        //printf("%d\n",temp);
        for(int j=0;j<CODE_LENGTH-1;j++){
            target_sequence[j]=target_sequence[j+1];
        }
        target_sequence[126]=temp;
        //printf("sequence\n");
        //print_code(target_sequence,CODE_LENGTH);
        //printf("code\n");
        //print_code(targetCode,CODE_LENGTH);
        //gettimeofday(&time_code_start, NULL);
        checkpnCode(targetCode, target_sequence);
        //*/
        //gettimeofday(&time_code_end, NULL);
        //DOA
        //printf("\n");
    }
    //timecode_end = clock();
    gettimeofday(&time_code_end, NULL);
    //printf( "Total code REAL time : \t%.3f(us)\n" , (timecode_end - timecode_start) / CLOCKS_PER_SEC * 1000000);

    //gettimeofday
    timersub(&time_code_end, &time_code_start, &time_code_diff); 
    time_code = time_code_diff.tv_usec;
    printf("Total code time: \t\t%.5f(us)\n", time_code );
    //printf("Total code time: \t\t%.5f(ns)\n", time_code *1000);

    

    return 0;
}
