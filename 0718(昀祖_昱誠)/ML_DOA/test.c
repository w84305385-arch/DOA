#include <stdio.h>
#include <stdlib.h>
#include "math_func.h"
// g++ -o test test.c math_func.a
// ./test
int main(void)
{
    float a = 14.55;
    float b = 6.03;
    float c = 25.22;
    float d = 0.0;
    float *p_a = &a;
    float *p_b = &b;
    float *p_c = &c;
    float *p_d = &d;

    //cpp_sqrt(p_a, p_b);
    // printf("verify a = %04f , b = %04f\n", *p_a, *p_b);

    cpp_division(p_a, p_b, p_c, p_d);
    printf("verify a = %04f , b = %04f\n", *p_a, *p_b);

    return 0;
}