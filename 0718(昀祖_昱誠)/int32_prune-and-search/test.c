#include <stdio.h>
#include <stdlib.h>
#include "math_func_int.h"
// g++ -o test test.c math_func_int.a
// ./test
int main(void)
{
    int a = 14.55;
    int b = 6.03;
    int c = 25.22;
    int d = 0.0;
    int *p_a = &a;
    int *p_b = &b;
    int *p_c = &c;
    int *p_d = &d;

    //cpp_sqrt(p_a, p_b);
    // printf("verify a = %04f , b = %04f\n", *p_a, *p_b);

    //cpp_division_i(p_a, p_b, p_c, p_d);
    printf("verify a = %d , b = %d\n", *p_a, *p_b);

    return 0;
}