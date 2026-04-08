#include <stdio.h>
#include <stdint.h>

int main()
{   
    int32_t a[5] = {32, 33,  35, 36};
    double b[5] = {12.2, 512.52, 559.6, 40.9, 41.6};

    printf("a[2] : %d\n",a[2]);
    printf("a[2] : %04X\n",a[2]);

    ((int16_t*)(&a[0]))[0] = (int16_t)b[2];
    ((int16_t*)(&a[0]))[1] = (int16_t)b[3];
    printf("b[2] : %04X\n",(int16_t)b[2]);
    printf("a[2] : %08X\n",a[2]);

    printf("b[2] : %d\n",(int16_t)b[2]);
    printf("a[2] : %d\n",a[2]);


    // printf("((int16_t*)(&a[0])) :%p\n",((int16_t*)(&a[0])));
    // printf("(a[0]) :%p\n",(&a[0]));
    // printf("((int16_t*)(&a[0])[1]) :%p\n",((int16_t*)(&a[0]))[1]);
    // printf("(a[0]) :%p\n",(&a[0+1]));

    // printf("(&b[0]) :%p\n",(&b[0]));
    // printf("(&b[0+1]) :%p\n",(&b[0+1]));


    // //printf("((int16_t*)(&a[0]))[1] :%p\n",&((int16_t*)(&a[0]))[1]);
   
    // for (int i = 0; i < 5; i++) {
    //     printf("a[%d]:%04X ; b[%d]:%04X\n" ,i,a[i],i,b[i]);
    // }
    

    return 0;

}