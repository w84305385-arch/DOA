#include <stdio.h>
#include "immintrin.h"

void print(char *name, float *a, int num)

{

int i;

printf("%s =%6.1f",name,a[0]);

for (i = 1; i < num; i++)

printf(",%s%4.1f",(i&3)?"":" ",a[i]);

printf("\n");

}

int main(int argc, char *argv[]) {

float a[] = { 9.9,-1.2, 3.3,4.1, -1.1,0.2, -1.3,4.4, 2.4,3.1, -1.3,6.0, 1.5,2.4, 3.1,4.2 };
float b[] = { 0.3, 7.5, 3.2,2.4, 7.2,7.2, 0.6,3.4, 4.1,3.4, 6.5,0.7, 4.0,3.1, 2.4,1.3 };
float c[] = { 0.1, 0.2, 0.3,0.4, 1.0,1.0, 1.0,1.0, 2.0,2.0, 2.0,2.0, 3.0,3.0, 3.0,3.0 };
float o[] = { 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };

__m512 simd1, simd2, simd3, simd4;

__mmask16 m16z = 0;
__mmask16 m16s = 0xAAAA;
__mmask16 m16a = 0xFFFF;

print(" a[]",a,16);
print(" b[]",b,16);
print(" c[]",c,16);



simd1 = _mm512_load_ps(a);
simd2 = _mm512_load_ps(b);
simd3 = _mm512_load_ps(c);

simd4 = _mm512_add_ps(simd1, simd2);
_mm512_store_ps(o, simd4);
print(" a+b",o,16);

simd4 = _mm512_sub_ps(simd1, simd2);
_mm512_store_ps(o, simd4);
print(" a-b",o,16);

simd4 = _mm512_mul_ps(simd1, simd2);
_mm512_store_ps(o, simd4);
print(" a*b",o,16);

simd4 = _mm512_div_ps(simd1, simd2);
print(" a/b",(float *)&simd4,16);

printf("FMAs with mask 0, then mask 0xAAAA, ");
printf("then mask 0xFFFF\n");

simd4 = _mm512_maskz_fmadd_ps(m16z,simd1,simd2,simd2);
print("a*b+b",(float *)&simd4,16);

simd4 = _mm512_maskz_fmadd_ps(m16s,simd1,simd2,simd3);
print("a*b+b",(float *)&simd4,16);

simd4 = _mm512_maskz_fmadd_ps(m16a,simd1,simd2, simd3);
print("a*b+b",(float *)&simd4,16);



return 0;
}