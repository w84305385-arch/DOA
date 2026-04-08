
g++ -mavx512f -mavx512bw -O2 -o cu_avx512 cu_function.cpp -Wall -Wextra -std=c++14
./cu_avx512