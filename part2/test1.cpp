#include <iostream>

#include "fasttime.h"
#include "test.h"

void test1(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
    __builtin_assume(N == 1024);
    a = (float*)__builtin_assume_aligned(a, 4096);
    b = (float*)__builtin_assume_aligned(b, 4096);
    c = (float*)__builtin_assume_aligned(c, 4096);

    fasttime_t time1 = gettime();
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < N; j++) {
            c[j] = a[j] + b[j];
        }
    }

    fasttime_t time2 = gettime();

    double elapsedf = tdiff(time1, time2);
    std::cout << "Elapsed execution time of the loop in test1():\n"
              << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}