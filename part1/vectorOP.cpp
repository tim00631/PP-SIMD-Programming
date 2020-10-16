#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // All ones
        maskAll = _pp_init_ones();

        // All zeros
        maskIsNegative = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll);  // x = values[i];

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll);  // if (x < 0) {

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative);  //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative);  // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative);  //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //
    // Handle N % VECTOR_WIDTH != 0
    int add_zero_cnt = N / VECTOR_WIDTH; // quotient
    int need_add_zero = N % VECTOR_WIDTH; // remainder
    __pp_vec_float result;
    __pp_vec_float x;
    __pp_vec_int y;
    __pp_vec_int zero = _pp_vset_int(0);
    __pp_vec_float float_zero = _pp_vset_float(0.f);
    __pp_vec_float float_one = _pp_vset_float(1.f);
    __pp_mask maskAll, maskAllZero, maskExpIsZero, maskExpIsNotZero;
    // 0 < 7, 4 < 7
    // 0123, 456"7", need to fill zero is 1 bit
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // All ones
        maskAll = _pp_init_ones();
        // All zeros
        maskAllZero = _pp_init_ones(0);
        maskExpIsZero = _pp_init_ones(0);
        if (!add_zero_cnt){
            maskAll = _pp_init_ones(need_add_zero); 
        } else {
            add_zero_cnt--;
        }
        _pp_vload_float(x, values + i, maskAll); // x = values[i];
        _pp_vload_int(y, exponents + i, maskAll); // y = exponent[i];
        _pp_veq_int(maskExpIsZero, y, zero, maskAll); // if(y == 0) {
        _pp_vstore_float(output + i, float_one, maskExpIsZero); // output[i] = 1.f;
        maskExpIsNotZero = _pp_mask_not(maskExpIsZero); // } else {
        if (!add_zero_cnt){
            maskExpIsNotZero = _pp_mask_and(maskExpIsNotZero, maskAll); 
        }
        __pp_vec_float temp;
        _pp_vmove_float(temp, x, maskExpIsNotZero); // float temp = x; 
        __pp_vec_int count; // int count;
        __pp_vec_int constant_one = _pp_vset_int(1);
        _pp_vsub_int(count, y, constant_one, maskExpIsNotZero); // count = y - 1;

        __pp_mask mask_cntIsGreaterThanZero = _pp_init_ones(0);
        _pp_vgt_int(mask_cntIsGreaterThanZero, count, zero, maskExpIsNotZero);
        while(_pp_cntbits(mask_cntIsGreaterThanZero)) { // while (count > 0) {
            _pp_vmult_float(temp, temp, x, mask_cntIsGreaterThanZero); // temp *= x;
            _pp_vsub_int(count, count, constant_one, mask_cntIsGreaterThanZero); // count--;
            _pp_vgt_int(mask_cntIsGreaterThanZero, count, zero, maskExpIsNotZero);
        }
        __pp_vec_float float_upper_bound = _pp_vset_float(9.999999f);
        __pp_mask mask_overflow = _pp_init_ones(0);
        _pp_vgt_float(mask_overflow, temp, float_upper_bound, maskExpIsNotZero);
        _pp_vset_float(temp, 9.999999f, mask_overflow);
        _pp_vstore_float(output + i, temp, maskExpIsNotZero);
        
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //
    // [0 1 2 3 4 5 6 7]:
    // hadd -> [interleave -> hadd] -> interleave -> hadd -> return value[0]
    //                      count = 4 ,                           count = 2
    // [1 1 5 5 4 5 6 7] -> [1 5 1 5 4 5 6 7] -> [6 6 6 6 4 5 6 7]
    // 
    // [6 6 6 6 9 9 13 13] -> [6 6 6 6 9 13 9 13] -> [6 6 6 6 22 22 22 22]
    __pp_mask maskAll = _pp_init_ones();
    // __pp_mask mask_first = _pp_init_ones(1);
    __pp_vec_float sum = _pp_vset_float(0.f);
    // double sum = 0;
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __pp_vec_float x;
        _pp_vload_float(x, values + i, maskAll);
        _pp_hadd_float(x, x);
        int count = VECTOR_WIDTH;
        while(count > 2){
            _pp_interleave_float(x, x);
            _pp_hadd_float(x, x);
            count /= 2;
        }
        // sum += x.value[0];
        _pp_vadd_float(sum, sum, x, maskAll);
        // for(int j = i; j < i+VECTOR_WIDTH; j++){
        //     printf("values[%d]: %.7f\n", j, values[j]);
        // }
        // printf("第%d次: sum:%.7f\n",i/VECTOR_WIDTH, sum.value[0]);
    }
    return sum.value[0];
    // return sum;

}