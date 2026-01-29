#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#define checkCudaError(...) \
    checkCudaErrorImpl(__FILE__, __LINE__, __VA_ARGS__)

inline void checkCudaErrorImpl(const char* file, int line, cudaError_t code, bool checkGetLastError) {
    if (cudaSuccess != code) {
        fprintf(stderr, "CUDA Error (%s : %d) --- %s\n", file, line, cudaGetErrorString(code));
        exit(1);
    }
    if (checkGetLastError) {
        checkCudaErrorImpl(file, line, cudaGetLastError(), false);
    }
}