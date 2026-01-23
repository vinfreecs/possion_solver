#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* * 1. The Implementation Function
 * We use 'static inline' so this function can be defined in a header
 * without causing "multiple definition" linker errors.
 */
static inline void checkCudaErrorImpl(const char *file, int line,
                                      cudaError_t code, int checkGetLastError) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error (%s : %d) --- %s\n", file, line,
            cudaGetErrorString(code));
    exit(1);
  }

  if (checkGetLastError) {
    // Recursive call with 0 (false) to prevent infinite loops
    checkCudaErrorImpl(file, line, cudaGetLastError(), 0);
  }
}

/*
 * 2. The Macro
 * We explicitly pass '0' for the checkGetLastError argument
 * because C functions cannot have default parameters.
 */
#define checkCudaError(val) checkCudaErrorImpl(__FILE__, __LINE__, val, 0)

// Optional: A helper macro specifically for Kernel launches (which return void)
#define checkKernelLaunch()                                                    \
  checkCudaErrorImpl(__FILE__, __LINE__, cudaGetLastError(), 0)

#endif