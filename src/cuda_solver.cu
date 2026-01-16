

#include "cuda_runtime.h"
#include "cuda_solver.cuh"
#include <iostream>
__global__ void stencil_cuda(double res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= jmaxLocal)
    return;

  double epssq = eps * eps;

  printf("Entering stencil \n");

  // for (int j = 1; j < jmaxLocal + 1; j++)
  for (int i = 1; i < imax + 1; i++) {

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

    P(i, j) -= (factor * r);
    // res += (r * r);
    double temp = r * r;
    atomicAdd(&res, temp);
  }
}

__global__ void outer_boundary_cuda(double *p, int rank, int size, int imax,
                                    int jmaxLocal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Entering boundary \n");

  if (i >= imax + 1) {
    return;
  }
  if (rank == 0) {

    P(i, 0) = P(i, 1);
  }

  if (rank == (size - 1)) {

    P(i, jmaxLocal + 1) = P(i, jmaxLocal);
  }

  P(0, i) = P(1, i);
  P(imax + 1, i) = P(imax, i);
}

__global__ void reduce_(int n, double res) {}

extern "C" void launch_stencil_kernel(double res, double eps, double factor,
                                      int imax, int jmaxLocal, double r,
                                      double idx2, double idy2, double *rhs,
                                      double *p, int rank, int size,
                                      int blocksPerGrid, int threadsPerBlock) {

  stencil_cuda<<<blocksPerGrid, threadsPerBlock>>>(
      res, eps, factor, imax, jmaxLocal, r, idx2, idy2, rhs, p);
  outer_boundary_cuda<<<blocksPerGrid, threadsPerBlock>>>(p, rank, size, imax,
                                                          jmaxLocal);

  cudaDeviceSynchronize();
}