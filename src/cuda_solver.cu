

#include "cuda_error.h"
#include "cuda_runtime.h"
#include "cuda_solver.cuh"
#include <cstdio>

#define PI 3.14159265358979323846
#define P(i, j) p[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]
__global__ void stencil_cuda(double *d_res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p) {

  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j > jmaxLocal) {
    return;
  }

  double epssq = eps * eps;

  // printf("Entering stencil \n");
  double temp = 0;

  // for (int j = 1; j < jmaxLocal + 1; j++)
  for (int i = 1; i < imax + 1; i++) {
    // printf("the value of j : %d i : %d value of p : %f rhs : %f \n", j, i,
    //        P(i, j), RHS(i, j));

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

    P(i, j) -= (factor * r);
    temp += r * r;
  }
  atomicAdd(d_res, temp);
}

__global__ void outer_boundary_cuda(double *p, int rank, int size, int imax,
                                    int jmaxLocal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > imax + 1) {
    return;
  }
  if (rank == 0) {
    // printf("Entering boundary \n");

    P(i, 0) = P(i, 1);
  }

  if (rank == (size - 1)) {
    // printf("Entering boundary \n");

    P(i, jmaxLocal + 1) = P(i, jmaxLocal);
  }

  P(0, i) = P(1, i);
  P(imax + 1, i) = P(imax, i);
}

// __global__ void reduce_(int n, double res) {}

extern "C" void launch_stencil_kernel(double *d_res, double *h_res, double eps,
                                      double factor, int imax, int jmaxLocal,
                                      double r, double idx2, double idy2,
                                      double *rhs, double *p, int rank,
                                      int size, int blocksPerGrid,
                                      int threadsPerBlock) {

  // as exchanche is not cuda kernal how to know for sure the the exchange has
  // happeded before starting the next iteration
  // printf("start stencil From Host\n");

  stencil_cuda<<<blocksPerGrid, threadsPerBlock>>>(
      d_res, eps, factor, imax, jmaxLocal, r, idx2, idy2, rhs, p);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  // printf("end stencil From Host\n");
  // printf("start boundary From Host\n");

  checkCudaError(
      cudaMemcpy(h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost));

  int boundary_blocks = (imax + 2 + threadsPerBlock - 1) / threadsPerBlock;
  outer_boundary_cuda<<<boundary_blocks, threadsPerBlock>>>(p, rank, size, imax,
                                                            jmaxLocal);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  // printf("end Boundary From Host\n");
}