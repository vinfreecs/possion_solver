

#include "cuda_error.h"
#include "cuda_runtime.h"
#include "cuda_solver.cuh"
#include <cstdio>
#include <cub/cub.cuh>

#define PI 3.14159265358979323846
#define P(i, j) p[(j) * (imax + 2) + (i)]
#define P_N(i, j) p_new[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

__global__ void stencil_cuda(double *d_res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p, double *p_new,
                             bool compute_norm) {

  using BlockReduce = cub::BlockReduce<double, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  double temp = 0;
  if (j <= jmaxLocal && i <= imax) {

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);
    P_N(i, j) = P(i, j) - (factor * r);
    if (compute_norm)
      temp = r * r;
  }
  if (compute_norm) {
    double aggregate = BlockReduce(temp_storage).Sum(temp);
    if (threadIdx.x == 0 && threadIdx.y == 0)
      atomicAdd(d_res, aggregate);
  }
}

__global__ void outer_boundary_cuda(double *p_new, int rank, int size, int imax,
                                    int jmaxLocal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > imax + 1) {
    return;
  }
  if (rank == 0 && i <= imax + 1) {
    P_N(i, 0) = P_N(i, 1);
  }
  if (rank == (size - 1) && i <= imax + 1) {
    P_N(i, jmaxLocal + 1) = P_N(i, jmaxLocal);
  }
  if (i <= jmaxLocal + 1) {
    P_N(0, i) = P_N(1, i);
    P_N(imax + 1, i) = P_N(imax, i);
  }
}

extern "C" void launch_stencil_kernel(double *d_res, double *h_res, double eps,
                                      double factor, int imax, int jmaxLocal,
                                      double r, double idx2, double idy2,
                                      double *rhs, double *p, double *p_new,
                                      int rank, int size, int blocksPerGrid,
                                      int threadsPerBlock, bool compute_norm) {

  dim3 threads(32, 8);
  dim3 blocks((imax + threads.x - 1) / threads.x,
              (jmaxLocal + threads.y - 1) / threads.y);

  stencil_cuda<<<blocks, threads>>>(d_res, eps, factor, imax, jmaxLocal, r,
                                    idx2, idy2, rhs, p, p_new, compute_norm);

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
  int boundary_blocks = (imax + 2 + threadsPerBlock - 1) / threadsPerBlock;
  outer_boundary_cuda<<<boundary_blocks, threadsPerBlock>>>(p_new, rank, size,
                                                            imax, jmaxLocal);
}