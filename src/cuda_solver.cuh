#include "cuda_runtime.h"

__global__ void stencil_cuda(double res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p);

__global__ void outer_boundary_cuda(double *p, int rank, int size, int imax,
                                    int jmaxLocal);

__global__ void reduce_(int n, double res);

// TODO device function
// __global__ static void exchange_cuda(int rank, int size, double *p,
//                                      int jmaxLocal, int imax);

#define CUDA_SETUP()                                                           \
  int size_p = (solver.imax + 2) * (solver.jmaxLocal + 2) * sizeof(double);    \
  int size_rhs = (solver.imax + 2) * (solver.jmax + 2) * sizeof(double);       \
  double *p_d, *p_new_d, *rhs_d, *d_res;                                       \
  checkCudaError(cudaMalloc((void **)&p_d, size_p));                           \
  checkCudaError(cudaMemcpy(p_d, solver.p, size_p, cudaMemcpyHostToDevice));   \
  checkCudaError(cudaMalloc((void **)&p_new_d, size_p));                       \
  checkCudaError(cudaMemcpy(p_new_d, p_d, size_p, cudaMemcpyDeviceToDevice));  \
  checkCudaError(cudaMalloc((void **)&rhs_d, size_rhs));                       \
  checkCudaError(                                                              \
      cudaMemcpy(rhs_d, solver.rhs, size_rhs, cudaMemcpyHostToDevice));        \
  checkCudaError(cudaMalloc((void **)&d_res, sizeof(double)));                 \
  int threadsPerBlock = 256;                                                   \
  int blocksPerGrid =                                                          \
      (solver.jmaxLocal + threadsPerBlock - 1) / threadsPerBlock;              \
  int highPriority = 0, lowPriority = 0;                                       \
  checkCudaError(                                                              \
      cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));          \
  cudaStream_t stream_stencil;                                                 \
  checkCudaError(cudaStreamCreateWithPriority(                                 \
      &stream_stencil, cudaStreamDefault, lowPriority));                       \
  cudaStream_t stream_boundary;                                                \
  checkCudaError(cudaStreamCreateWithPriority(                                 \
      &stream_boundary, cudaStreamDefault, highPriority));                     \
  cudaEvent_t event_boundary;                                                  \
  checkCudaError(cudaEventCreate(&event_boundary));                            \
                                                                               \
  double r;                                                                    \
  int it = 0;                                                                  \
  double res, res1;                                                            \
  int imax = solver.imax;                                                      \
  int jmax = solver.jmax;                                                      \
  int jmaxLocal = solver.jmaxLocal;                                            \
  double eps = solver.eps;                                                     \
  double omega = solver.omega;                                                 \
  int itermax = solver.itermax;                                                \
  double dx2 = solver.dx * solver.dx;                                          \
  double dy2 = solver.dy * solver.dy;                                          \
  double idx2 = 1.0 / dx2;                                                     \
  double idy2 = 1.0 / dy2;                                                     \
  double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);                     \
  double *p = solver.p;                                                        \
  double *rhs = solver.rhs;                                                    \
  double epssq = eps * eps;                                                    \
  double size = solver.size;                                                   \
  res = eps + 1.0;

#define CLEAN_UP_CUDA()                                                        \
  checkCudaError(cudaMemcpy(solver.p, p_d, size_p, cudaMemcpyDeviceToHost));   \
  checkCudaError(                                                              \
      cudaMemcpy(solver.rhs, rhs_d, size_rhs, cudaMemcpyDeviceToHost));        \
  checkCudaError(cudaStreamDestroy(stream_stencil));                           \
  checkCudaError(cudaStreamDestroy(stream_boundary));                          \
  checkCudaError(cudaEventDestroy(event_boundary));                            \
  cudaFree(p_d);                                                               \
  cudaFree(p_new_d);                                                           \
  cudaFree(rhs_d);                                                             \
  checkCudaError(cudaFree(d_res));
