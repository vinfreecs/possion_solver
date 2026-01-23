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
