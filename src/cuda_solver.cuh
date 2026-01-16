#include "cuda_runtime.h"

#define PI 3.14159265358979323846
#define P(i, j) p[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

__global__ void stencil_cuda(double res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p);

__global__ void outer_boundary_cuda(double *p, int rank, int size, int imax,
                                    int jmaxLocal);

__global__ void reduce_(int n, double res);

// TODO device function
// __global__ static void exchange_cuda(int rank, int size, double *p,
//                                      int jmaxLocal, int imax);
