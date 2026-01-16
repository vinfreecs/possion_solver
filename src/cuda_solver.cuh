#include "solver.h"

#define PI 3.14159265358979323846
#define P(i, j) p_d[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs_d[(j) * (imax + 2) + (i)]

__global__ void stencil_cuda(double res, double eps, double factor, int imax,
                             int jmaxLocal, double r, double idx2, double idy2,
                             double *rhs, double *p) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= jmaxLocal)
    return;

  double epssq = eps * eps;

  // for (int j = 1; j < jmaxLocal + 1; j++)
  for (int i = 1; i < imax + 1; i++) {

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

    P(i, j) -= (factor * r);
    // res += (r * r);
    atomicAdd(res, r * r);
  }
}

__global__ void outer_boundary_cuda(double *p_d, int rank, int size, int imax,
                                    int jmaxLocal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ static void exchange_cuda(int rank, int size, double *p,
                                     int jmaxLocal, int imax) {
  MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  /* exchange ghost cells with top neighbor */
  if (rank + 1 < size) {
    int top = rank + 1;
    double *src = p + (jmaxLocal) * (imax + 2) + 1;
    double *dst = p + (jmaxLocal + 1) * (imax + 2) + 1;

    MPI_Isend(src, imax, MPI_DOUBLE, top, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(dst, imax, MPI_DOUBLE, top, 2, MPI_COMM_WORLD, &requests[1]);
  }

  /* exchange ghost cells with bottom neighbor */
  if (rank > 0) {
    int bottom = rank - 1;
    double *src = p + (imax + 2) + 1;
    double *dst = p + 1;

    MPI_Isend(src, imax, MPI_DOUBLE, bottom, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(dst, imax, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &requests[3]);
  }

  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}