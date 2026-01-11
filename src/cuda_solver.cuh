#include "solver.h"

#define PI 3.14159265358979323846
#define P(i, j) p[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

__global__ void stencil(int n, double *rhs, double *p, Solver *solve) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n)
    return;

  double r;
  int it = 0;
  double res, res1;

  int imax = solver->imax;
  double eps = solver->eps;
  double omega = solver->omega;

  double dx2 = solver->dx * solver->dx;
  double dy2 = solver->dy * solver->dy;
  double idx2 = 1.0 / dx2;
  double idy2 = 1.0 / dy2;
  double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
  double epssq = eps * eps;

  res = eps + 1.0;
  // for (int j = 1; j < jmaxLocal + 1; j++)
  for (int i = 1; i < imax + 1; i++) {

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

    P(i, j) -= (factor * r);
    res += (r * r);
  }
}

__global__ void reduce_(int n, double res) {}