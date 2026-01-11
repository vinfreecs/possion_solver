#include "cuda_solver.cuh"
#include <float.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "solver.h"

int solve_cuda(Solver *solver) {
  double r;
  int it = 0;
  double res, res1;

  int imax = solver->imax;
  int jmax = solver->jmax;
  int jmaxLocal = solver->jmaxLocal;
  double eps = solver->eps;
  double omega = solver->omega;
  int itermax = solver->itermax;

  double dx2 = solver->dx * solver->dx;
  double dy2 = solver->dy * solver->dy;
  double idx2 = 1.0 / dx2;
  double idy2 = 1.0 / dy2;
  double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
  double *p = solver->p;
  double *rhs = solver->rhs;
  double epssq = eps * eps;

  res = eps + 1.0;

  int threadsPerBlock = 64;
  int numBlocks = ((jmax + 2) * (imax + 2)) / threadsPerBlock;

  while ((res >= epssq) && (it < itermax)) {
    res = 0.0;
    exchange(solver);

    stencil<<<threadsPerBlock, numBlocks>>>();

    if (solver->rank == 0) {
      for (int i = 1; i < imax + 1; i++) {
        P(i, 0) = P(i, 1);
      }
    }

    if (solver->rank == (solver->size - 1)) {
      for (int i = 1; i < imax + 1; i++) {
        P(i, jmaxLocal + 1) = P(i, jmaxLocal);
      }
    }

    for (int j = 1; j < jmaxLocal + 1; j++) {
      P(0, j) = P(1, j);
      P(imax + 1, j) = P(imax, j);
    }

    MPI_Allreduce(&res, &res1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    res = res1;
    res = sqrt(res / (imax * jmax));
#ifdef DEBUG
    if (solver->rank == 0) {
      printf("%d Residuum: %e\n", it, res1);
    }
#endif
    it++;
  }

  if (solver->rank == 0) {
    printf("Solver took %d iterations\n", it);
  }
  if (res < eps) {
    return 1;
  } else {
    return 0;
  }
}
