/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include "cuda_runtime.h"
#include "cuda_solver.cuh"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "solver.h"

void launch_stencil_kernel(double res, double eps, double factor, int imax,
                           int jmaxLocal, double r, double idx2, double idy2,
                           double *rhs, double *p, int rank, int size,
                           int blocksPerGrid, int threadsPerBlock);

static void exchange_cuda(int rank, int size, double *p, int jmaxLocal,
                          int imax) {
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

int main(int argc, char **argv) {
  int rank;
  Parameter params;
  Solver solver;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  initParameter(&params);

  if (argc != 2) {
    printf("Usage: %s <configFile>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  readParameter(&params, argv[1]);
  if (rank == 0) {
    printParameter(&params);
  }

  int num_devices = 0;

  // Gets number of GPU device per node.
  cudaGetDeviceCount(&num_devices);
  // Particular MPI rank invoking this selects the GPU for execution
  cudaSetDevice(rank % num_devices);

  // intialising the data on the gpu
  initSolver(&solver, &params, 2);
  int size_p = (solver.imax + 2) * (solver.jmaxLocal + 2) * sizeof(double);
  int size_rhs = (solver.imax + 2) * (solver.jmax + 2) * sizeof(double);

  double *p_d;
  cudaMalloc((void **)&p_d, size_p);
  double *rhs_d;
  cudaMalloc((void **)&rhs_d, size_rhs);

  cudaMemcpy(p_d, solver.p, size_p, cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_d, solver.rhs, size_rhs, cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_devices + threadsPerBlock - 1) / threadsPerBlock;

  solve(&solver);

  /*

  for(iter){
    exchange() //basically exchange the data between ranks so the data con be
  transfered to gpus cudaMemcpy() // cpy the halo data from cpu to gpu
    kernal_to_do_stencil_op
    allreduce -> atomic reduce
  }
*/
  double r;
  int it = 0;
  double res, res1;

  int imax = solver.imax;
  int jmax = solver.jmax;
  int jmaxLocal = solver.jmaxLocal;
  double eps = solver.eps;
  double omega = solver.omega;
  int itermax = solver.itermax;

  double dx2 = solver.dx * solver.dx;
  double dy2 = solver.dy * solver.dy;
  double idx2 = 1.0 / dx2;
  double idy2 = 1.0 / dy2;
  double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
  double *p = solver.p;
  double *rhs = solver.rhs;
  double epssq = eps * eps;
  double size = solver.size;

  while ((res >= epssq) && (it < itermax)) {
    res = 0.0;
    exchange_cuda(rank, size, p_d, jmaxLocal, imax);
    launch_stencil_kernel(res, eps, factor, imax, jmaxLocal, r, idx2, idy2,
                          rhs_d, p_d, rank, size, blocksPerGrid,
                          threadsPerBlock);
    MPI_Allreduce(&res, &res1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    res = res1;
    res = sqrt(res / (imax * jmax));
    it++;
  }

  if (rank == 0) {
    cudaMemcpy(solver.p, p_d, size_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(solver.rhs, rhs_d, size_rhs, cudaMemcpyDeviceToHost);
  }

  getResult(&solver);

  cudaFree(p_d);
  cudaFree(rhs_d);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
