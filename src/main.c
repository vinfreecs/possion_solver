/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include "cuda_error.h"
#include "cuda_runtime.h"
#include "cuda_solver.cuh"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "solver.h"
#include "timing.h"

void launch_stencil_kernel(double *d_res, double *h_res, double eps,
                           double factor, int imax, int jmaxLocal, double r,
                           double idx2, double idy2, double *rhs, double *p_old,
                           double *p_new, int rank, int size, int blocksPerGrid,
                           int threadsPerBlock, bool compute_norm);

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

  // CUDA
  // TODO in host initialzing only once
  //  Gets number of GPU device per node.
  checkCudaError(cudaGetDeviceCount(&num_devices));
  // Particular MPI rank invoking this selects the GPU for execution
  int device_id = rank % num_devices;
  checkCudaError(cudaSetDevice(device_id));
  printf("Rank %d selected GPU %d out of %d GPUs\n", rank, device_id,
         num_devices);

  // CUDA

  // intialising the data on the gpu
  initSolver(&solver, &params, 2);

  // CUDA
  int size_p = (solver.imax + 2) * (solver.jmaxLocal + 2) * sizeof(double);
  int size_rhs = (solver.imax + 2) * (solver.jmax + 2) * sizeof(double);

  double *p_d;
  checkCudaError(cudaMalloc((void **)&p_d, size_p));
  checkCudaError(cudaMemcpy(p_d, solver.p, size_p, cudaMemcpyHostToDevice));

  double *p_new_d;
  checkCudaError(cudaMalloc((void **)&p_new_d, size_p));
  checkCudaError(cudaMemcpy(p_new_d, p_d, size_p, cudaMemcpyDeviceToDevice));

  double *rhs_d;
  checkCudaError(cudaMalloc((void **)&rhs_d, size_rhs));
  checkCudaError(
      cudaMemcpy(rhs_d, solver.rhs, size_rhs, cudaMemcpyHostToDevice));
  int threadsPerBlock = 256;
  // TODO
  //  int blocksPerGrid = (num_devices + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGrid =
      (solver.jmaxLocal + threadsPerBlock - 1) / threadsPerBlock;
  // CUDA

  // solve(&solver);

  // CUDA
  double *d_res;
  checkCudaError(cudaMalloc((void **)&d_res, sizeof(double)));
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
  res = eps + 1.0;
  double start_time = getTimeStamp();
  while ((res >= epssq) && (it < itermax)) {
    bool compute_norm = (it % 1000 == 0);

    if (compute_norm)
      checkCudaError(cudaMemset(d_res, 0, sizeof(double)));

    exchange_cuda(rank, size, p_d, jmaxLocal, imax);

    launch_stencil_kernel(d_res, &res, eps, factor, imax, jmaxLocal, r, idx2,
                          idy2, rhs_d, p_d, p_new_d, rank, size, blocksPerGrid,
                          threadsPerBlock, compute_norm);

    double *temp = p_d;
    p_d = p_new_d;
    p_new_d = temp;

    if (compute_norm) {
      checkCudaError(
          cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost));
      MPI_Allreduce(&res, &res1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = res1;
      res = sqrt(res / (imax * jmax));

      if (rank == 0)
        printf("Iter %d, Residual: %f\n", it, res);
      if (isnan(res))
        break;
    }
    it++;
  }
  double stop_time = getTimeStamp();

  checkCudaError(cudaMemcpy(solver.p, p_d, size_p, cudaMemcpyDeviceToHost));
  checkCudaError(
      cudaMemcpy(solver.rhs, rhs_d, size_rhs, cudaMemcpyDeviceToHost));

  // CUDA

  // getResult(&solver);

  if (rank == 0) {
    double time_taken = stop_time - start_time;
    printf("Solver took %d iterations\n", it);
    printf("Time taken is %f \n", time_taken);
    double perf = (double)it * (double)imax * (double)jmax / (time_taken * 1e6);
    printf("The performance %f in MLUP/s \n", perf);
  }

  // CUDA
  cudaFree(p_d);
  cudaFree(p_new_d);
  cudaFree(rhs_d);
  checkCudaError(cudaFree(d_res));
  // CUDA
  MPI_Finalize();
  return EXIT_SUCCESS;
}
