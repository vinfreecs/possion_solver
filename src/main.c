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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "solver.h"

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
  cudaMalloc(p_d, size_p);
  double *rhs_d;
  cudaMalloc(rhs_d, size_rhs);

  cudaMemcpy(p_d, solver.p, size_p, cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_d, solver.rhs, size_rhs, cudaMemcpyHostToDevice);

  solve(&solver);

  /*

  for(iter){
    exchange() //basically exchange the data between ranks so the data con be
  transfered to gpus cudaMemcpy() // cpy the halo data from cpu to gpu
    kernal_to_do_stencil_op
    allreduce -> atomic reduce
  }

   */

  getResult(&solver);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
