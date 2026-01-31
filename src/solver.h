/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "parameter.h"
#include "cuda-util.h"
#include "cuda_runtime.h"
#include "util.h"

typedef struct {
    double dx, dy;
    double ys;
    int imax, jmax;
    int jmaxLocal;
    int rank;
    int size;

    double *p, *rhs;
    double *p_d, *rhs_d;
    double eps, omega;
    int itermax;
} Solver;

#ifdef __cplusplus
extern "C" {
#endif

extern void debug(Solver*);
extern void initSolver(int argc, char** argv, Solver*, Parameter*, int problem);
extern void getResult(Solver*, char*);
extern void writeResult(Solver*, double*, char*);
extern int solve(Solver*);
extern void finalize(Solver* solver);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
extern __global__ void res_kernel(double*, double*, double*, int, int, double, double, double);
extern __global__ void init_kernel(double* , double*, int , int, double, double, double,  int);
extern __global__ void outerBoundaryCopy(double*, int, int, int, int);
#endif

#endif
