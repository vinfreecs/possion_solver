/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocate.h"
#include "parameter.h"
#include "solver.h"
// #include "util.h"
// #include "cuda-util.h"

#define PI        3.14159265358979323846
#define P(i, j)   p[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]
#define P_D(i, j) p_d[(j) * (imax + 2) + (i)]
#define RHS_D(i, j) rhs_d[(j) * (imax + 2) + (i)]

// #define checkCudaError(...) \
//     checkCudaErrorImpl(__FILE__, __LINE__, __VA_ARGS__)

// inline void checkCudaErrorImpl(const char* file, int line, cudaError_t code, bool checkGetLastError) {
//     if (cudaSuccess != code) {
//         fprintf(stderr, "CUDA Error (%s : %d) --- %s\n", file, line, cudaGetErrorString(code));
//         exit(1);
//     }
//     if (checkGetLastError) {
//         checkCudaErrorImpl(file, line, cudaGetLastError(), false);
//     }
// }

static int sizeOfRank(int rank, int size, int N) {
    return N / size + ((N % size > rank) ? 1 : 0);
}

static void print(Solver* solver) {
    double* p = solver->p;
    int imax  = solver->imax;

    printf("### RANK %d #######################################################\n",
        solver->rank);
    for (int j = 0; j < solver->jmaxLocal + 2; j++) {
        printf("%02d: ", j);
        for (int i = 0; i < solver->imax + 2; i++) {
            printf("%12.8f  ", P(i, j));
        }
        printf("\n");
    }
    fflush(stdout);
}

static void exchange(Solver* solver) {
    MPI_Request requests[4] = { MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL };

    /* exchange ghost cells with top neighbor */
    if (solver->rank + 1 < solver->size) {
        int top     = solver->rank + 1;
        double* src = solver->p + (solver->jmaxLocal) * (solver->imax + 2) + 1;
        double* dst = solver->p + (solver->jmaxLocal + 1) * (solver->imax + 2) + 1;

        MPI_Isend(src, solver->imax, MPI_DOUBLE, top, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(dst, solver->imax, MPI_DOUBLE, top, 2, MPI_COMM_WORLD, &requests[1]);
    }

    /* exchange ghost cells with bottom neighbor */
    if (solver->rank > 0) {
        int bottom  = solver->rank - 1;
        double* src = solver->p + (solver->imax + 2) + 1;
        double* dst = solver->p + 1;

        MPI_Isend(src, solver->imax, MPI_DOUBLE, bottom, 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(dst, solver->imax, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &requests[3]);
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

void getResult(Solver* solver, char filename[]) {
    double* Pall = NULL;
    int *rcvCounts, *displs;

    if (solver->rank == 0) {
        checkCudaError(cudaMallocHost(&Pall, (solver->imax + 2) * (solver->jmax + 2) * sizeof(double)), false);
        rcvCounts    = (int*)malloc(solver->size * sizeof(int));
        displs       = (int*)malloc(solver->size * sizeof(int));
        rcvCounts[0] = solver->jmaxLocal * (solver->imax + 2);
        displs[0]    = 0;
        int cursor   = rcvCounts[0];

        for (int i = 1; i < solver->size; i++) {
            rcvCounts[i] = sizeOfRank(i, solver->size, solver->jmax) * (solver->imax + 2);
            displs[i]    = cursor;
            cursor += rcvCounts[i];
        }
    }

    int cnt            = solver->jmaxLocal * (solver->imax + 2);
    double* sendbuffer = solver->p + (solver->imax + 2);
    MPI_Gatherv(sendbuffer,
        cnt,
        MPI_DOUBLE,
        Pall,
        rcvCounts,
        displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    if (solver->rank == 0) {
        writeResult(solver, Pall, filename);
    }
}

void initSolver(int argc, char** argv, Solver* solver, Parameter* params, int problem) {
    MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_rank(MPI_COMM_WORLD, &(solver->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(solver->size));
    solver->imax      = params->imax;
    solver->jmax      = params->jmax;
    solver->jmaxLocal = sizeOfRank(solver->rank, solver->size, solver->jmax);
    printf("RANK %d: imaxLocal : %d, jmaxLocal : %d\n",
        solver->rank,
        solver->imax,
        solver->jmaxLocal);

    solver->dx      = params->xlength / params->imax;
    solver->dy      = params->ylength / params->jmax;
    solver->ys      = solver->rank * solver->jmaxLocal * solver->dy;
    solver->eps     = params->eps;
    solver->omega   = params->omg;
    solver->itermax = params->itermax;

    int imax      = solver->imax;
    int jmax      = solver->jmax;
    int jmaxLocal = solver->jmaxLocal;

    checkCudaError(cudaMallocHost(&solver->p , (imax + 2) * (jmaxLocal + 2) * sizeof(double)), false);
    checkCudaError(cudaMallocHost(&solver->rhs, (imax + 2) * (jmaxLocal + 2) * sizeof(double)), false);

    checkCudaError(cudaMalloc(&solver->p_d , (imax + 2) * (jmaxLocal + 2) * sizeof(double)), false);
    checkCudaError(cudaMalloc(&solver->rhs_d , (imax + 2) * (jmaxLocal + 2) * sizeof(double)), false);
    checkCudaError(cudaMemcpy(solver->p_d, solver->p, (imax + 2) * (jmaxLocal + 2) * sizeof(double), cudaMemcpyHostToDevice), false);
    checkCudaError(cudaMemcpy(solver->rhs_d, solver->rhs, (imax + 2) * (jmaxLocal + 2) * sizeof(double), cudaMemcpyHostToDevice), false);
    init_kernel<<<32,32>>>(solver, problem);
    checkCudaError(cudaGetLastError(), true);
    checkCudaError(cudaDeviceSynchronize(), true);
    getResult(solver, "init.dat");
}

void debug(Solver* solver) {
    int imax  = solver->imax;
    int rank  = solver->rank;
    double* p = solver->p;

        for( int j=0; j < solver->jmaxLocal+2; j++ ) {
             for( int i=0; i < solver->imax+2; i++ ) {
                 P(i, j) = (double) rank; 
             } 
         }

         for ( int i=0; i < solver->size; i++) {
             if ( i == rank ) {
                print(solver);
             }
             MPI_Barrier(MPI_COMM_WORLD);
         }

         if ( rank == 0 ) {
             printf("##########################################################\n"); 
             printf("##  Exchange ghost layers\n");
             printf("##########################################################\n");
         }
         exchange(solver);

    for (int i = 0; i < solver->size; i++) {
        if (i == rank) {
            print(solver);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
__global__
void init_kernel(double* p_d, int problem) {
    int imax = solver->imax;
    int jmaxLocal = solver->jmaxLocal;
    double dx   = solver->dx;
    double dy   = solver->dy;
    double ys   = solver->ys;
    double* p_d   = solver->p_d;
    double* rhs_d = solver->rhs_d;

    for (int j = 0; j < jmaxLocal + 2; j++) { // Can be done in GPU? Initialize Pressure field to (sin(4piπx) + sin(4piπy))
        double y = ys + j * dy;
        for (int i = 0; i < imax + 2; i++) {
            P_D(i, j) = sin(4.0 * PI * i * dx) + sin(4.0 * PI * y);
        }
    }

    if (problem == 2) {// Offload to CUDA Kernel
        for (int j = 0; j < jmaxLocal + 2; j++) { // Can be done in GPU? Initialize RHS_D to sin(2piπx)
            for (int i = 0; i < imax + 2; i++) {
                RHS_D(i, j) = sin(2.0 * PI * i * dx);
            }
        }
    } else {
        for (int j = 0; j < jmaxLocal + 2; j++) { // Can be done in GPU directly? Initialize RHS_D to 0
            for (int i = 0; i < imax + 2; i++) {
                RHS_D(i, j) = 0.0;
            }
        }
    }

}
__global__
void res_kernel(Solver* solver, double factor, double* res) {

    int imax = solver->imax;
    int jmaxLocal = solver->jmaxLocal;
    double* p_d = solver->p_d;
    double* rhs_d = solver->rhs_d;
    double r;
    double dx2    = solver->dx * solver->dx;
    double dy2 = solver->dy * solver->dy;
    double idx2   = 1.0 / dx2;
    double idy2   = 1.0 / dy2;
    for (int j = 1; j < jmaxLocal + 1; j++) {
        for (int i = 1; i < imax + 1; i++) {

            r = RHS_D(i, j) - ((P_D(i - 1, j) - 2.0 * P_D(i, j) + P_D(i + 1, j)) * idx2 +
                                (P_D(i, j - 1) - 2.0 * P_D(i, j) + P_D(i, j + 1)) * idy2);

            P_D(i, j) -= (factor * r);
            *res += (r * r);
        }
    }
    
}

__global__
void copyHorizantal(Solver* solver) {
    double* p_d = solver->p_d;
    int imax = solver->imax;
    int jmaxLocal = solver->jmaxLocal;
    // impelementation of horizantal boundaries
    for (int j = 1; j < jmaxLocal + 1; j++) {
            P_D(0, j)        = P_D(1, j);
            P_D(imax + 1, j) = P_D(imax, j);
    }
}

__global__
void copyVertical(Solver* solver) {
    int imax = solver->imax;
    int jmaxLocal = solver->jmaxLocal;
    double* p_d = solver->p_d;
    int rank = solver->rank;
    int size = solver->size;

    // impelementation of vertical boundaries
    if (solver->rank == 0) {
            for (int i = 1; i < imax + 1; i++) {
                P_D(i, 0) = P_D(i, 1);
            }
        }

    if (solver->rank == (solver->size - 1)) {
        for (int i = 1; i < imax + 1; i++) {
            P_D(i, jmaxLocal + 1) = P_D(i, jmaxLocal);
        }
    }
}

int solve(Solver* solver) {
    double r;
    int it = 0;
    double res, res1;

    int imax      = solver->imax;
    int jmax      = solver->jmax;
    int jmaxLocal = solver->jmaxLocal;
    double eps    = solver->eps;
    double omega  = solver->omega;
    int itermax   = solver->itermax;

    double dx2    = solver->dx * solver->dx;
    double dy2    = solver->dy * solver->dy;
    double idx2   = 1.0 / dx2;
    double idy2   = 1.0 / dy2;
    double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
    double* p     = solver->p;
    double* rhs   = solver->rhs;
    double epssq  = eps * eps;

    res = eps + 1.0;

    while ((res >= epssq) && (it < itermax)) {
        res = 0.0;
        exchange(solver);

        // for (int j = 1; j < jmaxLocal + 1; j++) {
        //     for (int i = 1; i < imax + 1; i++) {

        //         r = RHS_D(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
        //                             (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

        //         P(i, j) -= (factor * r);
        //         res += (r * r);
        //     }
        // }
        res_kernel<<<imax/256, 256>>>(solver, factor, &res);
        checkCudaError(cudaDeviceSynchronize(), true);

        // if (solver->rank != 0) {
        //     for (int i = 1; i < imax + 1; i++) {
        //         P(i, 0) = P(i, 1);                
        //     }
        // }

        // if (solver->rank != (solver->size - 1)) {
        //     for (int i = 1; i < imax + 1; i++) {
        //         P(i, jmaxLocal + 1) = P(i, jmaxLocal);
        //     }
        // }

        // for (int j = 1; j < jmaxLocal + 1; j++) {
        //     P(0, j)        = P(1, j);
        //     P(imax + 1, j) = P(imax, j);
        // }
        copyVertical<<<res/256, 256>>>(solver);
        copyHorizantal<<<res/256, 256>>>(solver);
        checkCudaError(cudaDeviceSynchronize(), true);
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

int solveRB(Solver* solver) {
    double r;
    int it = 0;
    double res, res1;

    int imax      = solver->imax;
    int jmax      = solver->jmax;
    int jmaxLocal = solver->jmaxLocal;
    double eps    = solver->eps;
    double omega  = solver->omega;
    int itermax   = solver->itermax;

    double dx2    = solver->dx * solver->dx;
    double dy2    = solver->dy * solver->dy;
    double idx2   = 1.0 / dx2;
    double idy2   = 1.0 / dy2;
    double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
    double* p     = solver->p;
    double* rhs   = solver->rhs;
    int pass, jsw, isw;
    double epssq = eps * eps;

    res = eps + 1.0;

    while ((res >= epssq) && (it < itermax)) {
        res = 0.0;
        jsw = 1;

        // for (pass = 0; pass < 2; pass++) {
        //     isw = jsw;
        //     exchange(solver);

        //     for (int j = 1; j < jmaxLocal + 1; j++) {
        //         for (int i = isw; i < imax + 1; i += 2) {

        //             double r = RHS_D(i, j) -
        //                        ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
        //                            (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

        //             P(i, j) -= (factor * r);
        //             res += (r * r);
        //         }
        //         isw = 3 - isw;
        //     }
        //     jsw = 3 - jsw;
        // }

        // for (int i = 1; i < imax + 1; i++) {
        //     P(i, 0)             = P(i, 1);
        //     P(i, jmaxLocal + 1) = P(i, jmaxLocal);
        // }

        // for (int j = 1; j < jmaxLocal + 1; j++) {
        //     P(0, j)        = P(1, j);
        //     P(imax + 1, j) = P(imax, j);
        // }
        // res_kernel<<<imax/256, 256>>>(solver, factor, res);
        MPI_Allreduce(&res, &res1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        res = res1;
        res = res / (double)(imax * jmax);
#ifdef DEBUG
        printf("%d Residuum: %e\n", it, res);
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

// int solveRBA(Solver* solver) {
//     double r;
//     int it = 0;
//     double res;

//     int imax      = solver->imax;
//     int jmax      = solver->jmax;
//     int jmaxLocal = solver->jmaxLocal;
//     double eps    = solver->eps;
//     double omega  = solver->omega;
//     int itermax   = solver->itermax;

//     double dx2    = solver->dx * solver->dx;
//     double dy2    = solver->dy * solver->dy;
//     double idx2   = 1.0 / dx2;
//     double idy2   = 1.0 / dy2;
//     double factor = omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
//     double* p     = solver->p;
//     double* rhs   = solver->rhs;
//     int pass, jsw, isw;
//     double rho   = solver->rho;
//     double epssq = eps * eps;

//     res = eps + 1.0;

//     while ((res >= epssq) && (it < itermax)) {
//         res = 0.0;
//         jsw = 1;

//         for (pass = 0; pass < 2; pass++) {
//             isw = jsw;
//             exchange(solver);

//             for (int j = 1; j < jmaxLocal + 1; j++) { // Offload to Kernel
//                 for (int i = isw; i < imax + 1; i += 2) {

//                     double r = RHS_D(i, j) -
//                                ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
//                                    (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

//                     P(i, j) -= (omega * factor * r);
//                     res += (r * r);
//                 }
//                 isw = 3 - isw;
//             }
//             jsw   = 3 - jsw;
//             omega = (it == 0 && pass == 0 ? 1.0 / (1.0 - 0.5 * rho * rho)
//                                           : 1.0 / (1.0 - 0.25 * rho * rho * omega));
//         }

//         for (int i = 1; i < imax + 1; i++) {
//             P(i, 0)             = P(i, 1);
//             P(i, jmaxLocal + 1) = P(i, jmaxLocal);
//         }

//         for (int j = 1; j < jmaxLocal + 1; j++) {
//             P(0, j)        = P(1, j);
//             P(imax + 1, j) = P(imax, j);
//         }

//         res = res / (double)(imax * jmax);
// #ifdef DEBUG
//         printf("%d Residuum: %e Omega: %e\n", it, res, omega);
// #endif
//         it++;
//     }

//     printf("Final omega: %f\n", omega);
//     printf("Solver took %d iterations to reach %f\n", it, sqrt(res));
// }

void writeResult(Solver* solver, double* m, char* filename) {
    int imax  = solver->imax;
    int jmax  = solver->jmax;
    double* p = solver->p;

    FILE* fp;
    fp = fopen(filename, "w");

    if (fp == NULL) {
        printf("Error!\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < jmax + 2; j++) {
        for (int i = 0; i < imax + 2; i++) {
            fprintf(fp, "%f ", m[j * (imax + 2) + i]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void finalize(Solver* solver) {
    MPI_Finalize();

    checkCudaError(cudaFreeHost(solver->p), true);
    checkCudaError(cudaFreeHost(solver->rhs), true);

    checkCudaError(cudaFree(solver->p_d), true);
    checkCudaError(cudaFree(solver->rhs_d), true);
}