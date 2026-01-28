/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "solver.h"

int main(int argc, char** argv) {
    int rank;
    Parameter params;
    Solver solver;

    initParameter(&params);

    if (argc != 2) {
        printf("Usage: %s <configFile>\n", argv[0]);
        exit(EXIT_SUCCESS);
    }

    readParameter(&params, argv[1]);
    if (rank == 0) printParameter(&params);

    initSolver(&argc, argv, &solver, &params, 2);
    // solve(&solver);
    // getResult(&solver, "result.dat");
    finalize();
    return EXIT_SUCCESS; 
}
