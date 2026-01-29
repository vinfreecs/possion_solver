#!/bin/bash -l 
source /etc/profile
module load intel likwid nvhpc openmpi cuda

nvcc -ccbin mpicc src/main.c src/parameter.c src/allocate.c src/affinity.c src/timing.c src/solver.cu -arch=sm_80 -o check -lm