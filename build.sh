#!/bin/bash -l 
source /etc/profile
module load intel likwid nvhpc openmpi cuda
module list

# Compile each source file separately
# nvcc -ccbin mpicc -c src/main.c -o NVCC/main.o
# nvcc -ccbin mpicc -c src/parameter.c -o NVCC/parameter.o
# nvcc -ccbin mpicc -c src/allocate.c -o NVCC/allocate.o
# nvcc -ccbin mpicc -c src/affinity.c -o NVCC/affinity.o
# nvcc -ccbin mpicc -c src/timing.c -o NVCC/timing.o
# nvcc -ccbin mpicc -c src/solver.cu -arch=sm_80 -o NVCC/solver.o

# nvcc -ccbin mpicc -arch=sm_80 NVCC/main.o NVCC/parameter.o NVCC/allocate.o NVCC/affinity.o NVCC/timing.o NVCC/solver.o -o exe-NVCC -lm
nsys profile --trace=mpi,cuda --mpi-impl=openmpi -o a40-$1 \
mpirun -n $1 --report-bindings ./exe-NVCC poisson.par