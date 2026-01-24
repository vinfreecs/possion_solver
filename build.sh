#!/bin/bash -l 
module load intel likwid nvhpc openmpi cuda
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/main.c main.o 
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/cuda_solver.cu cuda_solver.o -arch=sm_86
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/parameter.c parameter.o
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/allocate.c allocate.o
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/affinity.c affinity.o
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/timing.c timing.o
# nvcc -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include -c src/solver.c solver.o
# mpicc main.o cuda_solver.o parameter.o affinity.o allocate.o solver.o timing.o -o check -L/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.9.0-thcn547375nmyodcs5y4b7tuf6rz454p/lib64 -lcudart -lm

# nvcc -I/${OPENMPI_ROOT}/include -c src/main.c main.o 
# nvcc -I/${OPENMPI_ROOT}/include -c src/cuda_solver.cu cuda_solver.o -arch=sm_86
# nvcc -I/${OPENMPI_ROOT}/include -c src/parameter.c parameter.o
# nvcc -I/${OPENMPI_ROOT}/include -c src/allocate.c allocate.o
# nvcc -I/${OPENMPI_ROOT}/include -c src/affinity.c affinity.o
# nvcc -I/${OPENMPI_ROOT}/include -c src/timing.c timing.o
# nvcc -I/${OPENMPI_ROOT}/include -c src/solver.c solver.o
# mpicc main.o cuda_solver.o parameter.o affinity.o allocate.o solver.o timing.o -o check -L/${CUDA_HOME}/lib64 -lcudart -lm

nvcc -ccbin mpicc src/main.c src/cuda_solver.cu src/parameter.c src/allocate.c src/affinity.c src/timing.c src/solver.c -arch=sm_86 -o check -lm
