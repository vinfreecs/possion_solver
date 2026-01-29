CC   = nvcc
GCC  = mpicc
LINKER = $(GCC)

VERSION  = --version
CFLAGS   = -arch=sm_86 -lm -O3
LFLAGS   = -L${CUDA_HOME}/lib64 -lcudart -lm -ldl
DEFINES  = -D_GNU_SOURCE# -DDEBUG
INCLUDES = -I${OPENMPI_ROOT}/include
