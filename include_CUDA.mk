CC   = mpicc
NVCC = nvcc

# ifeq ($(ENABLE_OPENMP),true)
# OPENMP   = -fopenmp
#OPENMP   = -Xpreprocessor -fopenmp #required on Macos with homebrew libomp
# LIBS     = # -lomp 
LIBS = -lcudart
endif

VERSION  = --version
CFLAGS   = -Ofast -std=c99 $(OPENMP)
#CFLAGS   = -Ofast -fnt-store=aggressive  -std=c99 $(OPENMP) #AMD CLANG
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE# -DDEBUG
INCLUDES = -I/usr/local/include

# CUDA Path 
CUDA_PATH = /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.9.0-thcn547375nmyodcs5y4b7tuf6rz454p
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart

#OPENMPI PATH
OPENMPI_INCLUDE_PATH = /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-14.2.0/openmpi-5.0.5-6emtlk3wpfh7xgmcslo7p5fkl3g2nibx/include
