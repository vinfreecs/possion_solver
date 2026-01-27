CC  = nvcc
LINKER = $(CC)

ifeq ($(strip $(ENABLE_MPI)),true)
DEFINES += -D_MPI
MPI_LIB  = -L/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/nvhpc-23.7/openmpi-4.1.6-ojusv6lrh7e5o7ktibh2qaj2yuzxyzeg/lib \
           -L/usr/lib64 \
           -Xlinker -rpath -Xlinker /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/nvhpc-23.7/openmpi-4.1.6-ojusv6lrh7e5o7ktibh2qaj2yuzxyzeg/lib \
           -lmpi
MPI_HOME = -I/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/nvhpc-23.7/openmpi-4.1.6-ojusv6lrh7e5o7ktibh2qaj2yuzxyzeg/include
endif

ANSI_CFLAGS  = -ansi
ANSI_CFLAGS += -std=c99
ANSI_CFLAGS += -pedantic
ANSI_CFLAGS += -Wextra

#
# A100 + Native
# CFLAGS   = -O3 -arch=sm_80 -march=native -ffast-math -funroll-loops --forward-unknown-to-host-compiler # -fopenmp
# A40 + Native
CFLAGS   = -O3 -arch=sm_86 -march=native -ffast-math -funroll-loops --forward-unknown-to-host-compiler # -fopenmp
# Cascade Lake
#CFLAGS   = -O3 -march=cascadelake  -ffast-math -funroll-loops --forward-unknown-to-host-compiler # -fopenmp
# For GROMACS kernels, we need at least sm_61 due to atomicAdd with doubles
# TODO: Check if this is required for full neighbor-lists and just compile kernel for that case if not
#CFLAGS   = -O3 -g -arch=sm_61 # -fopenmp
CFLAGS   += $(OPTS)
# ASFLAGS  =  -masm=intel
LFLAGS   =
DEFINES  += -D_GNU_SOURCE -DCUDA_TARGET=0 -DNO_ZMM_INTRIN  #-DLIKWID_PERFMON
INCLUDES = $(MPI_HOME) $(LIKWID_INC)
LIBS     = -lm -lcuda -lcudart $(LIKWID_LIB) $(MPI_LIB)#-llikwid

