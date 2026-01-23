# GPU Acceleration through 1D Domain Decomposition of Poisson Solver 

## Build

1. Configure the tool chain and additional options in `config.mk`:
```
# Supported: GCC, CLANG, ICC
TAG ?= GCC
ENABLE_OPENMP ?= false

OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
```

The verbosity options enable detailed output about affinity settings, allocation sizes and timer resolution.


2. Build with:
```
make
```

You can build multiple tool chains in the same directory, but notice that the Makefile is only acting on the one currently set.
Intermediate build results are located in the `<TOOLCHAIN>` directory.

To output the executed commands use:
```
make Q=
```

3. Clean up with:
```
make clean
```
to clean intermediate build results.

```
make distclean
```
to clean intermediate build results and binary.

4. (Optional) Generate assembler:
```
make asm
```
The assembler files will also be located in the `<TOOLCHAIN>` directory.

Tasks:
1. Update Kernels for Computation.
2. Allocate Memory using cudaMalloc