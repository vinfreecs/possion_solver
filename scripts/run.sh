#!/bin/bash -l

make clean && make

RESULT_FILE="a100_2k_res.txt"

echo "Starting Benchmark on A100 (2k x 2k)"

# Create/Clear the specific results file
echo "Running Benchmarks..." > $RESULT_FILE

# Loop from 1 to 8 GPUs
for t in {1..8}
do
    echo "Running with $t GPUs..." | tee -a $RESULT_FILE
    
    # FIX: Use $t or ${t}, NOT $(t)
    # The '>>' appends the output to the file instead of printing to screen
    mpirun -n $t ./exe-CUDA poisson.par >> $RESULT_FILE
    
    echo "--------------------------------" >> $RESULT_FILE
done

echo "Done. Results saved in $RESULT_FILE"