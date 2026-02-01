#!/bin/bash -l
#SBATCH --gres=gpu:a40:8
#SBATCH --time=01:30:00
#SBATCH --output=job_output_%j.log
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Load modules
module load nvhpc cuda openmpi

# Compile
make clean && make

echo "Starting Multi-Domain Benchmark on A40"

# Outer Loop: Domain Sizes (10k, 5k, 2k)
for domain in 10k 5k 2k
do
    # Define a unique filename for this domain
    RESULT_FILE="results_${domain}.txt"
    
    # Clear/Create the file for this domain
    echo "Benchmark Results for Domain: $domain" > $RESULT_FILE
    echo "========================================" >> $RESULT_FILE

    echo "Processing Domain: $domain (Saving to $RESULT_FILE)"

    # Inner Loop: 1 to 8 GPUs
    for t in {1..8}
    do
        # 'tee -a' shows the message on screen AND saves it to the file
        echo "Running $domain with $t GPUs..." | tee -a $RESULT_FILE
        
        # Run MPI and append (>>) output to the specific file
        mpirun -n $t ./exe-CUDA poisson_${domain}.par >> $RESULT_FILE
        
        echo "--------------------------------" >> $RESULT_FILE
    done
done

echo "All Done."