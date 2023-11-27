#!/bin/bash
#SBATCH --chdir /home/bigi/mops/scripts/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72
#SBATCH --mem 480G
#SBATCH --time 1-0 

echo STARTING AT `date`
cd ../build/benchmarks
OMP_NUM_THREADS=1 ./hpe
OMP_NUM_THREADS=2 ./hpe
OMP_NUM_THREADS=4 ./hpe
OMP_NUM_THREADS=8 ./hpe
OMP_NUM_THREADS=16 ./hpe
OMP_NUM_THREADS=32 ./hpe
echo ""
OMP_NUM_THREADS=1 ./opsa
OMP_NUM_THREADS=2 ./opsa
OMP_NUM_THREADS=4 ./opsa
OMP_NUM_THREADS=8 ./opsa
OMP_NUM_THREADS=16 ./opsa
OMP_NUM_THREADS=32 ./opsa
echo ""
OMP_NUM_THREADS=1 ./sap
OMP_NUM_THREADS=2 ./sap
OMP_NUM_THREADS=4 ./sap
OMP_NUM_THREADS=8 ./sap
OMP_NUM_THREADS=16 ./sap
OMP_NUM_THREADS=32 ./sap
echo ""
OMP_NUM_THREADS=1 ./opsaw
OMP_NUM_THREADS=2 ./opsaw
OMP_NUM_THREADS=4 ./opsaw
OMP_NUM_THREADS=8 ./opsaw
OMP_NUM_THREADS=16 ./opsaw
OMP_NUM_THREADS=32 ./opsaw
echo ""
OMP_NUM_THREADS=1 ./sasaw
OMP_NUM_THREADS=2 ./sasaw
OMP_NUM_THREADS=4 ./sasaw
OMP_NUM_THREADS=8 ./sasaw
OMP_NUM_THREADS=16 ./sasaw
OMP_NUM_THREADS=32 ./sasaw
echo FINISHED at `date`
