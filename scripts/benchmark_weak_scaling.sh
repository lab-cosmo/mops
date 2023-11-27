#!/bin/bash
#SBATCH --chdir /home/bigi/mops/scripts/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72
#SBATCH --mem 480G
#SBATCH --time 1-0 

echo STARTING AT `date`
cd ../build/benchmarks
OMP_NUM_THREADS=1 ./hpe 100
OMP_NUM_THREADS=2 ./hpe 200
OMP_NUM_THREADS=4 ./hpe 400
OMP_NUM_THREADS=8 ./hpe 800
OMP_NUM_THREADS=16 ./hpe 1600
OMP_NUM_THREADS=32 ./hpe 3200
echo ""
OMP_NUM_THREADS=1 ./opsa 100
OMP_NUM_THREADS=2 ./opsa 200
OMP_NUM_THREADS=4 ./opsa 400
OMP_NUM_THREADS=8 ./opsa 800
OMP_NUM_THREADS=16 ./opsa 1600
OMP_NUM_THREADS=32 ./opsa 3200
echo ""
OMP_NUM_THREADS=1 ./sap 100
OMP_NUM_THREADS=2 ./sap 200
OMP_NUM_THREADS=4 ./sap 400
OMP_NUM_THREADS=8 ./sap 800
OMP_NUM_THREADS=16 ./sap 1600
OMP_NUM_THREADS=32 ./sap 3200
echo ""
OMP_NUM_THREADS=1 ./opsaw 100
OMP_NUM_THREADS=2 ./opsaw 200
OMP_NUM_THREADS=4 ./opsaw 400
OMP_NUM_THREADS=8 ./opsaw 800
OMP_NUM_THREADS=16 ./opsaw 1600
OMP_NUM_THREADS=32 ./opsaw 3200
echo ""
OMP_NUM_THREADS=1 ./sasaw 100
OMP_NUM_THREADS=2 ./sasaw 200
OMP_NUM_THREADS=4 ./sasaw 400
OMP_NUM_THREADS=8 ./sasaw 800
OMP_NUM_THREADS=16 ./sasaw 1600
OMP_NUM_THREADS=32 ./sasaw 3200
echo FINISHED at `date`
