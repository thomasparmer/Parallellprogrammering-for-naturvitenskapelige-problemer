#!/bin/bash
#SBATCH --job-name=image_denoising
#SBATCH --account=ln0001k
# Wall clock limit:
#SBATCH --time=’00:05:00’
# Number of MPI processes:
#SBATCH --ntasks=512
# Max memory usage per MPI process:
#SBATCH --mem-per-cpu=100m

mpirun -np 512 ./parallel_main 10000 0.1 mona_lisa_noisy.jpg out.jpg
