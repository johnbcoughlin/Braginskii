#!/bin/bash

#SBATCH --job-name=RT-kinetic-reference

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=1
#SBATCH --array=1-@@NTASKS@@%3
#SBATCH --mem=20G
#SBATCH --time=24:00:00

#SBATCH --chdir=@@WORKDIR@@
#SBATCH --output=@@WORKDIR@@/sims/RT-%a/sim.log
#SBATCH --error=@@WORKDIR@@/sims/RT-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
