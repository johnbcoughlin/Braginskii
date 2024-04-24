#!/bin/bash

#SBATCH --job-name=gkyl-vorticity

#SBATCH --account=aaplasma
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH --chdir=/mmfs1/gscratch/amath/johnbc/vorticity_2
#SBATCH --output=/mmfs1/gscratch/amath/johnbc/vorticity_2/SF-%a/sim.log
#SBATCH --error=/mmfs1/gscratch/amath/johnbc/vorticity_2/SF-%a/sim.err

pushd SF-$SLURM_ARRAY_TASK_ID
export LD_LIBRARY_PATH=$SCRATCH/gkylsoft/OpenBLAS/lib:$LD_LIBRARY_PATH
mpirun -np 8 /mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/gkyl/vorticity_2/incompressible $SLURM_ARRAY_TASK_ID
