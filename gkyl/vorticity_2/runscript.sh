#!/bin/bash

export LD_LIBRARY_PATH=$SCRATCH/gkylsoft/OpenBLAS/lib:$LD_LIBRARY_PATH

pushd SF-$1
mpirun -np 16 ../incompressible $1
popd
