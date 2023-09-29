module Braginskii

using LoopVectorization
using OffsetArrays: OffsetArrays, Origin, OffsetArray
using LinearAlgebra
using Bumper
using FFTW: FFTW
using FastBroadcast
using RecursiveArrayTools
using TimerOutputs
using StaticArrays
using SparseArrays
using IterativeSolvers
using DataFrames
using PDEHarness
using ProgressMeter
using FastGaussQuadrature
using CUDA
using cuDNN
using NNlib
using NNlibCUDA

import Base.size, Base.getproperty

export runsim_lightweight!, weno_interpolate!

export alloc_array, alloc_zeros

include("alloc.jl")

include("rk.jl")

include("discretizations.jl")

include("simulation.jl")
include("convolve.jl")
include("free_streaming.jl")
include("electrostatic.jl")
include("poisson.jl")
include("dfp.jl")
include("fourier.jl")
include("moments.jl")
include("diagnostics.jl")

include("hermite_utils.jl")

include("helpers.jl")

include("grid.jl")
include("dispersion_analysis.jl")

end
