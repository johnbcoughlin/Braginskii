module Braginskii

using LoopVectorization
using OffsetArrays: OffsetArrays, Origin, OffsetArray
using LinearAlgebra
using StrideArrays
using StrideArraysCore
using Bumper
using FFTW
using FastBroadcast
using RecursiveArrayTools
using TimerOutputs
using NNlib
using StaticArrays

StrideArraysCore.boundscheck() = true

export runsim_lightweight!, weno_interpolate!

include("rk.jl")
include("simulation.jl")

include("convolve.jl")
include("free_streaming.jl")
include("electrostatic.jl")
include("poisson.jl")
include("fourier.jl")

include("helpers.jl")

include("grid.jl")

end
