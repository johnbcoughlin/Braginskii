module Braginskii

using LoopVectorization
using OffsetArrays: Origin, OffsetArray
using LinearAlgebra
using StrideArrays
using StrideArraysCore
using Bumper
using FFTW
using FastBroadcast
using RecursiveArrayTools

StrideArraysCore.boundscheck() = true

export runsim_lightweight!, weno_interpolate!

include("rk.jl")
include("simulation.jl")
include("free_streaming.jl")

include("helpers.jl")

include("grid.jl")

end
