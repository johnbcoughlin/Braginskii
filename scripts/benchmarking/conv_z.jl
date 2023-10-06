using CUDA
using BenchmarkTools
using Braginskii
using Bumper

function via_nnlib(f_in, out)
    CUDA.@sync Braginskii.convolve_z!(out, f_in, [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0], true, Braginskii.allocator(:gpu)) 
end

f_in = CUDA.rand(Float64, 16, 1, 66, 48, 1, 48); out = CUDA.zeros(Float64, 16, 1, 60, 48, 1, 48)
@descend via_nnlib(f_in, out)

bench = @benchmark via_nnlib(f_in, out) setup = (f_in = CUDA.rand(Float64, 16, 1, 66, 48, 1, 48); out = CUDA.zeros(Float64, 16, 1, 60, 48, 1, 48))
display(bench)

