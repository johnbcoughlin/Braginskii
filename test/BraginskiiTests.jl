module BraginskiiTests

using LinearAlgebra
using ReTest
using Braginskii
using Braginskii.Helpers
using Bumper
using FFTW
using CUDA
using SparseArrays

supported_devices() = begin
    #CUDA.functional() ? (:cpu, :gpu) : (:cpu,)
    return (:cpu,)
end

include("test_utils.jl")
include("alloc_test.jl")
include("hermite_tests.jl")
include("laguerre_tests.jl")
include("free_streaming_tests.jl")
include("electrostatic_tests.jl")
include("poisson_tests.jl")
include("fourier_tests.jl")
include("landau_damping_test.jl")
include("dfp_test.jl")

end
