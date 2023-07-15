module BraginskiiTests

using LinearAlgebra
using ReTest
using Braginskii
using Braginskii.Helpers
using Bumper
using FFTW


set_default_buffer_size!(100_000_000)

include("test_utils.jl")
include("alloc_test.jl")
include("hermite_tests.jl")
include("free_streaming_tests.jl")
include("electrostatic_tests.jl")
include("poisson_tests.jl")
include("fourier_tests.jl")
include("landau_damping_test.jl")
include("dfp_test.jl")

end
