module BraginskiiTests

using LinearAlgebra
using ReTest
using Braginskii
using Braginskii.Helpers
using Bumper

include("test_utils.jl")
include("free_streaming_tests.jl")
include("electrostatic_tests.jl")
include("poisson_tests.jl")

end
