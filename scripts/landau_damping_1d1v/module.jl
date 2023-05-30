module LandauDamping1D1V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim()
    d = Dict{String, Any}()

    problem = "landau_damping"
    Ny = 24
    Nvy = 100
    δ = 0.001
    q = 1.5
    k = 1.0
    merge!(d, @strdict problem Ny Nvy δ q k)

    sim = single_species_1d1v_y(Ny, Nvy, 2π/k; q) do y, vy
        1 / sqrt(2π) * (1.0 + δ * cos(k*y)) * exp(-vy^2/2)
    end

    d = PDEHarness.normalize!(d)
    return d, sim
end

end
