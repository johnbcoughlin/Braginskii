module LandauDamping1D1V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu)
    d = Dict{String, Any}()

    problem = "landau_damping"
    Ny = 24
    Nvy = 100
    δ = 0.001
    q = 1.0
    k = 0.5
    merge!(d, @strdict problem Ny Nvy δ q k)

    sim = single_species_1d1v_y(; Ny, Nvy, Ly=2π/k, vdisc=:weno, q, device) do y, vy
        1 / sqrt(2π) * (1.0 + δ * cos(k*y)) * exp(-vy^2/2)
    end

    d = PDEHarness.normalize!(d)
    return d, sim
end

end
