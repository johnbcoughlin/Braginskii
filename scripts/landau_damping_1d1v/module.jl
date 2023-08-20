module LandauDamping1D1V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu)
    d = Dict{String, Any}()

    problem = "landau_damping"
    Nx = 48
    Nvx = 200
    δ = 0.001
    q = 1.0
    k = 0.5
    merge!(d, @strdict problem Nx Nvx δ q k)

    sim = single_species_1d1v_x(; Nx, Nvx, Lx=2π/k, vdisc=:weno, q, device) do x, vx
        1 / sqrt(2π) * (1.0 + δ * cos(k*x)) * exp(-vx^2/2)
    end

    d = PDEHarness.normalize!(d)
    return d, sim
end

end
