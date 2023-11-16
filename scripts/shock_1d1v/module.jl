module Shock1D1V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu, vdisc=:hermite)
    d = Dict{String, Any}()

    problem = "shock_1d1v"
    Nz = 128
    Nvx = 10
    Nvz = 30
    n_bg = 0.2

    merge!(d, @strdict problem Nz Nvx Nvz n_bg)

    #n0(z) = n_bg + (1 - n_bg) * (abs(z) < 0.3)
    n0(z) = 1.0
    #T0(z) = 1.0
    T0(z) = 0.5 + exp(-(z)^2 / 0.04)
    f0(z, vz) = n0(z) / sqrt(2π * T0(z)) * exp(-(vz^2) / (2T0(z)))

    sim = single_species_1d1v_z(f0; Nz, Nvz, vdisc=:hermite, q=0.0, ν_p=400.0)

    d = PDEHarness.normalize!(d)

    return d, sim
end

end
