module RayleighTaylor2D2V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu, vdisc=:hermite)
    d = Dict{String, Any}()

    problem = "rayleigh_taylor"
    Nx = Nz = 48
    Nvx = 20
    Nvz = 60
    δ = 0.001
    q = 1.0
    k = 0.5
    merge!(d, @strdict problem Nx Nz Nvx Nvz δ q k)
    
    n0(x, z) = begin
        return (1.0 + 0.001 * sin(x)) * (1.1 + tanh(-z))
    end

    f_0(x, z, vx, vz) = begin
        return n0(x, z) / 2π * exp(-(vx^2 + vz^2) / 2)
    end

    By0 = 0.01

    sim = Helpers.single_species_xz_2d2v((; f_0, By0); Nx, Nz, Nvx, Nvz, 
        vdisc, ϕ_left=1.0, ϕ_right=1.0, device);

    d = PDEHarness.normalize!(d)
    return d, sim
end

end
