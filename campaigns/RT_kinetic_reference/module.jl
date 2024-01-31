module RTKineticReference

using DrWatson
@quickactivate :Braginskii
using Unitful
using Braginskii.Helpers

include("../RT_shared/module.jl")

function make_sim_vlasov(::Val{device}; pt, Ae=1/25) where {device}
    ex = RTShared.kinetic_examples()[pt]
    opts, octs = RTShared.opt_oct_values()
    ωpτ = opts[ex]
    ωcτ = octs[ex]
    n0 = 1e22u"m^-3"
    
    params = RTShared.params(; n0, ωpτ, ωcτ, Ae)
    (; norm, Lx, Lz, Ai, Ae, Zi, Ze, vti, vte, gz, kx, δ, By0, ωg) = params
    @assert gz < 0
    νpτ = norm["νpτ"]
    @show νpτ

    display(norm)

    (; fe_eq, fi_eq, ne_eq, ni_eq) = RTShared.construct_vlasov_eq(params)
    fi_0(_, z, vx, vz) = fi_eq(z, vx, vz)
    theta(x) = 2pi*kx*x/Lx
    perturbation(x, z) = 1 + δ*exp(-z^4/0.01) * cos(theta(x))
    fe_0(x, z, vx, vz) = fe_eq(z, vx, vz) * perturbation(x, z)
    
    Nx = 96
    Nz = 200
    Nvx = 20
    Nvz = 20

    sim = Helpers.two_species_xz_2d2v(Val(device),
        (; fe_0, fi_0, By0);
        Nx, Nz, Nvx, Nvz,
        vdisc=:hermite,
        zmin=-Lz/2, zmax=Lz/2, Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth_i=vti, vth_e=vte,
        νpτ, ωpτ, ωcτ, Ze, Zi, Ae, Ai,
        gz, z_bcs=:reservoir);

    problem="rayleigh_taylor_kinetic_$id"
    d = Dict{String, Any}()
    merge!(d, @strdict problem Nx Nz Nvx Nvz Ae gz δ ωpτ ωcτ νpτ ωg)

    return (; d, sim)
end

end
