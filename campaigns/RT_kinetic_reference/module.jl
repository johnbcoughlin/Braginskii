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
    (; norm, Lx, Lz, Ai, Ae, Zi, Ze, vti, vte, gz, kx, δ, By0, ωg, α) = params
    @assert gz < 0
    νpτ = norm["νpτ"]
    display(norm)

    (; fe_eq, fi_eq, ne_eq, ni_eq) = RTShared.construct_vlasov_eq(params)
    theta(x) = 2pi*kx*x/Lx
    perturbation_x(x) = α^2*z*exp(-(z/(α/2))^2)
    perturbation_z(z) = δ * cos(theta(x))
    
    Nx = 96
    Nz = 200
    Nvx = 16
    Nvz = 16
    zmin = -Lz/2
    zmax = Lz/2
    x_grid = Helpers.z_grid_1d(Nz, zmin, zmax, allocator(device))
    (; fe_moments, fi_moments) = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, perturbation_x, perturbation_z,
        x_grid.X, x_grid.Z, Nvx, Nvz, vte, vti)

    dz = x_grid.z.dx
    left_grid = Helpers.z_grid_1d(3, zmin-3dz, zmin, allocator(device))
    fe_left, fi_left = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, perturbation_x, perturbation_z,
        left_grid.X, left_grid.Z, Nvx, Nvz, vte, vti) |> values
    right_grid = Helpers.z_grid_1d(3, zmax, zmax+3dz, allocator(device))
    fe_right, fi_right = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, perturbation_x, perturbation_z,
        right_grid.X, right_grid.Z, Nvx, Nvz, vte, vti) |> values

    sim = Helpers.two_species_xz_2d2v(Val(device),
        (; fe_0=nothing, fi_0=nothing, By0);
        Nz, Nvx, Nvz,
        vdisc=:hermite,
        zmin, zmax,
        ϕ_left=0.0, ϕ_right=0.0, vth_i=vti, vth_e=vte,
        νpτ=0.0, ωpτ, ωcτ, Ze, Zi, Ae, Ai,
        fe_ic=fe_moments, fi_ic=fi_moments,
        gz, 
        z_bcs=:reservoir,
        ion_bc_lr=(fi_left, fi_right), electron_bc_lr=(fe_left, fe_right)
        );



    problem="rayleigh_taylor_kinetic_$pt"
    d = Dict{String, Any}()
    merge!(d, @strdict problem Nx Nz Nvx Nvz Ae gz δ ωpτ ωcτ νpτ ωg)

    return (; d, sim)
end

end
