module RTMagnetization3

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Interpolations
using Unitful

include("../RT_shared/module.jl")

function reference_temp()
    return 5e-4
end

function gravity()
    return -1.5e-4
end

function make_sim_vlasov(::Val{device}; ωcτ, Ae) where {device}
    d = Dict{String, Any}()

    n0 = 1e22u"m^-3"
    ωpτ = ωcτ

    problem = "hybrid_rayleigh_taylor"
    params = RTShared.params(; T_ref=reference_temp(), gz=gravity(), 
        n0, ωpτ, ωcτ, Ae, Lz=1.5)

    (; norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ, ωg,
        kx, gz, δ,
        ωg, By0, gi, Pxi, Pxe, Wi, We, ϕ_star
        ) = params

    (; fe_eq, fi_eq, ne_eq) = RTShared.construct_vlasov_eq(params)
    
    Nμ = 3
    μ0 = T_ref / B_ref

    η = sqrt(ωcτ) / sqrt(.86) * .18

    Nx = 72
    Nz = 192
    Nvx = 16
    Nvz = 16
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)
    x_grid = Helpers.xz_grid_2d(Nx, Nz, zmin, zmax, Lx, buffer)
    @show device

    theta(x) = 2pi*kx*x/Lx
    perturbation_x(x) = cos(theta(x))
    perturbation_z(z) = δ * exp(-z^2/0.01)

    ion_perts = (Returns(0.0), Returns(0.0))
    electron_perts = (perturbation_x, perturbation_z)

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref kx δ gz ωg)

    (; fe_moments, fi_moments) = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, ion_perts..., electron_perts...,
        x_grid.X, x_grid.Z, Nvx, Nvz, vte, vti)

    dz = x_grid.z.dx
    left_grid = Helpers.xz_grid_2d(Nx, 3, zmin-3dz, zmin, Lx, buffer)
    fe_left, fi_left = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, ion_perts..., electron_perts...,
        left_grid.X, left_grid.Z, Nvx, Nvz, vte, vti)

    right_grid = Helpers.xz_grid_2d(Nx, 3, zmax, zmax+3dz, Lx, buffer)
    fe_right, fi_right = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, ion_perts..., electron_perts...,
        right_grid.X, right_grid.Z, Nvx, Nvz, vte, vti)

    @info "Setting up sim"
    sim = Helpers.two_species_xz_2d2v(Val(device),
        (; fe_0=nothing, fi_0=nothing, By0);
        Nx, Nz, Nvx, Nvz,
        vdisc=:hermite,
        zmin, zmax,
        ϕ_left=0.0, ϕ_right=0.0, vth_i=vti, vth_e=vte,
        νpτ=0.0, ωpτ, ωcτ, Ze, Zi, Ae, Ai,
        grid_scale_hyperdiffusion_coef=η,
        fe_ic=fe_moments, fi_ic=fi_moments,
        gz, 
        z_bcs=:reservoir,
        ion_bc_lr=(fi_left, fi_right), electron_bc_lr=(fe_left, fe_right)
        );
    @info "Done setting up sim"

    return (; d, sim)
end

end
