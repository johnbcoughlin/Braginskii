module KH3

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Unitful

include("../KH_shared/module.jl")

function reference_temp()
    return 1e-3
end

function interface_width()
    return 0.04
end

function make_sim_hybrid(::Val{device}; ωcτ, eta, sz, kx=0.0, mag, just_setup=false) where {device}
    d = Dict{String, Any}()

    n0 = 1e22u"m^-3"
    ωpτ = ωcτ
    problem = "hybrid_kelvin_helmholtz"

    T = reference_temp()
    α = interface_width()

    params = KHShared.params(; T_ref=reference_temp(),
        n0, ωpτ, ωcτ, Ae=1/1836, Lz=1.2, α)

    (; norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ,
        By0, gi, Pxi, Pxe, Wi, We, ϕ_star
        ) = params

    (; fi_eq, ni_eq, ne_eq) = KHShared.construct_vlasov_hybrid_eq(params)

    Nμ = 3
    μ0 = T / B_ref

    η = sqrt(ωcτ) * eta

    Nx = [32, 96, 120, 144][sz]
    Nz = [100, 216, 248, 280][sz]
    Nvx = [16, 22, 24, 26][sz]
    Nvz = [16, 22, 24, 26][sz]
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)
    @show device

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref kx mag)

    # Charge density perturbation is vti * mag * kx sin(kx*x)
    # So electric field perturbation is -vti * mag * cos(kx*x),
    # corresponding ExB z velocity perturbation is -vti * mag * cos(kx*x)
    n_perturbation(x) = 0.5*mag * vti * kx * sin(kx*x)
    perturbation(x, z) = n_perturbation(x) * exp(-z^2/(2*α^2))

    fi_0(x, z, vx, vz) = (1.0 + perturbation(x, z)) * fi_eq(z, vx, vz)
    Fe_0(x, z, μ) = (1.0 - perturbation(x, z)) * ne_eq(z) * exp(-μ / μ0)
    By0(args...) = B_ref

    @info "Setting up sim"
    sim = Helpers.two_species_2d_vlasov_dk_hybrid(Val(device), (; Fe_0, fi_0, By0);
        Nx, Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth=vti,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai,
        grid_scale_hyperdiffusion_coef=η,
        gz=0.0, z_bcs=:reservoir)
    @info "Done setting up sim"

    return (; d, sim, fi_eq, ne_eq)
end

function make_1d_sim_hybrid(::Val{device}; ωcτ, eta, sz, just_setup=false) where {device}
    d = Dict{String, Any}()

    n0 = 1e22u"m^-3"
    ωpτ = ωcτ
    problem = "hybrid_kelvin_helmholtz"

    T = reference_temp()
    α = interface_width()

    params = KHShared.params(; T_ref=reference_temp(),
        n0, ωpτ, ωcτ, Ae=1/1836, Lz=1.2, α)

    (; norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ,
        kx,
        By0, gi, Pxi, Pxe, Wi, We, ϕ_star
        ) = params

    (; fi_eq, ni_eq, ne_eq) = KHShared.construct_vlasov_hybrid_eq(params)

    Nμ = 3
    μ0 = T / B_ref

    η = sqrt(ωcτ) * eta

    Nz = [100, 216, 248, 280][sz]
    Nvx = [16, 22, 24, 26][sz]
    Nvz = [16, 22, 24, 26][sz]
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)
    @show device

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref kx)

    fi_0 = fi_eq
    Fe_0(z, μ) = ne_eq(z) * exp(-μ / μ0)
    By0(args...) = B_ref

    @info "Setting up sim"
    sim = Helpers.two_species_1d2v_vlasov_dk_hybrid((; Fe_0, fi_0, By0);
        Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, 
        ϕ_left=0.0, ϕ_right=0.0, vti, vte,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai,
        grid_scale_hyperdiffusion_coef=η,
        gz=0.0, z_bcs=:reservoir)
    @info "Done setting up sim"

    return (; d, sim, fi_eq, ne_eq)
end

end
