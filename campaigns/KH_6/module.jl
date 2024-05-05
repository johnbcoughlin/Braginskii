module KH6

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

function make_sim_hybrid(::Val{device}; ωcτ, eta, sz, kx=0.0, mag, ζ, just_setup=false) where {device}
    d = Dict{String, Any}()

    n0 = 1e22u"m^-3"
    ωpτ = ωcτ
    problem = "hybrid_kelvin_helmholtz"

    T = reference_temp()
    α = interface_width()

    params = KHShared.params(; T_ref=reference_temp(),
        n0, ωpτ, ωcτ, Ae=1/1836, Lz=1.2, α, n_ratio=0.2)

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
    if just_setup
        return (; d)
    end

    γ = 0.2
    p_ref = n_ref * T_ref

    #=
    tanh_profile(z) = (1.0 + γ * tanh(z / (α * Lz)))
    ni_hat(z) = n_ref * (1.0 + γ * tanh(z / (α * Lz)))
    pi_hat(z) = tanh_profile(z) * n_ref * T_ref
    tanh_profile_prime(z) = 1/(α * Lz) * sech(z/(α * Lz))^2

    ni0(z) = n_ref * tanh_profile(z)^ζ
    Ti0(z) = T_ref * tanh_profile(z)^(1-ζ)

    β = 2*α
    ûz = 0.05 * vti
    n1(x, z) = ûz * ωcτ * B_ref * n_ref / T_ref * sin(kx*x)/kx * exp(-z^2 / (2*(β*Lz)^2))
    dz_Ti0_n1(x, z) = n1(x, z) * (-(z/(β * Lz)^2) * Ti0(z) + T_ref * tanh_profile(z)^(-ζ) * (1 - ζ) * tanh_profile_prime(z))
    uix1(x, z) = 1 / ωcτ * dz_Ti0_n1(x, z) / (Zi * (ni0(z) + n1(x, z)) * B_ref)
    uiz1(x, z) = -ûz * (Ti0(z) * n_ref) / (T_ref * (ni0(z) + n1(x, z))) * cos(kx*x) * exp(-z^2 / (2*(β*Lz)^2))
    =#

    # Perturbation
    z0(x) = 2*α*Lz*sin(kx*x)
    z0_prime(x) = kx*2*α*Lz*cos(kx*x)
    wavy_profile(x, z) = (1.0 + γ * tanh((z - z0(x)) / (α * Lz)))

    # Ion density and temperature
    ni0(x, z) = n_ref * wavy_profile(x, z)^ζ
    Ti0(x, z) = T_ref * wavy_profile(x, z)^(1-ζ)

    # Electric field profile for shear
    width = 0.01
    u_s = 0.2 * vti
    phi_0 = -ωcτ * B_ref * u_s / ωpτ * sqrt(((Lz/2)^2 + (α*Lz)^2) / 4)
    phi_star(z) = phi_0 + ωcτ * B_ref * u_s / ωpτ * sqrt((z^2 + (α*Lz)^2)/4)

    # Electron density and temperature
    ne0(x, z) = ni0(x, z) + ωcτ * B_ref * u_s / ωpτ^2 * (1 / sqrt(z^2 + (α*Lz)^2) - z^2 / (z^2 + (α*Lz)^2)^(3/2))
    Te0(x, z) = pi0(x, z) / ne0(x, z)

    # Equilibrium ion diamagnetic velocity
    wavy_profile_dz(x, z) = γ/(α * Lz) * sech((z - z0(x)) / (α * Lz))^2
    wavy_profile_dx(x, z) = -z0_prime(x)*γ/(α * Lz) * sech((z - z0(x)) / (α * Lz))^2
    u_idx(x, z) = p_ref * wavy_profile_dz(x, z) / (ωcτ * Zi * ni0(x, z) * B_ref)
    u_idz(x, z) = -p_ref * wavy_profile_dx(x, z) / (ωcτ * Zi * ni0(x, z) * B_ref)

    # ExB velocity
    uEx(z) = ωcτ * B_ref * u_s / ωpτ * (z / (2*sqrt(z^2 + (α*Lz)^2)))
    uix0(x, z) = u_idx(x, z) + uEx(z)
    uiz0(x, z) = u_idz(x, z)

    fi_0(X, Z, VX, VZ) = begin
        ni = ni0.(X, Z)
        uix = uix0.(X, Z)
        uiz = uiz0.(X, Z)
        Ti = Ti0.(X, Z)
        return @. Ai * ni / (2π * Ti) * exp(-(Ai * ((VX-uix)^2 + (VZ-uiz)^2)) / (2*Ti))
    end
    Fe_0(x, z, μ) = ne0(x, z) * exp(-μ / μ0)
    By0(args...) = B_ref

    @info "Setting up sim"
    sim = Helpers.two_species_2d_vlasov_dk_hybrid(Val(device), (; Fe_0, fi_0=Braginskii.BatchFunc(fi_0), By0);
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
