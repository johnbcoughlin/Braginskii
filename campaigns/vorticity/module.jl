module Vorticity

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Unitful

function simulations()
    u_V_fac = 0.1
    γ = 0.25
    sims = [
        (; ωcτ=0.5, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=1.0, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=2.0, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=2.0, ζ=0.5, u_s_fac=-0.2, u_V_fac, γ),
        (; ωcτ=2.0, ζ=1.0, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=2.0, ζ=1.5, u_s_fac=0.2, u_V_fac, γ),
    ]
    return sims
end

function params(; ωcτ, γ, u_s_fac, u_V_fac)
    Ai = 1.0
    Ae = 1 / 1836
    Zi = 1.0
    Ze = -1.0
    ωpτ = ωcτ
    n_ref = 1.0
    T_ref = 1e-3
    α = 0.04
    ζ = 1.0
    B = 1.0

    vti = sqrt(T_ref / Ai)

    w = 2*α

    u_s = u_s_fac*vti
    u_V = u_V_fac*vti

    kx = 2pi

    return (; Ai, Ae, Zi, Ze, ωpτ, ωcτ, n_ref, T_ref, B, α, γ, ζ, w, u_s, u_V, kx, vti)
end

function profiles(; ωcτ, γ=0.2, u_s_fac=0.2, u_V_fac=0.1)
    (; 
    Ai, Ae, Zi, Ze, 
    ωpτ, ωcτ, n_ref, T_ref, 
    B, α, γ, ζ, 
    w, u_s, u_V, kx) = params(; ωcτ, γ, u_s_fac, u_V_fac)

    p_ref = n_ref * T_ref

    phat(z) = 1 + γ * tanh(z / α)
    ni0(z) = n_ref * phat(z)^ζ
    Ti0(z) = T_ref * phat(z)^(1 - ζ)

    phi_Z(z) = 1 + u_s * α / 2 * log(cosh(z / α))
    phi_X(x, z) = 1 + u_V / kx * sin(kx*x) * exp(-z^2 / (2w^2))

    phistar(x, z) = phi_Z(z) * phi_X(x, z)

    Ex(x, z) = begin
        h = 1e-6
        -(phistar(x + h, z) - phistar(x - h, z)) / (2.0*h)
    end
    Ez(x, z) = begin
        h = 1e-6
        -(phistar(x, z+h) - phistar(x, z-h)) / (2.0*h)
    end
    rhoc(x, z) = begin
        h = 1e-4
        L = phistar(x+h, z) + phistar(x-h, z) + phistar(x, z+h) + phistar(x, z-h) - 4phistar(x, z)
        -(L / h^2) / ωpτ
    end
    ne0(x, z) = begin
        (rhoc(x, z) - Zi*ni0(z)) / Ze
    end

    pi_z(z) = begin
        h = 1e-6
        p_ref * (phat(z + h) - phat(z - h)) / (2h)
    end
    uEx(x, z) = -ωpτ / ωcτ * Ez(x, z) / B
    uEz(x, z) = ωpτ / ωcτ * Ex(x, z) / B
    udx(z) = 1/ωcτ * pi_z(z) / B

    return (; phat, ni0, Ti0, ne0, Ex, Ez, rhoc, uEx, uEz, udx)
end

function make_sim_hybrid(::Val{device}; ωcτ, γ=0.2, u_s_fac=0.2, u_V_fac=0.1, sz=4, just_setup=false) where {device}
    (; 
    Ai, Ae, Zi, Ze, 
    ωpτ, ωcτ, n_ref, T_ref, 
    B, α, γ, ζ, 
    w, u_s, u_V, kx, vti) = params(; ωcτ, γ, u_s_fac, u_V_fac)

    (; phat, ni0, Ti0, ne0, Ex, Ez, rhoc, uEx, uEz, udx) = profiles(; ωcτ, γ, u_s_fac, u_V_fac)

    d = Dict{String, Any}()

    problem = "hybrid_vorticity"

    Nμ = 3
    B_ref = 1.0
    μ0 = T_ref / B_ref

    eta = 0.4
    η = sqrt(ωcτ) * eta

    Lz = 1.2
    Lx = 1.0
    Nx = [32, 96, 120, 144][sz]
    Nz = [100, 216, 248, 280][sz]
    Nvx = [16, 22, 24, 26][sz]
    Nvz = [16, 22, 24, 26][sz]
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)
    @show device

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref kx)
    if just_setup
        return (; d)
    end

    fi_0(X, Z, VX, VZ) = begin
        ni = ni0.(Z)
        uix = udx.(Z) .+ uEx.(X, Z)
        uiz = uEz.(X, Z)
        Ti = Ti0.(Z)
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

end
