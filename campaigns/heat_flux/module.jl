module HeatFlux

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

function make_sim_hybrid(::Val{device}; ωcτ, eta, sz, ζ, just_setup=false) where {device}
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

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref ζ)
    if just_setup
        return (; d)
    end

    interface_location(x) = 0.2*sin(2pi*x/Lx)
    d_interface_location_dx(x) = 2pi/Lx * 0.2 * cos(2pi*x/Lx)
    profile(x, z) = begin
        1.0 + 0.3 * tanh((z - interface_location(x)) / α)
    end
    dprofiledx(x, z) = begin
        -0.3 / α * d_interface_location_dx(x) * sech((z - interface_location(x)) / α)^2
    end
    dprofiledz(x, z) = begin
        0.3 / α * sech((z - interface_location(x)) / α)^2
    end

    ni(x, z) = n_ref * profile(x, z)^ζ
    Ti(x, z) = T_ref * profile(x, z)^(1-ζ)
    p0 = n_ref * T_ref
    uix(x, z) = p0 / ωcτ * dprofiledz(x, z) / ni(x, z)
    uiz(x, z) = -p0 / ωcτ * dprofiledx(x, z) / ni(x, z)

    fi_0(X, Z, VX, VZ) = begin
        ni0 = ni.(X, Z)
        Ti0 = Ti.(X, Z)
        uix0 = uix.(X, Z)
        uiz0 = uiz.(X, Z)
        @. ni0 / (2pi*Ti0) * exp(-((VX-uix0)^2 + (VZ-uiz0)^2) / (2Ti0))
    end
    Fe_0(x, z, μ) = ni(x, z) * exp(-μ / μ0)
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

    return (; d, sim)
end

end
