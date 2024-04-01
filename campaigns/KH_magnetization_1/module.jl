module KHMagnetization1

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Unitful

function reference_temp()
    return 1e-3
end

function interface_width()
    return 0.04
end

function make_sim_hybrid(::Val{device}; ωcτ, just_setup=false) where {device}
    d = Dict{String, Any}()

    ωpτ = ωcτ
    problem = "hybrid_kelvin_helmholtz"

    T = reference_temp()
    α = interface_width()
    B_ref = 1.0
    Ai = 1.0
    Ae = 1 / 1836
    Zi = 1.0
    Ze = -1.0

    Lx = 1.0
    Lz = 1.0
    
    vti = sqrt(T / Ai)
    vte = sqrt(T / Ae)

    Nμ = 3
    μ0 = T / B_ref

    Nx = 96
    Nz = 216
    Nvx = 20
    Nvz = 20
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)

    η = sqrt(ωcτ) / sqrt(.86) * .18

    u_ref = 0.1 * vti
    n_ratio = 0.8

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz B_ref α Lx Lz T u_ref)
    if just_setup
        return (; d)
    end

    n(x, z) = 1.5 + n_ratio * tanh(z / α)
    ux(x, z) = u_ref * (1.5 + 0.5 * tanh(z / α))
    uz(x, z) = u_ref * (0.5 * cos(2pi*x/Lx) * exp(-z^2/α))

    fi_0_batch(X, Z, VX, VZ) = begin
        N = n.(X, Z)
        UX = ux.(X, Z)
        UZ = uz.(X, Z)
        @. Ai * N / (2pi*T) * exp(-Ai * ((VX-UX)^2 + (VZ-UZ)^2) / (2T))
    end
    fi_0(x, z, vx, vz) = Ai * n(x, z) / (2pi * T) * exp(-Ai*((vx-ux(x, z))^2 + (vz-uz(x, z))^2)/(2T))
    Fe_0(x, z, μ) = n(x, z) * exp(-μ / μ0)
    By0(args...) = B_ref

    @info "Setting up sim"
    sim = Helpers.two_species_2d_vlasov_dk_hybrid(Val(device),
        (; Fe_0, fi_0=Braginskii.BatchFunc(fi_0_batch), By0);
        Nx, Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth=vti,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai,
        grid_scale_hyperdiffusion_coef=η,
        gz=0.0, z_bcs=:reservoir,
        fi_ic=nothing,
        ion_bc_lr=nothing)
    @info "Done setting up sim"

    return (; d, sim)
end

end
