module RTMagnetization2

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

function make_sim_hybrid(::Val{device}; ωcτ) where {device}
    d = Dict{String, Any}()

    n0 = 1e22u"m^-3"
    ωpτ = ωcτ

    problem = "hybrid_rayleigh_taylor"
    params = RTShared.params(; T_ref=reference_temp(), gz=gravity(), 
        n0, ωpτ, ωcτ, Ae=1/1836, Lz=1.5)

    (; norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ, ωg,
        kx, gz, δ,
        ωg, By0, gi, Pxi, Pxe, Wi, We, ϕ_star
        ) = params

    (; fi_eq, ne_eq) = RTShared.construct_vlasov_eq(params)
    
    Nμ = 3
    μ0 = T_ref / B_ref

    η = sqrt(ωcτ) / sqrt(.86) * .18

    Nx = 96
    Nz = 256
    Nvx = 16
    Nvz = 16
    zmin = -Lz/2
    zmax = Lz/2
    buffer = allocator(device)
    x_grid = Helpers.xz_grid_2d(Nx, Nz, zmin, zmax, Lx, buffer)
    @show device

    merge!(d, @strdict problem ωcτ ωpτ Ae Ai Ze Zi Nx Nz Nvx Nvz Nμ B_ref α Lx Lz n_ref T_ref kx δ gz ωg)

    fi_moments = RTShared.vlasov_eq_hermite_expansions_species(
        fi_eq, Returns(0.0), Returns(0.0),
        x_grid.X, x_grid.Z, Nvx, Nvz, vti, 11*vti) |> Braginskii.arraytype(buffer)

    dz = x_grid.z.dx
    left_grid = Helpers.xz_grid_2d(Nx, 3, zmin-3dz, zmin, Lx, buffer)
    fi_left = RTShared.vlasov_eq_hermite_expansions_species(
        fi_eq, Returns(0.0), Returns(0.0),
        left_grid.X, left_grid.Z, Nvx, Nvz, vti, 11vti) |> Braginskii.arraytype(buffer)
    right_grid = Helpers.xz_grid_2d(Nx, 3, zmax, zmax+3dz, Lx, buffer)
    fi_right = RTShared.vlasov_eq_hermite_expansions_species(
        fi_eq, Returns(0.0), Returns(0.0),
        right_grid.X, right_grid.Z, Nvx, Nvz, vti, 11vti) |> Braginskii.arraytype(buffer)

    Fe_eq(Rz, μ) = begin
        ne_eq(Rz) * exp(-μ / μ0)
    end
    theta(Rx) = 2pi*kx*Rx/Lx
    perturbation(Rx, Rz) = 1 + δ*exp(-(Rz)^2/0.01) * cos(theta(Rx))
    Fe_0(Rx, Rz, μ) = begin
        Fe_eq(Rz, μ) * perturbation(Rx, Rz)
    end

    @info "Setting up sim"
    sim = Helpers.two_species_2d_vlasov_dk_hybrid(Val(device),
        (; Fe_0, fi_0=nothing, By0);
        Nx, Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth=vti,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai, 
        grid_scale_hyperdiffusion_coef=η,
        gz, z_bcs=:reservoir,
        fi_ic=fi_moments,
        ion_bc_lr=(fi_left, fi_right));
    @info "Done setting up sim"

    return (; d, sim)
end

end
