module HybridRayleighTaylor

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Interpolations

function make_sim_equilibrium(::Val{device}) where device
    d = Dict{String, Any}()

    problem = "hybrid_rayleigh_taylor"
    Nz = 200
    Nμ = 30
    Nvx = 40
    Nvz = 40

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-3

    B_ref = 1.0

    α = 25.0

    Ai = 1.0
    Ae = 1/20
    Zi = 1.0
    Ze = -Zi

    vti = sqrt(T_ref / Ai)

    μ0 = T_ref / B_ref
    @show μ0

    ωcτ = 1.0
    ωpτ = 1.0

    kx = 1
    gz = -1e-4
    
    merge!(d, @strdict problem Nz Nμ B_ref α Lx Lz n_ref T_ref kx gz)

    # Magnetic field profile
    By0(z) = B_ref
    By0(x, z) = By0(z)
    
    # Desired density profile
    g(z) = begin
        return 0.5 * n_ref * tanh(α*z/Lz) + 1.5*n_ref
    end

    # x component of Canonical momentum
    Px(z, vx) = begin
        return Ai * vx + z * Zi * B_ref
    end
    E(z, vx, vz) = begin
        return Ai * (vx^2 + vz^2) / (2*T_ref) - Ai * gz * z / T_ref
    end

    ni_eq(z, vx) = g(Px(z, vx) / (Zi * B_ref))
    fi_eq(z, vx, vz) = Ai * ni_eq(z, vx) / (2pi * T_ref) * exp(-E(z, vx, vz))

    # Now calculate the zeroth moment of fi_eq to obtain ne_eq
    dv = vti/40
    vxs = -8vti:dv:8vti
    vys = -8vti:dv:8vti
    zs = -Lx:0.001:Lx

    ne(z) = begin
        sum(fi_eq.(z, vxs, vys')) * dv^2
    end
    ne_points = ne.(zs)
    ne_interp = cubic_spline_interpolation(zs, ne_points, extrapolation_bc=Line())

    Fe_eq(Rz, μ) = begin
        ne_interp(Rz) * Ae / (2pi * T_ref) * exp(-By0(Rz) * μ / T_ref)
    end

    sim = Helpers.two_species_1d2v_vlasov_dk_hybrid((; Fe_0=Fe_eq, fi_0=fi_eq, By0);
        Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, 
        ϕ_left=0.0, ϕ_right=0.0, vth=vti,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai, 
        gz, device, z_bcs=:reservoir);

    d = PDEHarness.normalize!(d)
    return (; d, sim)
end

function make_sim_equilibrium_2d(::Val{device}) where {device}
    d = Dict{String, Any}()

    problem = "hybrid_rayleigh_taylor"
    Nz = 200
    Nx = 64
    Nμ = 3
    Nvx = 20
    Nvz = 20

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-3

    B_ref = 1.0

    α = 25.0

    Ai = 1.0
    Ae = 1/20
    Zi = 1.0 * 10
    Ze = -Zi

    vti = sqrt(T_ref / Ai)

    μ0 = T_ref / B_ref
    @show μ0

    ωcτ = 1.0
    ωpτ = 1.0

    kx = 1
    gz = -1e-3

    δ = 1e-4
    
    merge!(d, @strdict problem Nz Nμ B_ref α Lx Lz n_ref T_ref kx δ gz)

    # Magnetic field profile
    By0(args...) = B_ref
    
    # Desired density profile
    g(z) = begin
        return 0.5 * n_ref * tanh(α*z/Lz) + 1.5*n_ref
    end

    # x component of Canonical momentum
    Px(z, vx) = begin
        return Ai * vx + z * Zi * B_ref
    end
    E(z, vx, vz) = begin
        return Ai * (vx^2 + vz^2) / (2*T_ref) - Ai * gz * z / T_ref
    end

    ni_eq(z, vx) = g(Px(z, vx) / (Zi * B_ref))
    fi_eq(z, vx, vz) = Ai * ni_eq(z, vx) / (2pi * T_ref) * exp(-E(z, vx, vz))

    # Defined all functions

    # Now calculate the zeroth moment of fi_eq to obtain ne_eq
    dv = vti/20
    vxs = -8vti:dv:8vti
    vys = -8vti:dv:8vti
    zs = -Lz:0.01:Lz

    ne(z) = begin
        sum(fi_eq.(z, vxs, vys')) * dv^2
    end
    ne_points = ne.(zs)
    @info "calcualted ne_points"
    ne_interp = cubic_spline_interpolation(zs, ne_points, extrapolation_bc=Line())
    @info "created interpolation"

    Fe_eq(Rz, μ) = begin
        ne_interp(Rz) * Ae / (2pi * T_ref) * exp(-By0(Rz) * μ / T_ref)
    end

    theta(Rx) = 2pi*kx*Rx/Lx
    perturbation(Rx, Rz) = 1 + δ*exp(-(Rz - 0.0cos(theta(Rx)))^2/0.01) * cos(theta(Rx))
    Fe_0(Rx, Rz, μ) = begin
        Fe_eq(Rz, μ) * perturbation(Rx, Rz)
    end
    fi_0(x, z, vx, vz) = fi_eq(z, vx, vz)

    @info "Setting up sim"
    sim = Helpers.two_species_2d_vlasov_dk_hybrid(Val(device), 
        (; Fe_0, fi_0, By0);
        Nx, Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth=vti,
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai, 
        gz, z_bcs=:reservoir);
    @info "Done setting up sim"

    d = PDEHarness.normalize!(d)
    return (; d, sim)
end

end

