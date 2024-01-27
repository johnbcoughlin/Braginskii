module RTLarmorRadius

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Interpolations
using SparseArrays
using LinearAlgebra

using Unitful
using PhysicalConstants.CODATA2018: e, μ_0, m_p, ε_0, m_e, c_0
using HCubature

function do_flexible_normalization(; n0=1e20u"m^-3", ωpτ, ωcτ)
    ωp = sqrt(e^2 * n0 / (ε_0 * m_p)) |> upreferred
    tau = ωpτ / ωp |> upreferred

    ωc = ωcτ / tau
    B0 = uconvert(u"T", ωc * m_p / e)
    v0 = B0 / sqrt(m_p * n0 * μ_0) |> upreferred
    L = v0 * tau |> upreferred
    p0 = B0^2 / μ_0
    T0 = uconvert(u"keV", p0 / n0)
    vt0 = sqrt(T0 / m_p) |> upreferred
    rL0 = vt0 / ωc |> upreferred
    rp0 = c_0 / ωp
    L_debye = sqrt(ε_0 * T0 / (n0 * e^2)) |> upreferred

    lnΛ = 10.0
    νp = e^4 * n0 * lnΛ / (3sqrt(2) * pi^(3/2) * ε_0^2 * sqrt(m_p) * T0^(3/2))
    νpτ = νp * tau |> upreferred

    β = upreferred(n0 * T0 / (B0^2 / 2μ_0))

    dict = @strdict n0 ωpτ ωcτ tau L v0 B0 T0 rL0 L_debye νpτ β
end

function params(; ωpτ, ωcτ, Ae=1/25, gz=-1e-3)
    norm = do_flexible_normalization(; ωpτ, ωcτ)

    Nz = 200
    Nx = 96
    Nμ = 3
    Nvx = 20
    Nvz = 20

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-3

    B_ref = 1.0

    α = 1 / 25.0

    Ai = 1.0
    Zi = 1.0
    Ze = -Zi

    vti = sqrt(T_ref / Ai)

    μ0 = T_ref / B_ref
    @show μ0

    kx = 1

    δ = 1e-4

    # Magnetic field profile
    By0(args...) = B_ref
    
    # Desired density profile
    gi(z) = begin
        return 0.5 * n_ref * tanh(z/(α*Lz)) + 1.5*n_ref
    end
    # x component of Canonical momentum
    Pxi(z, vx) = begin
        return Ai * vx / (Zi * ωcτ * B_ref) + z
    end
    Wi(z, vx, vz) = begin
        return Ai * (vx^2 + vz^2) / (2*T_ref) - Ai * gz * z / T_ref
    end
    Pxe(z, vx) = begin
        return Ae * vx / (Ze * ωcτ * B_ref) + z
    end
    We(z, vx, vz) = begin
        return Ae * (vx^2 + vz^2) / (2T_ref) - Ae * gz * z / T_ref
    end

    ϕ_star(z) = 0.0
    ni_eq(z, vx) = gi(Pxi(z, vx))
    fi_eq(z, vx, vz) = Ai * ni_eq(z, vx) / (2pi * T_ref) * exp(-Wi(z, vx, vz))

    return (;
        Nz, Nx, Nvx, Nvz,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, ωcτ, ωpτ,    
        kx, gz, δ,
        By0, gi, Pxi, Pxe, Wi, We, ϕ_star,
        ni_eq, fi_eq
        )
end

function construct_vlasov_eq(params)
    (; Pxi, Pxe, Wi, We, ϕ_star, gi,
    Zi, Ze, Ai, Ae,
    ωpτ, ωcτ, gz,
    T_ref, Lz
    ) = params

    d2ϕ_star_dz2(z) = begin
        dz = Lz/1000
        (ϕ_star(z-dz) - 2ϕ_star(z) + ϕ_star(z+dz)) / dz^2
    end

    Ti = Te = T_ref
    vti = sqrt(Ti / Ai)
    vte = sqrt(Ti / Ae)

    fi_aux(z, vx, vz) = begin
        Px = Pxi(z, vx)
        gi(Px) * exp(Zi*ωpτ*ϕ_star(Px) / Ti) * Ai / (2pi * Ti) * exp(
            -Ai*(vx^2+vz^2 - 2*z*gz) / (2Ti) - Zi*ωpτ*ϕ_star(z)/Ti)
    end
    ni_aux(z) = hcubature((v) -> fi_aux(z, v...), (-10vti, -10vti), (10vti, 10vti))[1]

    dz = 0.01
    zs = -Lz:dz:Lz

    ni_aux_data = ni_aux.(zs)
    ni_aux_fit = cubic_spline_interpolation(zs, ni_aux_data, extrapolation_bc=Line())

    fe_aux_density(z̃) = (Zi * ni_aux_fit(z̃) + 1/ωpτ * d2ϕ_star_dz2(z̃)) * exp(
        Ze*ωpτ*ϕ_star(z̃) / Te)
    fe_aux(z, vx, vz) = fe_aux_density(Pxe(z, vx)) * Ae / (2pi*Te) * exp(
        -Ae*(vx^2+vz^2 - 2*z*gz) / (2Te) - Ze*ωpτ*ϕ_star(z)/Ti)
    ne_aux(z) = hcubature((v) -> fe_aux(z, v...), (-10vte, -10vte), (10vte, 10vte))[1]
    ne_aux_data = ne_aux.(zs)
    ne_aux_fit = cubic_spline_interpolation(zs, ne_aux_data, extrapolation_bc=Line())

    ϕ_star_data = ϕ_star.(zs)

    N = length(zs)
    D2 = spdiagm(-1 => ones(N-1), 0 => -2*ones(N), 1 => ones(N-1))
    obj_func(ϕ) = begin
        ρc = @. Zi * ni_aux_data * exp(Zi * ωpτ * (ϕ_star_data - ϕ) / Ti)
        @. ρc += Ze * ne_aux_data * exp(Ze * ωpτ * (ϕ_star_data - ϕ) / Te)
        return 1/ωpτ * D2*ϕ + ρc
    end
    jacobian(ϕ) = begin
        J = 1/ωpτ*D2
        diag = @. -Zi^2*ωpτ/Ti*ni_aux_data * exp(Zi * ωpτ * (ϕ_star_data - ϕ) / Ti)
        @. diag += -Ze^2*ωpτ/Te*ne_aux_data * exp(Ze * ωpτ * (ϕ_star_data - ϕ) / Te)
        J += spdiagm(0 => diag)
        return J
    end

    # Do some Newton iterations
    ϕ = copy(ϕ_star_data)
    y = obj_func(ϕ)
    @show norm(y)
    iters = 0
    while norm(y) > 1e-12 && iters < 5
        iters += 1
        J = jacobian(ϕ)
        ϕ = ϕ - J \ y
        y = obj_func(ϕ)
        @show norm(y)
    end
    ϕ_fit = cubic_spline_interpolation(zs, ϕ, extrapolation_bc=Line())

    ϕ_factor_e(z) = exp(ωpτ * Ze * (ϕ_star(z) - ϕ_fit(z)) / Te)
    ϕ_factor_i(z) = exp(ωpτ * Zi * (ϕ_star(z) - ϕ_fit(z)) / Ti)

    fe_eq(z, vx, vz) = fe_aux(z, vx, vz) * ϕ_factor_e(z)
    fi_eq(z, vx, vz) = fi_aux(z, vx, vz) * ϕ_factor_i(z)
    ne_eq(z) = ne_aux_fit(z) * ϕ_factor_e(z)
    ni_eq(z) = ni_aux_fit(z) * ϕ_factor_i(z)

    return (; fe_eq, fi_eq, ne_eq, ni_eq)
end

function make_sim_hybrid(::Val{device}; rL) where {device}
    d = Dict{String, Any}()

    problem = "hybrid_rayleigh_taylor"

    (; Nz, Nx, Nvx, Nvz,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, ωcτ, ωpτ,    
        kx, gz, δ,
        By0, _, _, _,
        _, fi_eq) = params()
    Nμ = 3
    μ0 = T_ref / B_ref
    
    merge!(d, @strdict problem Nz Nμ B_ref α Lx Lz n_ref T_ref kx δ gz)

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

    return (; d, sim)
end

end
