module KHShared

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Interpolations
using SparseArrays
using LinearAlgebra
using Cubature
using FastGaussQuadrature

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
    return dict
end

function params(; T_ref, n0, ωpτ, ωcτ, Ae=1/1836, Lz=1.2, α)
    norm = do_flexible_normalization(; n0, ωpτ, ωcτ)

    Lx = 1.0
    n_ref = 1.0
    B_ref = 1.0

    Ai = 1.0
    Zi = 1.0
    Ze = -Zi

    vti = sqrt(T_ref / Ai)
    vte = sqrt(T_ref / Ae)

    μ0 = T_ref / B_ref
    @show μ0

    n_ratio = 0.4

    # Magnetic field profile
    By0(args...) = B_ref
    
    # Desired density profile
    gi(z) = begin
        return n_ratio * n_ref * tanh(z/(α*Lz)) + 1.5*n_ref
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

    # Desired ExB velocity
    ExB_desired = 0.05*vti
    E_desired = ExB_desired * B_ref
    ϕ_star(z) = begin
        sqrt((2z/Lz)^2 + α / 4) * E_desired - (Lz/2 * E_desired)
    end
    @show ϕ_star(-Lz/2), ϕ_star(0.0)

    return (;
        norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ,
        gz=0.0,
        By0, gi, Pxi, Pxe, Wi, We, ϕ_star
        )
end


function construct_vlasov_hybrid_eq(params)
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
    ni_aux(z) = HCubature.hcubature((v) -> fi_aux(z, v...), (-10vti, -10vti), (10vti, 10vti))[1]

    dz = Lz / 400
    zs = -Lz/2:dz:Lz/2

    ni_aux_data = ni_aux.(zs)
    ni_aux_fit = cubic_spline_interpolation(zs, ni_aux_data, extrapolation_bc=Line())

    ne_poisson_match_data = -(d2ϕ_star_dz2.(zs) .+ Zi * ni_aux_data) / Ze
    ne_fit = cubic_spline_interpolation(zs, ne_poisson_match_data, extrapolation_bc=Line())

    fi_eq(z, vx, vz) = fi_aux(z, vx, vz)
    ne_eq(z) = ne_fit(z)
    ni_eq(z) = ni_aux_fit(z)

    return (; fi_eq, ne_eq, ni_eq)
end

end
