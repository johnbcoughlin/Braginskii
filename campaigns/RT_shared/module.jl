module RTShared

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

function params(; n0, ωpτ, ωcτ, Ae=1/25)
    norm = do_flexible_normalization(; n0, ωpτ, ωcτ)

    gz = -3e-5

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-4

    B_ref = 1.0

    α = 1 / 25.0

    Ai = 1.0
    Zi = 1.0
    Ze = -Zi

    vti = sqrt(T_ref / Ai)
    vte = sqrt(T_ref / Ae)

    ωg = sqrt(abs(gz / α))

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

    return (;
        norm,
        Lz, Lx,
        n_ref, T_ref, B_ref,
        α,
        Ai, Ae, Zi, Ze,
        vti, vte, ωcτ, ωpτ, ωg,
        kx, gz, δ,
        By0, gi, Pxi, Pxe, Wi, We, ϕ_star
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
    ni_aux(z) = HCubature.hcubature((v) -> fi_aux(z, v...), (-10vti, -10vti), (10vti, 10vti))[1]

    dz = Lz / 400
    zs = -Lz/2:dz:Lz/2

    ni_aux_data = ni_aux.(zs)
    ni_aux_fit = cubic_spline_interpolation(zs, ni_aux_data, extrapolation_bc=Line())

    fe_aux_density(z̃) = (Zi * ni_aux_fit(z̃) + 1/ωpτ * d2ϕ_star_dz2(z̃)) * exp(
        Ze*ωpτ*ϕ_star(z̃) / Te)
    fe_aux(z, vx, vz) = fe_aux_density(Pxe(z, vx)) * Ae / (2pi*Te) * exp(
        -Ae*(vx^2+vz^2 - 2*z*gz) / (2Te) - Ze*ωpτ*ϕ_star(z)/Ti)
    ne_aux(z) = HCubature.hcubature((v) -> fe_aux(z, v...), (-10vte, -10vte), (10vte, 10vte))[1]
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

# Computes the Hermite expansion of our perturbed equilibrium,
# given the perturbation as product of separable factors depending
# solely on x and z.
function vlasov_eq_hermite_expansions(fe_eq, fi_eq, perturbation_x, perturbation_z, X, Z, Nvx, Nvz, vte, vti)
    # We can compute the equilibrium moments first

    ArrayType = typeof(X).name.wrapper

    fe_moments = vlasov_eq_hermite_expansions_species(fe_eq, perturbation_x, perturbation_z, X, Z, Nvx, Nvz, vte, 11*vte) |> ArrayType
    fi_moments = vlasov_eq_hermite_expansions_species(fi_eq, perturbation_x, perturbation_z, X, Z, Nvx, Nvz, vti, 11*vti) |> ArrayType

    return (; fe_moments, fi_moments)
end

# Computes the Hermite expansion of our perturbed equilibrium,
# given the perturbation as product of separable factors depending
# solely on x and z.
function vlasov_eq_hermite_expansions_species(f_eq, perturbation_x, perturbation_z, X, Z, Nvx, Nvz, vt, vmax)
    # We can compute the equilibrium moments first
       
    vx_nodes, vx_w = FastGaussQuadrature.gausslegendre(300)

    He_points_vand = Float64.(Braginskii.He_up_to_n(Nvx-1, vx_nodes * vmax / vt))

    f_at_vx_nodes(z) = f_eq.(Ref(z), vx_nodes * vmax, 0.0) / sqrt(1 / (2pi * vt^2))
    moments(z) = begin
        f_vx = f_at_vx_nodes(z) 
        res = He_points_vand * Diagonal(vx_w) * f_vx * vmax
        res
    end
 
    z_vx_moments = zeros(length(Z), Nvx)
    for i in 1:length(Z)
        z_vx_moments[i, :] .= moments(Z[i])
    end
    z_vx_moments = z_vx_moments
    result = zeros(length(X), 1, length(Z), Nvx, 1, Nvz)

    result[:, 1, :, :, 1, 1] .= reshape(z_vx_moments, (1, length(Z), Nvx))
    # Higher vz moments are all zero.

    Xa = Array(X)
    Za = Array(Z)
    @. result *= (1 + perturbation_x(Xa) * perturbation_z(Za))

    result
end

function opt_oct_values()
    points_oct = vcat(
    0.8*ones(5),
    1.2*ones(5),
    1.6*ones(4),
    2.4*ones(4),
    4.0*ones(4),
    6.0*ones(4),
    10.0*ones(4),
    15.0*ones(4),
    20.0*ones(4)
    ) ./ 2
    points_opt = vcat(
    2:4:18,
    10:4:26,
    15:5:30,
    15:5:30,
    15:5:30,
    15:5:30,
    15:5:30,
    15:5:30,
    15:5:30,
    ) .* 2.0

    return points_opt, points_oct
end

function kinetic_examples()
    return [5, 10, 14]
end

end
