module KelvinHelmholtz

using DrWatson
using TimerOutputs
using Parameters
using CSV
using Unitful

include("constants.jl")

export compute_normalization, create_kh_ic, electron_distribution, ion_distribution, Ex_ic, Bz_ic, ϕ_ic, load_spline_fit, KelvinHelmholtzIC, KHPerturbation, compute_Ne, compute_Ee, compute_Ni, compute_Ei

struct SplineFit
    xvec::Vector{Float64}
    a_coeffs::Vector{Float64}
    b_coeffs::Vector{Float64}

    SplineFit(xvec, a_coeffs, b_coeffs) = begin
        @assert length(xvec) == length(a_coeffs)
        @assert length(b_coeffs) == 2
        new(xvec, a_coeffs, b_coeffs)
    end
end

spline_fit(fit::SplineFit, x) = begin
    xs = fit.xvec
    as = fit.a_coeffs
    b = fit.b_coeffs

    sum(as[i] * abs(x - xs[i])^3 for i in 1:length(xs)) + b[2] * x + b[1]
end

load_spline_fit(dir, file) = begin
    csv = CSV.File(joinpath(dir, file), header=["xvec", "a", "b"])
    SplineFit(
        csv[:xvec],
        csv[:a],
        csv[:b][1:2]
    )
end

struct KHPerturbation
    center_x::Float64
    center_y::Float64

    amplitude::Float64
    wavelength::Float64
end

@with_kw struct KelvinHelmholtzIC
    ωpτ::Float64
    ωcτ::Float64
    Ai::Float64
    Ae::Float64
    Zi::Float64
    Ze::Float64
    Ω_c_i::Float64 = ωcτ * Zi / Ai
    Ω_c_e::Float64 = ωcτ * Ze / Ae
    T::Float64
    b::Float64
    d::Float64
    E0::Float64
    E_x_0::Float64 = E0 / ωcτ

    ni_aux_fit::SplineFit
    ϕ_fit::SplineFit
    Ay_fit::SplineFit

    Ex_fit::SplineFit
    Bz_fit::SplineFit

    pert::KHPerturbation

    normalization::Dict{String, Any}
end

function create_kh_ic(normalization_dict; T, b, d,
                      E0, Ai, Ae, Zi, Ze, case,
                      pert_amplitude, pert_wavelength, kwargs...)
    @unpack ω_p_τ, ω_c_τ = normalization_dict

    @assert case ∈ ("A1", "A2", "A3", "A4")

    spline_fit_dir = joinpath(dirname(pathof(@__MODULE__)), "../spline_fits/Case$(case)")

    ni_aux_fit = load_spline_fit(spline_fit_dir, "case$(case)_ni_aux_fit.csv")
    ϕ_fit = load_spline_fit(spline_fit_dir, "case$(case)_phi_fit.csv")
    Ay_fit = load_spline_fit(spline_fit_dir, "case$(case)_Ay_fit.csv")
    Ex_fit = load_spline_fit(spline_fit_dir, "case$(case)_Ex_fit.csv")
    Bz_fit = load_spline_fit(spline_fit_dir, "case$(case)_Bz_fit.csv")

    pert = KHPerturbation(0., 0., pert_amplitude, pert_wavelength)

    KelvinHelmholtzIC(
        ωpτ=ω_p_τ,
        ωcτ=ω_c_τ,
        Ai=Ai, Ae=Ae, Zi=Zi, Ze=Ze,
        T=T, b=b, d=d,
        E0=E0,
        ni_aux_fit=ni_aux_fit, ϕ_fit=ϕ_fit, Ay_fit=Ay_fit, Ex_fit=Ex_fit, Bz_fit=Bz_fit,
        pert=pert,
        normalization=normalization_dict
    )
end

ϕ_star(ic::KelvinHelmholtzIC, x::Real) = begin
    @unpack E_x_0, d = ic
    -E_x_0 * d / 2 * log(1.0 + exp(2 * x / d))
end

ϕ_star_double_prime(ic::KelvinHelmholtzIC, x::Real) = begin
    @unpack E_x_0, d = ic
    -E_x_0 / d / (1.0 + cosh(2 * x / d))
end

g_i(ic::KelvinHelmholtzIC, x::Real) = begin
    (1.0 + exp(2 * x / ic.d))^(ic.b * ic.d / 2)
end

g_e(ic::KelvinHelmholtzIC, x::Real) = begin
    @unpack Zi, Ze, ni_aux_fit, ωcτ, ωpτ = ic

    ni_aux = spline_fit(ni_aux_fit, x)
    -Zi / Ze * ni_aux - 1.0 / Ze * (ωcτ / ωpτ^2) * ϕ_star_double_prime(ic, x)
end

compute_Ni(ic::KelvinHelmholtzIC, x, vy) = begin
    @unpack Ay_fit, Ω_c_i, ωcτ, Zi, Ai, T = ic
    Ay = spline_fit(Ay_fit, x)
    xshift_i = Ay + vy / Ω_c_i
    Ni = g_i(ic, xshift_i) * exp(ωcτ * Zi / T * ϕ_star(ic, xshift_i))
    return Ni
end

compute_Ei(ic::KelvinHelmholtzIC, x, vx, vy) = begin
    @unpack Ay_fit, Ω_c_i, ωcτ, Zi, Ai, T = ic
    maxwellian_2d = ic.Ai / (2π * T) * exp(-Ai * (vx^2 + vy^2) / (2T))
    ϕ = spline_fit(ic.ϕ_fit, x)
    Ei = maxwellian_2d * exp(-ωcτ * Zi / T * ϕ)
    Ei
end

function ion_distribution(ic::KelvinHelmholtzIC, X1, Y1, VX, VY)
    X = X1 .- ic.pert.center_x
    Y = Y1 .- ic.pert.center_y

    @unpack Ay_fit, Ω_c_i, ωcτ, Zi, Ai, T = ic

    Ni = compute_Ni.(Ref(ic), X, VY)
    Ei = compute_Ei.(Ref(ic), X, VX, VY)

    return Ni .* Ei .* ones(size(Y1))
end

compute_Ne(ic::KelvinHelmholtzIC, x, vy) = begin
    @unpack Ay_fit, Ω_c_e, ωcτ, Ze, Ae, T, d = ic
    Ay = spline_fit(Ay_fit, x)
    xshift_e = Ay + vy / Ω_c_e

    Ne = g_e(ic, xshift_e) * exp(ωcτ * Ze / T * ϕ_star(ic, xshift_e))
end

compute_Ee(ic::KelvinHelmholtzIC, x, vx, vy) = begin
    @unpack Ay_fit, Ω_c_e, ωcτ, Ze, Ae, T, d = ic

    maxwellian_2d = ic.Ae / (2π * T) * exp(-Ae * (vx^2 + vy^2) / (2T))
    ϕ = spline_fit(ic.ϕ_fit, x)
    Ee = maxwellian_2d * exp(-ωcτ * Ze / T * ϕ)
end

function electron_distribution(ic::KelvinHelmholtzIC, X1, Y1, VX, VY)
    X = X1 .- ic.pert.center_x
    Y = Y1 .- ic.pert.center_y

    @unpack Ay_fit, Ω_c_e, ωcτ, Ze, Ae, T, d = ic

    Ne = compute_Ne.(Ref(ic), X, VY)
    Ee = compute_Ee.(Ref(ic), X, VX, VY)

    fe = Ee .* Ne

    pert = @. (1.0 + ic.pert.amplitude * sin(2π * Y / ic.pert.wavelength) * exp(-(X/d)^6))

    return fe .* pert
end

ϕ_ic(ic::KelvinHelmholtzIC, x) = spline_fit(ic.ϕ_fit, x)
Ex_ic(ic::KelvinHelmholtzIC, x) = spline_fit(ic.Ex_fit, x)

Ex_ic(ic::KelvinHelmholtzIC, X, Y) = spline_fit.(Ref(ic.Ex_fit), X)

Bz_ic(ic::KelvinHelmholtzIC, x::Real) = spline_fit(ic.Bz_fit, x)

Bz_ic(ic::KelvinHelmholtzIC, X, Y) = spline_fit.(Ref(ic.Bz_fit), X)

function initial_variable_values(kh_ic::KelvinHelmholtzIC, ion_grid, electron_grid)
    fe_0 = NDG.allocate_variable(electron_grid.patch, 1)
    @unpack X, Y, VX, VY = meshgrid(electron_grid); VXe, VYe = VX, VY;
    @timeit "electron ic" NDG.component(fe_0, 1).data .= electron_distribution(kh_ic, X, Y, VXe, VYe)

    fi_0 = NDG.allocate_variable(ion_grid.patch, 1)
    @unpack X, Y, VX, VY = meshgrid(ion_grid); VXi, VYi = VX, VY;
    @timeit "ion ic" NDG.component(fi_0, 1).data .= ion_distribution(kh_ic, X, Y, VX, VY)

    # Get X, Y and check that it matches between grids.
    X, Y = xy_meshgrid(ion_grid)
    Xe, Ye = xy_meshgrid(electron_grid)
    @assert X == Xe; @assert Y == Ye;

    xypatch = ion_grid.xypatch

    E0 = NDG.allocate_variable(xypatch, 3)
    NDG.component(E0, 1).data .= Ex_ic(kh_ic, X, Y)
    B0 = NDG.allocate_variable(xypatch, 3)
    NDG.component(B0, 3).data .= Bz_ic(kh_ic, X, Y)

    return (; fe=fe_0, fi=fi_0, E=E0, B=B0)
end

function compute_normalization(ω_p_τ::Real,
                               ω_c_τ::Real,
                               n0::typeof(1.0u"m^-3"))
    @unpack qe, μ0, mp, ϵ0, c_light = PhysicalConstants

    ω_p = uconvert(u"s^-1", sqrt((qe^2 * n0) / (ϵ0 * mp)))
    τ = uconvert(u"s", ω_p_τ / ω_p)
    ω_c = ω_c_τ / τ
    B0 = uconvert(u"T", ω_c * mp / qe)
    v0 = uconvert(u"m/s", B0 / sqrt(mp * n0 * μ0))
    L = v0 * τ
    p0 = uconvert(u"Pa", B0^2 / μ0)
    T0 = uconvert(u"eV", p0 / n0)
    v_th0 = uconvert(u"m/s", sqrt(T0 / mp))
    rl0 = v_th0 / ω_c
    δp = c_light / ω_p
    L_debye0 = uconvert(u"m", sqrt(ϵ0 * T0 / (n0 * qe^2)))
    c = c_light / v0

    @strdict ω_p_τ ω_c_τ n0 ω_p τ ω_c B0 v0 L p0 T0 v_th0 rl0 δp L_debye0 c
end
end # module KelvinHelmholtz
