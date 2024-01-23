module KHHybridVlasovComparison

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers

include("../KelvinHelmholtz/src/KelvinHelmholtz.jl")
using .KelvinHelmholtz

function params(case)
    if case == "A1"
        return (; Ly=0.785, d=1/20, b=-10.0, T=6.25e-4, E0=2e-2, kyd=0.4)
    else
        error()
    end
end

function make_sim_vlasov(::Val{device}; Ae=1/25, case) where {device}
    @assert case ∈ ("A1", "A2", "A3", "A4")

    spline_fit_dir = projectdir("campaigns", "KelvinHelmholtz", "spline_fits", "Case$(case)")

    ni_aux_fit = load_spline_fit(spline_fit_dir, "case$(case)_ni_aux_fit.csv")
    ϕ_fit = load_spline_fit(spline_fit_dir, "case$(case)_phi_fit.csv")
    Ay_fit = load_spline_fit(spline_fit_dir, "case$(case)_Ay_fit.csv")
    Ex_fit = load_spline_fit(spline_fit_dir, "case$(case)_Ex_fit.csv")
    Bz_fit = load_spline_fit(spline_fit_dir, "case$(case)_Bz_fit.csv")

    (; Ly, d, b, T, E0, kyd) = params(case)
    ky = kyd/d
    λy = 2pi / ky
    perturbation = KHPerturbation(0.0, 0.0, 1e-4, λy)

    kh_ic = KelvinHelmholtz.KelvinHelmholtzIC(;
        ωpτ=1.0,
        ωcτ=1.0,
        Ai=1.0,
        Ae=Ae,
        Zi=1.0,
        Ze=1.0,
        T,
        b,
        d,
        E0,
        ni_aux_fit, ϕ_fit, Ay_fit, Ex_fit, Bz_fit,
        pert=perturbation,
        normalization=Dict{String, Any}()
    )
    (; Ai, Ae, Zi, Ze, ωpτ, ωcτ) = kh_ic
    vth_i = sqrt(T / Ai)
    vth_e = sqrt(T / Ae)

    fi(args...) = ion_distribution(kh_ic, args...)
    fe(args...) = electron_distribution(kh_ic, args...)
    By0(args...) = 1.0
    ϕ_left = KelvinHelmholtz.spline_fit(ϕ_fit, -0.5)
    ϕ_right = KelvinHelmholtz.spline_fit(ϕ_fit, 0.5)

    Nx = 100
    Nz = 32
    Nvx = 20
    Nvz = 20

    sim = Helpers.two_species_xz_2d2v(Val(device), (; fe_0=fe, fi_0=fi, By0);
        Nx, Nz, Nvx, Nvz, q=1.0, ν_p=0.0, vdisc=:hermite, free_streaming=true,
        Lx=Ly, zmin=-0.5, zmax=0.5,
        ωpτ, ωcτ, Ze, Zi, Ae, Ai,
        vth_i, vth_e, gz=0.0,
        z_bcs=:reservoir,
        ϕ_left, ϕ_right)
end

function ion_distribution(ic::KelvinHelmholtzIC, x1, y1, vx, vy)
    x = x1 - ic.pert.center_x
    y = y1 - ic.pert.center_y

    @unpack Ay_fit, Ω_c_i, ωcτ, Zi, Ai, T = ic

    Ni = KelvinHelmholtz.compute_Ni(ic, x, vy)
    @show Ni
    Ei = KelvinHelmholtz.compute_Ei(ic, x, vx, vy)
    @show Ei

    return Ni * Ei
end

function electron_distribution(ic::KelvinHelmholtzIC, x1, y1, vx, vy)
    x = x1 - ic.pert.center_x
    y = y1 - ic.pert.center_y

    @unpack Ay_fit, Ω_c_e, ωcτ, Ze, Ae, T, d = ic

    Ne = compute_Ne(ic, x, vy)
    Ee = compute_Ee(ic, x, vx, vy)

    pert = (1.0 + ic.pert.amplitude * sin(2π * y / ic.pert.wavelength) * exp(-(x/d)^6))

    return pert * Ne * Ee
end

end
