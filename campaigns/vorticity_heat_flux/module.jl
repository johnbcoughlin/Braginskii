module VorticityHeatFlux

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Unitful
using StaticArrays
using LinearAlgebra

function simulations()
    u_V_fac = 0.06
    γ = 0.4
    sims = [
        (; ωcτ=0.5, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=0.75, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=1.0, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=1.5, ζ=0.5, u_s_fac=-0.2, u_V_fac, γ),
        (; ωcτ=2.0, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
        (; ωcτ=3.0, ζ=0.5, u_s_fac=0.2, u_V_fac, γ),
    ]
    return sims
end

function params(; ωcτ, γ, u_s_fac, u_V_fac, ζ)
    Ai = 1.0
    Ae = 1 / 1836
    Zi = 1.0
    Ze = -1.0
    ωpτ = ωcτ
    n_ref = 1.0
    T_ref = 1e-3
    α = 0.04
    B = 1.0

    vti = sqrt(T_ref / Ai)

    w = 2*α

    u_s = u_s_fac*vti
    u_V = u_V_fac*vti

    kx = 2pi

    return (; Ai, Ae, Zi, Ze, ωpτ, ωcτ, n_ref, T_ref, B, α, γ, ζ, w, u_s, u_V, kx, vti)
end

function profiles(; ωcτ, γ=0.2, u_s_fac=0.2, u_V_fac=0.1, ζ=0.5)
    (; 
    Ai, Ae, Zi, Ze, 
    ωpτ, ωcτ, n_ref, T_ref, 
    B, α, γ, ζ, 
    w, u_s, u_V, kx) = params(; ωcτ, γ, u_s_fac, u_V_fac, ζ)

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

    d_dx(x, z, f, h) = (f(x+h, z) - f(x-h, z)) / (2h)
    d_dz(x, z, f, h) = (f(x, z+h) - f(x, z-h)) / (2h)

    W_tensor(x, z) = begin
        uix(x, z) = uEx(x, z) + udx(z)
        uiz(x, z) = uEz(x, z)
        h = 1e-6

        Wxx = 1.5*d_dx(x, z, uix, h) - 0.5*d_dz(x, z, uiz, h)     
        Wxz = d_dx(x, z, uiz, h) + d_dz(x, z, uix, h)
        Wzz = 1.5*d_dz(x, z, uiz, h) - 0.5*d_dx(x, z, uix, h)
        return @SMatrix [Wxx  Wxz;
                         Wxz  Wzz]
    end

    Pi_tensor(x, z) = begin
        eta3 = 1.0 / (2 * ωcτ * B)
        Wxx, Wxz, _, Wzz = W_tensor(x, z)
        pi0 = phat(z) * p_ref
        Pi_xx = eta3 * pi0 * (-Wxz)
        Pi_xz = eta3 * pi0 * 0.5 * (Wxx - Wzz)
        Pi_zz = eta3 * pi0 * Wxz
        return @SMatrix[Pi_xx  Pi_xz;
                        Pi_xz  Pi_zz]
    end

    P_tensor(x, z) = begin
        Pi = Pi_tensor(x, z)
        pi0 = phat(z) * p_ref
        return @SMatrix[pi0 + Pi[1, 1]   Pi[1, 2];
                        Pi[2, 1]         pi0 + Pi[2, 2]] 
    end
    T_tensor(x, z) = P_tensor(x, z) / ni0(z)

    qi0(x, z) = begin
        h = 1e-6
        dTdx = d_dx(x, z, (x, z) -> Ti0(z), h)
        dTdz = d_dz(x, z, (x, z) -> Ti0(z), h)
        pi0 = phat(z) * p_ref
        qx = 2 * pi0 * dTdz / (ωcτ * B * Zi)
        qz = -2 * pi0 * dTdx / (ωcτ * B * Zi)
        return @SVector [qx, qz]
    end

    return (; phat, ni0, Ti0, ne0, Ex, Ez, rhoc, uEx, uEz, udx, P_tensor, T_tensor, qi0)
end

function make_sim_hybrid(::Val{device}; ωcτ, γ=0.2, u_s_fac=0.2, u_V_fac=0.1, sz=4, ζ=0.5, just_setup=false) where {device}
    (; 
    Ai, Ae, Zi, Ze, 
    ωpτ, ωcτ, n_ref, T_ref, 
    B, α, γ, ζ, 
    w, u_s, u_V, kx, vti) = params(; ωcτ, γ, u_s_fac, u_V_fac, ζ)

    (; phat, ni0, Ti0, ne0, Ex, Ez, rhoc, uEx, 
        uEz, udx, P_tensor, T_tensor, qi0) = profiles(; ωcτ, γ, u_s_fac, u_V_fac, ζ)

    d = Dict{String, Any}()

    problem = "hybrid_vorticity"

    Nμ = 3
    B_ref = 1.0
    μ0 = T_ref / B_ref

    eta = 0.4
    η = sqrt(ωcτ) * eta

    Lz = 1.2
    Lx = 1.0
    Nx = [16, 96, 120, 144][sz]
    Nz = [60, 216, 248, 280][sz]
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

    He3(w) = (w^3 - 3w) / sqrt(6)
    fi_0(X, Z, VX, VZ) = begin
        Nx = length(X); Nz = length(Z); Nvx = length(VX); Nvz = length(VZ);
        @show Nx, Nz, Nvx, Nvz

        ni = ni0.(Z)
        uix = udx.(Z) .+ uEx.(X, Z)
        uiz = uEz.(X, Z)
        T = T_tensor.(X, Z)

        fi_hat = zeros(length(X), 1, length(Z), length(VX), 1, length(VZ))
        # Pre-allocate for W
        W = ((vx, vz) -> @SVector[vx - uix[1, 1, 1], vz - uiz[1, 1, 1]]).(VX, VZ)
        numerator = zeros(1, 1, 1, Nvx, 1, Nvz)
        result = zeros(1, 1, 1, Nvx, 1, Nvz)
        for i in eachindex(X), j in eachindex(Z)
            nixz = ni[j]
            Txz = T[i, 1, j]
            Tinv = inv(Txz)
            W = ((vx, vz) -> @SVector[vx - uix[i, 1, j], vz - uiz[i, 1, j]]).(VX, VZ)
            det_T = det(Txz)
            numerator .= dot.(W, Ref(Tinv), W)
            @. result = Ai * nixz / (2π * sqrt(det_T)) * exp(-(Ai * numerator / 2))
            fi_hat[i, 1, j, :, 1, :] .= reshape(result, (length(VX), length(VZ)))
        end

        T = Ti0.(Z)
        Wx = VX .- uix
        Wz = VZ .- uiz
        M = @. Ai * ni / (2pi * T) * exp(-Ai * ((VX - uix)^2 + Wz^2) / (2T))
        vti = sqrt.(T / Ai)
        H3x = He3.(Wx ./ vti)
        H3z = He3.(Wz ./ vti)
        qi = qi0.(X, Z)
        qix = ((s) -> s[1]).(qi)
        qiz = ((s) -> s[2]).(qi)

        return @. fi_hat + M * 2 / (Ai * ni * vti^3 * sqrt(6)) * (qix * H3x + qiz * H3z)
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

    return (; d, sim, fi_0, qi0, P_tensor)
end

end
