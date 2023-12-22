module ElectrostaticFlute2D2V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim_maxwellians(device=:cpu)
    d = Dict{String, Any}()

    problem = "electrostatic_flute_drift_kinetic"
    Nx = 32
    Nz = 80
    Nμ = 10
    Nvx = 10
    Nvz = 10

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-3 * 5

    r0 = 10.0*Lz
    B_ref = 1.0

    α = 10.0

    Ai = 1.0
    Ae = 1/20
    Zi = 1.0 * 60
    Ze = -Zi

    μ0 = T_ref / B_ref
    @show μ0

    ωcτ = 1.0
    ωpτ = 10.0

    kx = 1
    
    merge!(d, @strdict problem Nx Nz Nμ B_ref r0 α Lx Lz n_ref T_ref kx)

    n0(z) = begin
        return 0.5 * n_ref * tanh(-α*z/Lz) + 1.5*n_ref
        #return n_ref
    end
    n1(x, z) = begin
        return 0.1 * n_ref * exp(-(z/Lz)^2/0.01) * cos(kx * x * 2π / Lx)
    end
    n(x, z) = n0(z) + n1(x, z)

    By0(x, z) = r0 * B_ref / (z + r0)
    grad_p(z) = begin
        -0.5 * T_ref * n_ref * α/Lz * sech(-α*z/Lz)^2
    end
    uix0(z) = begin
        0*grad_p(z) / (n0(z) * By0(0.0, z) * Zi)
    end

    # Electron drift kinetic distribution
    Fe_0(Rx, Rz, μ) = begin
        n(Rx, Rz) * Ae / (2π * T_ref) * exp(-By0(0.0, Rz) * μ / T_ref)
    end

    # x component of Canonical momentum
    Px(z, vx) = begin
        Ax = -Zi * r0 * B_ref * log(r0 + z)
        return Ai * vx + Ax
    end

    fi_0(x, z, vx, vz) = begin
        n(x, z) * Ai / (2π * T_ref) * exp(-((vx-uix0(z))^2 + vz^2) / (2T_ref))
    end
    Fi_0(Rx, Rz, μ) = begin
        n(Rx, Rz) * Ai / (2π * T_ref) * exp(-By0(0.0, Rz) * μ / T_ref)
    end

    sim = Helpers.two_species_2d_vlasov_dk_hybrid((; Fe_0, fi_0, By0);
        Nx, Nz, Nμ, Nvx, Nvz, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx=Lx,
        ϕ_left=0.0, ϕ_right=0.0, vth=sqrt(T_ref),
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai, 
        gz=0.0, device, z_bcs=:reservoir);
    #=
    sim = Helpers.two_species_2d_drift_kinetic((; fe_0=Fe_0, fi_0=Fi_0, By0);
        Nx, Nz, Nμ, μ0,
        zmin=-Lz/2, zmax=Lz/2, Lx=Lx,
        ϕ_left=1.0, ϕ_right=1.0, vth=sqrt(T_ref),
        ν_p=0.0, ωpτ, ωcτ, qe=Ze, qi=Zi, me=Ae, mi=Ai, 
        gz=0.0, device, z_bcs=:reservoir);
        =#

    d = PDEHarness.normalize!(d)
    return (; d, sim)
end


function make_sim_equilibrium(device=:cpu)
    d = Dict{String, Any}()

    problem = "electrostatic_flute_drift_kinetic"
    Nz = 80
    Nμ = 10
    Nvx = 10
    Nvz = 10

    Lz = 1.0
    Lx = 1.0

    n_ref = 1.0
    T_ref = 1e-3 * 5

    r0 = 10.0*Lz
    B_ref = 1.0

    α = 10.0

    Ai = 1.0
    Ae = 1/20
    Zi = 1.0 * 60
    Ze = -Zi

    μ0 = T_ref / B_ref
    @show μ0

    ωcτ = 1.0
    ωpτ = 10.0

    kx = 1
    
    merge!(d, @strdict problem Nz Nμ B_ref r0 α Lx Lz n_ref T_ref kx)

    n0(z) = begin
        return 0.5 * n_ref * tanh(-α*z/Lz) + 1.5*n_ref
        #return n_ref
    end

end


end

