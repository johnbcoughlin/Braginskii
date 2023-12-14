module RayleighTaylorDriftKinetic

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu)
    d = Dict{String, Any}()

    problem = "rayleigh_taylor_drift_kinetic"
    Nx = 32
    Nz = 120
    Nμ = 10

    q = 1.0

    mi = 1.0
    me = 0.1

    k = 0.5
    By0 = 1.0

    Lx = 0.75
    Ly = 1.5

    n_ref = 0.5
    T_ref = 5.0

    gz = -1.0

    Kn = 0.01
    α = 25.0 # Sets width of the interface
    δ = 0.1 # Size of velocity perturbation

    merge!(d, @strdict problem Nx Nz Nμ By0 Lx Ly n_ref T_ref q k Kn α gz δ)

    #=============================================
    # Define initial condition functions
    ==============================================#
    
    k = π / Lx
    yr = Ly / 8

    # Density perturbation
    n1(x, z) = begin
        return 0.1 * exp(-z^2 / 2yr^2) * cos(k*x)
    end

    # Density
    n0(x, z) = begin
        return n_ref/2 * tanh(α*z / Ly) + 1.5*n_ref
    end
    # Pressure
    p0(x, z) = begin
        return -gz * n_ref/2 * (-log(cosh(α*z / Ly))/α*Ly - 3*z) + 1.5*n_ref * T_ref
    end
    T0(x, z) = p0(x, z) / n0(x, z)

    # Velocity perturbation
    uz0(x, z) = -δ*cos(k*x)*exp(-z^2/(2*yr^2))

    # Putting it all together in the distribution function
    f_0(x, z, μ) = begin
        return (n0(x, z) + n1(x, z)) / (2π*T0(x, z)) * exp(-μ / T0(x, z))
    end

    sim = Helpers.two_species_2d_drift_kinetic((; fe_0=f_0, fi_0=f_0, By0);
        Nx, Nz, Nμ, μ0=T_ref/(2By0),
        zmin=-Ly, zmax=Ly, Lx=2Lx,
        ϕ_left=1.0, ϕ_right=1.0, vth=sqrt(T_ref),
        ν_p=0.0, qe=-q, qi=q, me, mi, gz, device, z_bcs=:reservoir);

    d = PDEHarness.normalize!(d)
    return (; d, sim)
end

end
