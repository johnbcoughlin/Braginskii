module RayleighTaylorDriftKinetic

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

function make_sim(device=:cpu)
    d = Dict{String, Any}()

    problem = "rayleigh_taylor_drift_kinetic"
    Nx = 32
    Nz = 160
    Nμ = 3

    q = 1000.0

    mi = 1.0
    me = 0.5

    Bref = 50.0

    Lx = 0.375 / 4
    Ly = 1.5

    k = π / Lx

    n_ref = 0.5
    T_ref = 5.0

    gz = -1.0

    Kn = 0.01
    α = 25.0 # Sets width of the interface
    δ = 0.1 # Size of velocity perturbation

    merge!(d, @strdict problem Nx Nz Nμ Bref Lx Ly n_ref T_ref q k Kn α gz δ)

    #=============================================
    # Define initial condition functions
    ==============================================#
    
    yr = Ly / 8

    # Density perturbation
    n1(x, z) = begin
        return 0.1 * exp(-z^2 / 2yr^2) * cos(k*x)
    end

    # Density
    n0(x, z) = begin
        return n_ref/2 * tanh(α*(z - 0.1cos(k*x)) / Ly) + 1.5*n_ref
    end
    @show n0(0.0, 1.0)
    # Pressure
    p0(x, z) = begin
        return -gz * n_ref/2 * (-log(cosh(α*z / Ly))/α*Ly - 3*z) + 1.5*n_ref * T_ref
    end

    By0(x, z) = Bref
    T0(x, z) = p0(x, z) / n0(x, z)

    μ0 = T_ref / (2Bref)

    # Velocity perturbation
    uz0(x, z) = -δ*cos(k*x)*exp(-z^2/(2*yr^2))

    # Putting it all together in the distribution function
    f_0(m, x, z, μ) = begin
        return (n0(x, z)) / (2π*T0(x, z)/m) * exp(-Bref * (μ / μ0) / T0(x, z))
    end
    fe_0(args...) = f_0(me, args...)
    fi_0(args...) = f_0(mi, args...)

    sim = Helpers.two_species_2d_drift_kinetic((; fe_0, fi_0, By0);
        Nx, Nz, Nμ, μ0,
        zmin=-Ly, zmax=Ly, Lx=2Lx,
        ϕ_left=1.0, ϕ_right=1.0, vth=sqrt(T_ref),
        ν_p=0.0, qe=-q, qi=q, me, mi, gz, device, z_bcs=:reservoir);

    d = PDEHarness.normalize!(d)
    return (; d, sim)
end

end
