module RayleighTaylor2D2V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness

preferred_device() = Sys.isapple() ? :cpu : :gpu

function make_sim(; Kn, f_ic)
    device = preferred_device()

    #=============================================
    # Set up the scalar parameters of the problem
    ==============================================#
    d = Dict{String, Any}()

    problem = "rayleigh_taylor"
    Nx = 16
    Nz = 120
    Nvx = 20
    Nvz = 20
    q = 0.0
    k = 0.5
    By0 = 0.0

    Lx = 0.75
    Ly = 1.5

    n_ref = 0.5
    T_ref = 5.0

    gz = -1.0

    α = 25.0 # Sets width of the interface
    δ = 0.1 # Size of velocity perturbation

    merge!(d, @strdict problem Nx Nz Nvx Nvz By0 Lx Ly n_ref T_ref q k Kn α gz δ)

    #=============================================
    # Define initial condition functions
    ==============================================#
    
    # Density
    n0(x, z) = begin
        return n_ref/2 * tanh(α*z / Ly) + 1.5*n_ref
    end
    # Pressure
    p0(x, z) = begin
        return -gz * n_ref/2 * (-log(cosh(α*z / Ly))/α*Ly - 3*z) + 1.5*n_ref * T_ref
    end
    T0(x, z) = p0(x, z) / n0(x, z)

    k = π / Lx
    yr = Ly / 10

    # Velocity perturbation
    uz0(x, z) = -δ*cos(k*x)*exp(-z^2/(2*yr^2))

    # Putting it all together in the distribution function
    f_0(x, z, vx, vz) = begin
        return n0(x, z) / (2π*T0(x, z)) * exp(-(vx^2 + (vz-uz0(x, z))^2) / (2T0(x, z)))
    end

    #=============================================
    # Create the simulation object
    ==============================================#

    λ_mfp = Kn * Lx
    vth = sqrt(T_ref)
    ν_p = vth / λ_mfp

    sim = Helpers.single_species_xz_2d2v((; f_0, By0);
        Nx, Nz, Nvx, Nvz,
        zmin=-Ly, zmax=Ly, Lx=2*Lx,
        vdisc=:hermite, ϕ_left=1.0, ϕ_right=1.0, vth=sqrt(T_ref),
        ν_p, q, gz, device, z_bcs=:reservoir, f_ic);
    
    #=============================================
    # Return values
    ==============================================#

    # Determine the characteristic timescale of the instability growth.
    A = (1 - n_ref) / (1 + n_ref)
    @show tau = 1 / sqrt(k*A*abs(gz))

    dt_advection = (Ly / Nz) / (2vth)
    @show dt_advection
    dt_collisions = 1 / (ν_p * max(Nx, Nz)) * 2 # Factor of 2 is approximate
    @show dt_collisions

    dt = min(dt_advection, dt_collisions)

    return (; d, sim, tau, dt)
end

function ckpt_entrypoint()
    (; d, sim, tau) = make_sim()
end

end
