module SpectralBlockingDebugging

using DrWatson
@quickactivate :Braginskii
using Unitful
using Braginskii.Helpers

include("../../campaigns/RT_shared/module.jl")

function make_sim_vlasov(::Val{device}) where {device}
    ωpτ = 20.0
    ωcτ = 1.0
    n0 = 1e22u"m^-3"
    
    Ae = 1/25
    Ai = 1.0
    Zi = 1.0
    Ze = -1.0
    Tref = 1e-4
    vte = sqrt(Tref / Ae)
    vti = sqrt(Tref / Ai)
    fe_eq(z, vx, vz) = Ae / (2pi * Tref) * exp(-Ae*(vx^2 + vz^2)/(2Tref))
    fi_eq(z, vx, vz) = Ai / (2pi * Tref) * exp(-Ai*(vx^2 + vz^2)/(2Tref))
    perturbation_z(z) = 1.0
    perturbation_x(x) = 0.001*sin(8pi * x) + 1e-8*rand()
    Nx = 36
    Nz = 24
    Nvx = 6
    Nvz = 6

    Lx = 1.0
    Lz = 1.0
    
    zmin = -Lz/2
    zmax = Lz/2
    x_grid = Helpers.xz_grid_2d(Nx, Nz, zmin, zmax, Lx, allocator(device))
    (; fe_moments, fi_moments) = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, perturbation_x, perturbation_z,
        x_grid.X, x_grid.Z, Nvx, Nvz, vte, vti)

    dz = x_grid.z.dx
    left_grid = Helpers.xz_grid_2d(Nx, 3, zmin-3dz, zmin, Lx, allocator(device))
    fe_left, fi_left = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, Returns(0.0), Returns(0.0),
        left_grid.X, left_grid.Z, Nvx, Nvz, vte, vti) |> values
    right_grid = Helpers.xz_grid_2d(Nx, 3, zmax, zmax+3dz, Lx, allocator(device))
    fe_right, fi_right = RTShared.vlasov_eq_hermite_expansions(
        fe_eq, fi_eq, Returns(0.0), Returns(0.0),
        right_grid.X, right_grid.Z, Nvx, Nvz, vte, vti) |> values

    sim = Helpers.two_species_xz_2d2v(Val(device),
        (; fe_0=nothing, fi_0=nothing, By0=Returns(1.0));
        Nx, Nz, Nvx, Nvz,
        vdisc=:hermite,
        zmin, zmax,
        ϕ_left=0.0, ϕ_right=0.0, vth_i=vti, vth_e=vte,
        νpτ=0.0, ωpτ, ωcτ, Ze, Zi, Ae, Ai,
        fe_ic=fe_moments, fi_ic=fi_moments,
        gz=0.0, 
        z_bcs=:reservoir,
        ion_bc_lr=(fi_left, fi_right), electron_bc_lr=(fe_left, fe_right)
        );



    problem="spectral_blocking_debugging"
    d = Dict{String, Any}()
    merge!(d, @strdict problem Nx Nz Nvx Nvz Ae ωpτ ωcτ)

    return (; d, sim)
end

end
