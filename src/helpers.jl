module Helpers

import ..grid1d, ..periodic_grid1d, ..VGrid, ..XGrid, ..Species, ..Simulation, ..SimulationMetadata, ..CollisionalMoments, ..Hermite, ..WENO5, ..XVDiscretization, ..approximate_f
import ..plan_ffts
using RecursiveArrayTools

export y_grid_1d, vy_grid_1v, single_species_1d1v_y, single_species_1d1v_x, single_species_0d2v,
    x_grid_3d, hermite_disc

#=

1D1V helpers

=#

x_grid_1d(Nx, xmin, xmax) = begin
    x_grid = grid1d(Nx, xmin, xmax)
    y_grid = periodic_grid1d(1, 0.0)
    z_grid = periodic_grid1d(1, 0.0)

    XGrid(x_grid, y_grid, z_grid)
end

y_grid_1d(Ny, L) = begin
    x_grid = grid1d(1, 0.0, 0.0)
    y_grid = periodic_grid1d(Ny, L)
    z_grid = periodic_grid1d(1, 0.0)

    XGrid(x_grid, y_grid, z_grid)
end

vx_grid_1v(Nvx, vxmax, vxmin=-vxmax) = begin
    vx_grid = grid1d(Nvx, vxmin, vxmax)
    vy_grid = grid1d(1, 0.0, 0.0)
    vz_grid = grid1d(1, 0.0, 0.0)
    VGrid([:x], vx_grid, vy_grid, vz_grid)
end

vy_grid_1v(Nvy, vymax, vymin=-vymax) = begin
    vx_grid = grid1d(1, 0.0, 0.0)
    vy_grid = grid1d(Nvy, vymin, vymax)
    vz_grid = grid1d(1, 0.0, 0.0)
    VGrid([:y], vx_grid, vy_grid, vz_grid)
end

hermite_disc(; Nvx=1, Nvy=1, Nvz=1, vth=1.0) = begin
    Hermite(Nvx, Nvy, Nvz, vth)
end

weno_v_disc(dims; Nvx=1, Nvy=1, Nvz=1, vxmax=0.0, vymax=0.0, vzmax=0.0, 
    vxmin=-vxmax, vymin=-vymax, vzmin=-vzmax) = begin
    vx_grid = grid1d(Nvx, vxmin, vxmax)
    vy_grid = grid1d(Nvy, vymin, vymax)
    vz_grid = grid1d(Nvz, vzmin, vzmax)
    vgrid = VGrid(dims, vx_grid, vy_grid, vz_grid)
    WENO5(vgrid)
end

function collisional_moments(xgrid, species)
    result = Dict{Tuple{String, String}, CollisionalMoments}()
    for α in species
        for β in species
            ux = zeros(size(xgrid))
            uy = zeros(size(xgrid))
            uz = zeros(size(xgrid))
            T = zeros(size(xgrid))
            ν = zeros(size(xgrid))
            cm = CollisionalMoments(ux, uy, uz, T, ν)
            push!(result, (α, β) => cm)
        end
    end
    result
end

v_discretization(method, dims; kwargs...) = begin
    if method == :weno
        weno_v_disc(dims; kwargs...)
    elseif method == :hermite
        hermite_disc(; kwargs...)
    else
        error("Unknown v discretization $method")
    end
end

function single_species_1d1v_x(f; Nx, Nvx,
    xmin=-1., xmax=1., vdisc, vxmax=8.0,
    free_streaming=true, q=1.0, ϕ_left=0., ϕ_right=0., ν_p=0.0,
    device=:cpu)
    x_grid = x_grid_1d(Nx, xmin, xmax)

    v_disc = v_discretization(vdisc, [:x]; Nvx, vxmax)
    disc = XVDiscretization(x_grid, v_disc)

    fe = zeros(size(disc))
    fe = approximate_f(f, disc, (1, 4))

    Bz = zeros(size(x_grid))
    ϕl = zeros(1, 1)
    ϕl .= ϕ_left
    ϕr = zeros(1, 1)
    ϕr .= ϕ_right
    ϕ = zeros(size(x_grid))

    electrons = Species("electrons", [:x], [:vx], q, 1.0, plan_ffts(disc), disc)
    cms = collisional_moments(x_grid, ["electrons"])
    sim = SimulationMetadata([:x], x_grid, Bz, ϕl, ϕr, ϕ, free_streaming, 
        ν_p, cms, (electrons,), plan_ffts(x_grid), device)
    Simulation(sim, ArrayPartition(fe))
end

function single_species_1d1v_y(f; Ny, Nvy, Ly=2π, vymax=8.0, q=1.0, ν_p=0.0, vdisc, free_streaming=true,
    device=:cpu)
    x_grid = y_grid_1d(Ny, Ly)

    v_disc = v_discretization(vdisc, [:y]; Nvy, vymax)
    disc = XVDiscretization(x_grid, v_disc)

    fe = approximate_f(f, disc, (2, 5))

    Bz = zeros(size(x_grid))
    ϕl = zeros(Ny, 1)
    ϕr = zeros(Ny, 1)
    ϕ = zeros(size(x_grid))

    electrons = Species("electrons", [:y], [:vy], q, 1.0, plan_ffts(disc), disc)
    cms = collisional_moments(x_grid, ["electrons"])
    sim = SimulationMetadata([:y], x_grid, Bz, ϕl, ϕr, ϕ, free_streaming, 
        ν_p, cms, (electrons,), plan_ffts(x_grid), device)
    Simulation(sim, ArrayPartition(fe))
end

#=

0D2V helpers

=#

x_grid_0d() = begin
    XGrid(grid1d(1, 0., 0.), periodic_grid1d(1, 0.0), periodic_grid1d(1, 0.0))
end

vxvy_grid_2v(Nvx, Nvy, vxmax, vymax, vxmin=-vxmax, vymin=-vymax) = begin
    vx_grid = grid1d(Nvx, vxmin, vxmax)
    vy_grid = grid1d(Nvy, vymin, vymax)
    vz_grid = grid1d(1, 0.0, 0.0)
    VGrid([:x, :y], vx_grid, vy_grid, vz_grid)
end

function single_species_0d2v((; f, Bz), Nvx, Nvy; vxmax=8.0, vymax=8.0, 
    q=1.0, ν_p=0.0, vdisc, free_streaming=true, device=:cpu)
    x_grid = x_grid_0d()

    v_disc = v_discretization(vdisc, [:x, :y]; Nvx, Nvy, vxmax, vymax)
    disc = XVDiscretization(x_grid, v_disc)

    fe = approximate_f(f, disc, (4, 5))

    Bz0 = zeros(size(x_grid))
    Bz0 .= (Bz::Number)
    ϕl = zeros(1, 1)
    ϕr = zeros(1, 1)
    ϕ = zeros(size(x_grid))

    electrons = Species("electrons", Symbol[], [:vx, :vy], q, 1.0, plan_ffts(disc), disc)
    cms = collisional_moments(x_grid, ["electrons"])
    sim = SimulationMetadata(Symbol[], x_grid, Bz0, ϕl, ϕr, ϕ, 
        free_streaming, ν_p, cms, (electrons,), plan_ffts(x_grid), device)
    Simulation(sim, ArrayPartition(fe))
end

# 3D

function x_grid_3d(Nx, Ny, Nz)
    x_grid = grid1d(Nx, -1., 1.)
    y_grid = periodic_grid1d(Ny, 2π)
    z_grid = periodic_grid1d(Nz, 2π)

    XGrid(x_grid, y_grid, z_grid)
end

end
