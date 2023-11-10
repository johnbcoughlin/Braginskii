module Helpers

using TimerOutputs

import ..make_bcs
import ..grid1d, ..periodic_grid1d, ..VGrid, ..XGrid, ..Species, ..Simulation, 
..SimulationMetadata, ..CollisionalMoments, ..Hermite, ..WENO5, ..XVDiscretization, ..approximate_f, ..allocator, ..alloc_zeros, ..construct_sim_metadata
import ..plan_ffts
using RecursiveArrayTools
using TimerOutputs

export y_grid_1d, vy_grid_1v, single_species_1d1v_x, single_species_1d1v_y, single_species_1d1v_z, single_species_0d2v,
    x_grid_3d, hermite_disc

#=

1D1V helpers

=#

z_grid_1d(Nz, zmin, zmax, buffer) = begin
    x_grid = periodic_grid1d(1, 0.0)
    y_grid = periodic_grid1d(1, 0.0)
    z_grid = grid1d(Nz, zmin, zmax)

    XGrid(x_grid, y_grid, z_grid, buffer)
end

x_grid_1d(Nx, L, buffer) = begin
    x_grid = periodic_grid1d(Nx, L)
    y_grid = periodic_grid1d(1, 0.0)
    z_grid = grid1d(1, 0.0, 0.0)

    XGrid(x_grid, y_grid, z_grid, buffer)
end

xz_grid_2d(Nx, Nz, zmin, zmax, Lx, buffer) = begin
    x_grid = periodic_grid1d(Nx, Lx)
    y_grid = periodic_grid1d(1, 0.0)
    z_grid = grid1d(Nz, zmin, zmax)
    XGrid(x_grid, y_grid, z_grid, buffer)
end

y_grid_1d(Ny, L, buffer) = begin
    x_grid = periodic_grid1d(1, 0.0)
    y_grid = periodic_grid1d(Ny, L)
    z_grid = grid1d(1, 0.0, 0.0)

    XGrid(x_grid, y_grid, z_grid, buffer)
end

vz_grid_1v(Nvz, vzmax, buffer) = begin
    vy_grid = grid1d(1, 0.0, 0.0)
    vx_grid = grid1d(1, 0.0, 0.0)
    vz_grid = grid1d(Nvz, vzmin, vzmax)
    VGrid([:vz], vx_grid, vy_grid, vz_grid, buffer)
end

vy_grid_1v(Nvy, vymax, buffer) = begin
    vx_grid = grid1d(1, 0.0, 0.0)
    vy_grid = grid1d(Nvy, vymin, vymax)
    vz_grid = grid1d(1, 0.0, 0.0)
    VGrid([:vy], vx_grid, vy_grid, vz_grid, buffer)
end

hermite_disc(; Nvx=1, Nvy=1, Nvz=1, vth=1.0, device, kwargs...) = begin
    Hermite(Nvx, Nvy, Nvz, vth, device)
end

weno_v_disc(dims; Nvx=1, Nvy=1, Nvz=1, vxmax=0.0, vymax=0.0, vzmax=0.0, 
    vxmin=-vxmax, vymin=-vymax, vzmin=-vzmax, vth=1.0, buffer, kwargs...) = begin
    vx_grid = grid1d(Nvx, vxmin*vth, vxmax*vth)
    vy_grid = grid1d(Nvy, vymin*vth, vymax*vth)
    vz_grid = grid1d(Nvz, vzmin*vth, vzmax*vth)
    vgrid = VGrid(dims, vx_grid, vy_grid, vz_grid, buffer)
    WENO5(vgrid)
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

function single_species_1d1v_z(f; Nz, Nvz,
    zmin=-1., zmax=1., vdisc, vzmax=8.0,
    free_streaming=true, q=1.0, ϕ_left=0., ϕ_right=0., ν_p=0.0,
    device=:cpu, vth=1.0, gz=0.0, z_bcs=:reflecting)
    buffer = allocator(device)

    x_grid = z_grid_1d(Nz, zmin, zmax, buffer)

    v_disc = v_discretization(vdisc, [:vz]; Nvz, vzmax, buffer, vth, device)
    disc = XVDiscretization(x_grid, v_disc)
    bcs = make_bcs(x_grid, v_disc, f, buffer, z_bcs)

    fe = approximate_f(f, disc, (3, 6), buffer)

    By = alloc_zeros(Float64, buffer, size(x_grid)...)
    ϕl = alloc_zeros(Float64, buffer, 1, 1)
    ϕl .= ϕ_left
    ϕr = alloc_zeros(Float64, buffer, 1, 1)
    ϕr .= ϕ_right

    electrons = Species("electrons", [:z], [:vz], q, 1.0, plan_ffts(disc, buffer), disc, bcs)

    sim = construct_sim_metadata(
        [:z], x_grid, (electrons,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)
    Simulation(sim, ArrayPartition(fe))
end

function single_species_1d1v_x(f; Nx, Nvx, Lx=2π, vxmax=8.0, q=1.0, ν_p=0.0, gz=0.0, 
    vdisc, free_streaming=true, device=:cpu, vth=1.0, z_bcs=nothing)
    buffer = allocator(device)
    @timeit "xgrid" x_grid = x_grid_1d(Nx, Lx, buffer)

    @timeit "vdisc" v_disc = v_discretization(vdisc, [:vx]; Nvx, vxmax, buffer, vth, device)
    disc = XVDiscretization(x_grid, v_disc)

    fe = approximate_f(f, disc, (1, 4), buffer)

    By = alloc_zeros(Float64, buffer, size(x_grid)...)
    ϕl = alloc_zeros(Float64, buffer, Nx, 1)
    ϕr = alloc_zeros(Float64, buffer, Nx, 1)

    electrons = Species("electrons", [:x], [:vx], q, 1.0, plan_ffts(disc, buffer), disc, nothing)
    sim = construct_sim_metadata(
        [:x], x_grid, (electrons,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)
    Simulation(sim, ArrayPartition(fe))
end

function single_species_1d1v_y(f; Ny, Nvy, Ly=2π, vymax=8.0, q=1.0, ν_p=0.0, vdisc, free_streaming=true,
    device=:cpu, vth=1.0, gz=0.0, z_bcs=nothing)
    buffer = allocator(device)
    x_grid = y_grid_1d(Ny, Ly, buffer)

    v_disc = v_discretization(vdisc, [:vy]; Nvy, vymax, buffer, vth, device)
    disc = XVDiscretization(x_grid, v_disc)

    fe = approximate_f(f, disc, (2, 5), buffer)

    By = alloc_zeros(Float64, buffer, size(x_grid)...)
    ϕl = alloc_zeros(Float64, buffer, 1, Ny)
    ϕr = alloc_zeros(Float64, buffer, 1, Ny)

    electrons = Species("electrons", [:y], [:vy], q, 1.0, plan_ffts(disc, buffer), disc, nothing)
    sim = construct_sim_metadata(
        [:y], x_grid, (electrons,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)
    Simulation(sim, ArrayPartition(fe))
end

#=

0D2V helpers

=#

x_grid_0d(buffer) = begin
    XGrid(periodic_grid1d(1, 0.0), periodic_grid1d(1, 0.0), grid1d(1, 0., 0.), buffer)
end

vxvy_grid_2v(Nvx, Nvy, vxmax, vymax, buffer) = begin
    vx_grid = grid1d(Nvx, vxmin, vxmax)
    vy_grid = grid1d(Nvy, vymin, vymax)
    vz_grid = grid1d(1, 0.0, 0.0)
    VGrid([:vx, :vy], vx_grid, vy_grid, vz_grid, buffer)
end

function single_species_0d2v((; f, By), Nvx, Nvz; vxmax=5.0, vzmax=5.0, 
    q=1.0, ν_p=0.0, gz=0.0, vdisc, free_streaming=true, vth=1.0, device=:cpu, z_bcs=:reflecting)
    buffer = allocator(device)
    x_grid = x_grid_0d(buffer)

    v_disc = v_discretization(vdisc, [:vx, :vz]; Nvx, Nvz, vxmax, vzmax, vth, buffer, device)
    bcs = make_bcs(x_grid, vdisc, f, buffer, z_bcs)
    disc = XVDiscretization(x_grid, v_disc)

    fe = approximate_f(f, disc, (4, 6), buffer)

    By0 = alloc_zeros(Float64, buffer, size(x_grid)...)
    By0 .= (By::Number)
    ϕl = alloc_zeros(Float64, buffer, 1, 1)
    ϕr = alloc_zeros(Float64, buffer, 1, 1)

    electrons = Species("electrons", Symbol[], [:vx, :vz], q, 1.0, plan_ffts(disc, buffer), disc, bcs)
    sim = construct_sim_metadata(
        Symbol[], x_grid, (electrons,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)
    Simulation(sim, ArrayPartition(fe))
end

# 3D

function x_grid_3d(Nx, Ny, Nz, buffer=allocator(:cpu))
    x_grid = periodic_grid1d(Nx, 2π)
    y_grid = periodic_grid1d(Ny, 2π)
    z_grid = grid1d(Nz, -1., 1.)

    XGrid(x_grid, y_grid, z_grid, buffer)
end

# 2D2V

function single_species_xz_2d2v((; f_0, By0); Nx, Nz, Nvx, Nvz, 
    q=1.0, ν_p=0.0, gz=0.0, vdisc, free_streaming=true, 
    device=:cpu, vth=1.0, z_bcs=:reflecting,
    Lx=2π, zmin=-1.0, zmax=1.0,
    ϕ_left, ϕ_right
    )
    buffer = allocator(device)
    x_grid = xz_grid_2d(Nx, Nz, zmin, zmax, Lx, buffer)

    By = alloc_zeros(Float64, buffer, size(x_grid)...)
    By .= (By0::Number)

    ϕl = alloc_zeros(Float64, buffer, Nx, 1)
    ϕl .= ϕ_left
    ϕr = alloc_zeros(Float64, buffer, Nx, 1)
    ϕr .= ϕ_right

    v_disc = v_discretization(vdisc, [:vx, :vz]; Nvx, Nvz, vxmax=8.0, vzmax=8.0, buffer, vth, device)
    bcs = make_bcs(x_grid, v_disc, f_0, buffer, z_bcs)
    ion_disc = XVDiscretization(x_grid, v_disc)

    @timeit "approx" fi = approximate_f(f_0, ion_disc, (1, 3, 4, 6), buffer)
    ions = Species("ions", [:x, :z], [:vx, :vz], q, 1.0,
        plan_ffts(ion_disc, buffer), ion_disc, bcs)

    sim = construct_sim_metadata(
        Symbol[:x, :z], x_grid, (ions,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)

    Simulation(sim, ArrayPartition(fi))
end

function two_species_xz_2d2v((; fe_0, fi_0, By0); Nx, Nz, Nvx, Nvz, 
    q=1.0, ν_p=0.0, vdisc, free_streaming=true, 
    device=:cpu, vth=1.0, gz=0.0,
    Lx=2π, zmin=-1.0, zmax=1.0,
    ϕ_left, ϕ_right
    )
    buffer = allocator(device)
    x_grid = xz_grid_2d(Nx, Nz, zmin, zmax, Lx, buffer)

    By = alloc_zeros(Float64, buffer, size(x_grid)...)
    By .= (By0::Number)

    ϕl = alloc_zeros(Float64, buffer, Nx, 1)
    ϕl .= ϕ_left
    ϕr = alloc_zeros(Float64, buffer, Nx, 1)
    ϕr = ϕ_right

    ve_disc = v_discretization(vdisc, [:vx, :vz]; Nvx, Nvz, vxmax=8.0, vzmax=8.0, buffer, vth)
    electron_disc = XVDiscretization(x_grid, ve_disc)
    @timeit "approx" fe = approximate_f(fe_0, electron_disc, (1, 3, 4, 6), buffer)
    electrons = Species("electrons", [:x, :z], [:vx, :vz], q, 1.0, 
        plan_ffts(electron_disc, buffer), electron_disc)

    vi_disc = v_discretization(vdisc, [:vx, :vz]; Nvx, Nvz, vxmax=8.0, vzmax=8.0, buffer, vth)
    ion_disc = XVDiscretization(x_grid, vi_disc)
    @timeit "approx" fi = approximate_f(fi_0, ion_disc, (1, 3, 4, 6), buffer)
    ions = Species("ions", [:x, :z], [:vx, :vz], q, 1.0,
        plan_ffts(ion_disc, buffer), ion_disc)

    sim = construct_sim_metadata(
        [:x, :z], x_grid, (ions,), free_streaming, By, ϕl, ϕr, ν_p, gz, device, buffer)
    Simulation(sim, ArrayPartition(fe, fi))
end

end
