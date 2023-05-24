module Helpers

import ..grid1d, ..periodic_grid1d, ..VGrid, ..XGrid, ..Grid, ..Species, ..Simulation, ..SimulationMetadata
using StrideArrays
using RecursiveArrayTools

export y_grid_1d, vy_grid_1v, single_species_1d1v_y, single_species_1d1v_x

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

function single_species_1d1v_x(f, Nx, Nvx, xmin=-1., xmax=1., vxmax=8.0)
    x_grid = x_grid_1d(Nx, xmin, xmax)
    v_grid = vx_grid_1v(Nvx, vxmax, -vxmax)
    grid = Grid(x_grid, v_grid)

    fe = @StrideArray zeros(size(grid))
    fe .= reshape(f.(grid.X, grid.VX), size(grid))

    electrons = Species("electrons", grid, v_grid, [:x], [:vx])
    sim = SimulationMetadata([:x], x_grid, [electrons])
    Simulation(sim, ArrayPartition(fe))
end

function single_species_1d1v_y(f, Ny, Nvy, Ly=2π, vymax=8.0)
    x_grid = y_grid_1d(Ny, Ly)
    v_grid = vy_grid_1v(Nvy, vymax, -vymax)
    grid = Grid(x_grid, v_grid)

    fe = @StrideArray zeros(size(grid))
    fe .= reshape(f.(grid.Y, grid.VY), size(grid))

    electrons = Species("electrons", grid, v_grid, [:y], [:vy])
    sim = SimulationMetadata([:y], x_grid, [electrons])
    Simulation(sim, ArrayPartition(fe))
end

end
