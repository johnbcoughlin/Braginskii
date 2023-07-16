#==============================
# Physical space discretization
===============================#

struct Grid1D
    N::Int
    dx::Float64
    min::Float64
    max::Float64
    nodes::Vector{Float64}
end

# A periodic grid from [0, L), with nodes at 0, Δx, 2Δx, ...
struct PeriodicGrid1D
    N::Int
    dx::Float64
    L::Float64
    nodes::Vector{Float64}
end

# Create a non-periodic uniform grid, with points at cell centers.
grid1d(N, min, max) = begin
    dx = (max - min) / N
    cell_centers = collect(LinRange(min+dx/2, max-dx/2, N))
    Grid1D(N, dx, min, max, cell_centers)
end

periodic_grid1d(N, L) = begin
    dx = L / N
    nodes = collect((0:N-1) * dx)
    PeriodicGrid1D(N, dx, L, nodes)
end

struct XGrid{XA, YA, ZA}
    x::Grid1D
    y::PeriodicGrid1D
    z::PeriodicGrid1D

    X::XA
    Y::YA
    Z::ZA

    XGrid(xgrid, ygrid, zgrid, buffer) = begin
        X = alloc_zeros(Float64, buffer, length(xgrid.nodes), 1, 1)
        X .= reshape(xgrid.nodes, (:, 1, 1))

        Y = alloc_zeros(Float64, buffer, 1, length(ygrid.nodes), 1)
        Y .= reshape(ygrid.nodes, (1, :, 1))

        Z = alloc_zeros(Float64, buffer, 1, 1, length(zgrid.nodes))
        Z .= reshape(zgrid.nodes, (1, 1, :))

        new{typeof(X), typeof(Y), typeof(Z)}(xgrid, ygrid, zgrid, X, Y, Z)
    end
end

size(grid::XGrid) = (grid.x.N, grid.y.N, grid.z.N)

#==============================
# Velocity space discretizations
===============================#

export approximate_f

struct VGrid{XA, YA, ZA}
    x::Grid1D
    y::Grid1D
    z::Grid1D

    VX::XA
    VY::YA
    VZ::ZA

    VGrid(dims, x, y, z, buffer) = begin
        # Check that it's suitable for reflecting wall BCs
        if :x ∈ dims
            @assert iseven(x.N) && x.max == -x.min
        end

        X = alloc_zeros(Float64, buffer, 1, 1, 1, length(x.nodes), 1, 1)
        X .= reshape(x.nodes, (1, 1, 1, :, 1, 1))

        Y = alloc_zeros(Float64, buffer, 1, 1, 1, 1, length(y.nodes), 1)
        Y .= reshape(y.nodes, (1, 1, 1, 1, :, 1))

        Z = alloc_zeros(Float64, buffer, 1, 1, 1, 1, 1, length(z.nodes))
        Z .= reshape(z.nodes, (1, 1, 1, 1, 1, :))

        new{typeof(X), typeof(Y), typeof(Z)}(x, y, z, X, Y, Z)
    end
end

size(grid::VGrid) = (grid.x.N, grid.y.N, grid.z.N)

struct WENO5
    grid::VGrid
end

size(fd::WENO5) = size(fd.grid)

struct Hermite
    Nvx::Int
    Nvy::Int
    Nvz::Int

    vth::Float64
end

size(hd::Hermite) = (hd.Nvx, hd.Nvy, hd.Nvz)

struct XVDiscretization{VDISC}
    x_grid::XGrid
    vdisc::VDISC
end

size(disc::XVDiscretization) = tuple(size(disc.x_grid)..., size(disc.vdisc)...)

approximate_f(f, disc::XVDiscretization, dims, buffer) = begin
    result = alloc_zeros(Float64, buffer, size(disc)...)
    approximate_f!(result, f, disc, dims)
    result
end

approximate_f!(result, f, disc::XVDiscretization{WENO5}, dims) = begin
    all_dimensions = [disc.x_grid.X, disc.x_grid.Y, disc.x_grid.Z, disc.vdisc.grid.VX, disc.vdisc.grid.VY, disc.vdisc.grid.VZ]
    dimensions = all_dimensions[[dims...]]
    result .= f.(dimensions...)
end

approximate_f!(result, f, disc::XVDiscretization{Hermite}, dims) = begin
    vdims = sum(dims .>= 4)
    factor = 1 / (2π)^((3-vdims)/2)
    f_all(args...) = factor * f((args[dim] for dim in dims)...)
    hd = disc.vdisc
    (; Nvx, Nvy, Nvz, vth) = hd
    (; X, Y, Z) = disc.x_grid
    result .= Float64.(bigfloat_weighted_hermite_expansion(f_all, Nvx-1, Nvy-1, Nvz-1, X, Y, Z, vth))
end
