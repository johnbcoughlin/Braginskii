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
    x::PeriodicGrid1D
    y::PeriodicGrid1D
    z::Grid1D

    X::XA
    Y::YA
    Z::ZA

    XGrid(xgrid, ygrid, zgrid, buffer) = begin
        X = alloc_zeros(Float64, buffer, length(xgrid.nodes), 1, 1)
        copyto!(X, reshape(xgrid.nodes, (:, 1, 1)))

        Y = alloc_zeros(Float64, buffer, 1, length(ygrid.nodes), 1)
        copyto!(Y, reshape(ygrid.nodes, (1, :, 1)))

        Z = alloc_zeros(Float64, buffer, 1, 1, length(zgrid.nodes))
        copyto!(Z, reshape(zgrid.nodes, (1, 1, :)))

        new{typeof(X), typeof(Y), typeof(Z)}(xgrid, ygrid, zgrid, X, Y, Z)
    end
end

size(grid::XGrid) = (grid.x.N, grid.y.N, grid.z.N)

#==============================
# Velocity space discretizations
===============================#

export approximate_f, vgrid_of, expand_f

struct VGrid{XA, YA, ZA}
    x::Grid1D
    y::Grid1D
    z::Grid1D

    VX::XA
    VY::YA
    VZ::ZA

    VGrid(dims, x, y, z, buffer) = begin
        @assert dims ⊆ [:vx, :vy, :vz]
        # Check that it's suitable for reflecting wall BCs
        if :vx ∈ dims
            @assert iseven(x.N) && x.max == -x.min
        end

        X = alloc_zeros(Float64, buffer, 1, 1, 1, length(x.nodes), 1, 1)
        copyto!(X, reshape(x.nodes, (1, 1, 1, :, 1, 1)))

        Y = alloc_zeros(Float64, buffer, 1, 1, 1, 1, length(y.nodes), 1)
        copyto!(Y, reshape(y.nodes, (1, 1, 1, 1, :, 1)))

        Z = alloc_zeros(Float64, buffer, 1, 1, 1, 1, 1, length(z.nodes))
        copyto!(Z, reshape(z.nodes, (1, 1, 1, 1, 1, :)))

        new{typeof(X), typeof(Y), typeof(Z)}(x, y, z, X, Y, Z)
    end
end

size(grid::VGrid) = (grid.x.N, grid.y.N, grid.z.N)

struct WENO5
    grid::VGrid
end

size(fd::WENO5) = size(fd.grid)

struct Hermite{SPARSE}
    Nvx::Int
    Nvy::Int
    Nvz::Int

    vth::Float64

    Ξx::SPARSE
    Ξy::SPARSE
    Ξz::SPARSE

    Ξx⁻::SPARSE
    Ξy⁻::SPARSE
    Ξz⁻::SPARSE

    Ξx⁺::SPARSE
    Ξy⁺::SPARSE
    Ξz⁺::SPARSE

    Dvx::SPARSE
    Dvy::SPARSE
    Dvz::SPARSE
end

Hermite(Nvx, Nvy, Nvz, vth, device) = begin
    N = max(Nvx, Nvy, Nvz)

    Ξ = spdiagm(-1 => sqrt.(1:N-1), 1 => sqrt.(1:N-1))
    
    Ξx = Ξ[1:Nvx, 1:Nvx]
    Ξy = Ξ[1:Nvy, 1:Nvy]
    Ξz = Ξ[1:Nvz, 1:Nvz]

    Λx, Rx = eigen(Array(Ξx))
    Ξx⁻ = kron(I(Nvz), I(Nvy), sparsify(Rx * Diagonal(min.(Λx, 0.0)) / Rx))
    Ξx⁺ = kron(I(Nvz), I(Nvy), sparsify(Rx * Diagonal(max.(Λx, 0.0)) / Rx))

    Λy, Ry = eigen(Array(Ξy))
    Ξy⁻ = kron(I(Nvz), sparsify(Ry * Diagonal(min.(Λy, 0.0)) / Ry), I(Nvx))
    Ξy⁺ = kron(I(Nvz), sparsify(Ry * Diagonal(max.(Λy, 0.0)) / Ry), I(Nvx))

    Λz, Rz = eigen(Array(Ξz))
    Ξz⁻ = kron(sparsify(Rz * Diagonal(min.(Λz, 0.0)) / Rz), I(Nvy), I(Nvx))
    Ξz⁺ = kron(sparsify(Rz * Diagonal(max.(Λz, 0.0)) / Rz), I(Nvy), I(Nvx))

    D = spdiagm(-1 => -sqrt.(1:N-1))

    Ξx = kron(I(Nvz), I(Nvy), Ξ[1:Nvx, 1:Nvx])
    Ξy = kron(I(Nvz), Ξ[1:Nvy, 1:Nvy], I(Nvx))
    Ξz = kron(Ξ[1:Nvz, 1:Nvz], I(Nvy), I(Nvx))

    Dvx = kron(I(Nvz), I(Nvy), D[1:Nvx, 1:Nvx])
    Dvy = kron(I(Nvz), D[1:Nvy, 1:Nvy], I(Nvx))
    Dvz = kron(D[1:Nvz, 1:Nvz], I(Nvy), I(Nvx))

    cx = if device == :cpu
        identity
    elseif device == :gpu
        CuSparseMatrixCSC
    end

    Hermite(Nvx, Nvy, Nvz, vth, cx(Ξx), cx(Ξy), cx(Ξz), cx(Ξx⁻), cx(Ξy⁻), cx(Ξz⁻), 
        cx(Ξx⁺), cx(Ξy⁺), cx(Ξz⁺), cx(Dvx), cx(Dvy), cx(Dvz))
end

function sparsify(A)
    A[abs.(A) .< 1e-15] .= 0.0
    sparse(A)
end

size(hd::Hermite) = (hd.Nvx, hd.Nvy, hd.Nvz)

function vgrid_of(fd::WENO5, args...)
    return fd.grid
end

function vgrid_of(hd::Hermite, K=50, vmax=5.0, buffer=default_buffer())
    vmax = vmax*hd.vth

    dims = Symbol[]
    if hd.Nvx == 1
        vx_grid = grid1d(1, 0.0, 0.0)
    else
        vx_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vx)
    end
    if hd.Nvy == 1
        vy_grid = grid1d(1, 0.0, 0.0)
    else
        vy_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vy)
    end
    if hd.Nvz == 1
        vz_grid = grid1d(1, 0.0, 0.0)
    else
        vz_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vz)
    end

    return VGrid(dims, vx_grid, vy_grid, vz_grid, buffer)
end

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

approximate_f!(result, f, disc::XVDiscretization{<:Hermite}, dims) = begin
    vdims = sum(dims .>= 4)
    factor = 1 / (2π)^((3-vdims)/2)
    f_all(args...) = factor * f((args[dim] for dim in dims)...)
    hd = disc.vdisc
    (; Nvx, Nvy, Nvz, vth) = hd
    (; X, Y, Z) = disc.x_grid
    result .= Float64.(bigfloat_weighted_hermite_expansion(f_all, Nvx-1, Nvy-1, Nvz-1, X, Y, Z, vth))
end

expand_f(f, disc::XVDiscretization{WENO5}, vgrid) = begin
    return f
end

expand_f(coefs, disc::XVDiscretization{<:Hermite}, vgrid) = begin
    expand_bigfloat_hermite_f(coefs, vgrid, disc.vdisc.vth)
end
