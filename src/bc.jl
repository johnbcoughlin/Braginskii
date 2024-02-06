struct ReflectingWallBCs end

struct ReservoirBC{A}
    f_with_bcs::A
end

function make_bcs(x_grid, vdisc, f0, buffer, type)
    if type == :reservoir
        return make_reservoir_bcs(x_grid, vdisc, f0, buffer)
    elseif type == :reflecting
        return ReflectingWallBCs()
    end
end

function make_reservoir_bcs(x_grid, vdisc, left_f, right_f, buffer)
    Nx, Ny, Nz = size(x_grid)
    Nvx, Nvy, Nvz = size(vdisc)

    f_with_bcs = alloc_zeros(Float64, buffer, Nx, Ny, Nz+6, Nvx, Nvy, Nvz)
    f_with_bcs[:, :, 1:3, :, :, :] .= left_f
    f_with_bcs[:, :, end-2:end, :, :, :] .= right_f

    return ReservoirBC(f_with_bcs)
end

function make_reservoir_bcs(x_grid, vdisc, f0::Function, buffer)
    Nx, Ny, Nz = size(x_grid)
    Nvx, Nvy, Nvz = size(vdisc)

    if Nz == 1
        return nothing
    end
    (; X, Y) = x_grid
    zgrid = x_grid.z
    Nz = zgrid.N
    dz = zgrid.dx

    dims = findall(>(1), [Nx, Ny, Nz, Nvx, Nvy, Nvz])
    dims = tuple(dims...)
    @show dims

    # Construct a new pair of z grids

    left_z_grid = grid1d(3, zgrid.min - 3dz, zgrid.min)
    left_grid = XGrid(x_grid.x, x_grid.y, left_z_grid, buffer)
    left_f = alloc_zeros(Float64, buffer, Nx, Ny, 3, Nvx, Nvy, Nvz)
    approximate_f!(left_f, f0, left_grid, vdisc, dims)

    right_z_grid = grid1d(3, zgrid.max, zgrid.max + 3dz)
    right_grid = XGrid(x_grid.x, x_grid.y, right_z_grid, buffer)
    right_f = alloc_zeros(Float64, buffer, Nx, Ny, 3, Nvx, Nvy, Nvz)
    approximate_f!(right_f, f0, right_grid, vdisc, dims)

    return make_reservoir_bcs(x_grid, vdisc, left_f, right_f, buffer)
end

function make_reservoir_bcs(x_grid, vdisc::HermiteLaguerre, f0, buffer)
    Nx, Ny, Nz = size(x_grid)
    Nμ, Nvy = size(vdisc)

    if Nz == 1
        return nothing
    end
    (; X, Y) = x_grid
    zgrid = x_grid.z
    Nz = zgrid.N
    dz = zgrid.dx

    dims = findall(>(1), [Nx, Ny, Nz, Nμ, Nvy])
    dims = tuple(dims...)

    # Construct a new pair of z grids

    left_z_grid = grid1d(3, zgrid.min - 3dz, zgrid.min)
    left_grid = XGrid(x_grid.x, x_grid.y, left_z_grid, buffer)
    left_f = alloc_zeros(Float64, buffer, Nx, Ny, 3, Nμ, Nvy)
    approximate_f!(left_f, f0, left_grid, vdisc, dims)

    right_z_grid = grid1d(3, zgrid.max, zgrid.max + 3dz)
    right_grid = XGrid(x_grid.x, x_grid.y, right_z_grid, buffer)
    right_f = alloc_zeros(Float64, buffer, Nx, Ny, 3, Nμ, Nvy)
    approximate_f!(right_f, f0, right_grid, vdisc, dims)

    f_with_bcs = alloc_zeros(Float64, buffer, Nx, Ny, Nz+6, Nμ, Nvy)
    f_with_bcs[:, :, 1:3, :, :] .= left_f
    f_with_bcs[:, :, end-2:end, :, :] .= right_f

    return ReservoirBC(f_with_bcs)
end
