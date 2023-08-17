# Moments are computed using a "periodic" Trapezoid rule

function moments(f, disc::XVDiscretization{WENO5}, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1y = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M2 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    grid = disc.vdisc.grid

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.z.dx
    end

    fvx = f .* grid.VX
    fvy = f .* grid.VY
    fvz = f .* grid.VZ
    fv2 = @. f * (grid.VX^2 + grid.VY^2 + grid.VZ^2)
    sum!(M0, f)
    sum!(M1x, fvx)
    sum!(M1y, fvy)
    sum!(M1z, fvz)
    sum!(M2, fv2)

    M0 .*= dv
    M1x .*= dv
    M1y .*= dv
    M1z .*= dv
    M2 .*= dv

    M0, (M1x, M1y, M1z), M2
end

function density(f, disc::XVDiscretization{WENO5}, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    grid = disc.vdisc.grid

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.z.dx
    end

    sum!(M0, f)
    M0 .*= dv

    return M0
end

function collisional_moments!(sim, f, buffer)
    if length(sim.species) > 1
        error("Not implemented")
    end

    α = sim.species[1]

    f = f.x[1]

    (ux, uy, uz), T, ν = collisional_moments_single_species(α, f, sim.x_grid, sim.ν_p, buffer)

    cm = sim.collisional_moments[(α.name, α.name)]
    cm.ux .= ux
    cm.uy .= uy
    cm.uz .= uz
    cm.T .= T
    cm.ν .= ν
end

function collisional_moments_single_species(α, f, x_grid, ν_p, buffer)
    M0, M1, M2 = moments(f, α.discretization, α.v_dims, buffer)
    M1x, M1y, M1z = M1

    ux = M1x ./ M0
    uy = M1y ./ M0
    uz = M1z ./ M0
    d = length(α.v_dims)

    T = (M2 .- (ux.^2 + uy.^2 + uz.^2)) ./ (d .* M0)
    ν = alloc_zeros(Float64, buffer, size(x_grid)...)
    ν .= ν_p

    #@show sum(M1x) * x_grid.x.dx

    return (ux, uy, uz), T, ν
end
