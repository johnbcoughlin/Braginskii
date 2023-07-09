# Moments are computed using a "periodic" Trapezoid rule

function moments(f, grid, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1y = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M2 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.v.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.v.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.v.z.dx
    end

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            vx = grid.VX[λvx]
            vy = grid.VY[λvy]
            vz = grid.VZ[λvz]
            v² = vx^2 + vy^2 + vz^2
            M0[λxyz] += f[λxyz, λvx, λvy, λvz] * dv
            M1x[λxyz] += f[λxyz, λvx, λvy, λvz] * dv * vx
            M1y[λxyz] += f[λxyz, λvx, λvy, λvz] * dv * vy
            M1z[λxyz] += f[λxyz, λvx, λvy, λvz] * dv * vz
            M2[λxyz] += f[λxyz, λvx, λvy, λvz] * dv * v²
        end
    end

    M0, (M1x, M1y, M1z), M2
end

function density(f, grid, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.v.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.v.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.v.z.dx
    end

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            M0[λxyz] += f[λxyz, λvx, λvy, λvz] * dv
        end
    end

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
    M0, M1, M2 = moments(f, α.grid, α.v_dims, buffer)
    M1x, M1y, M1z = M1

    ux = M1x ./ M0
    uy = M1y ./ M0
    uz = M1z ./ M0
    d = length(α.v_dims)

    T = (M2 .- (ux.^2 + uy.^2 + uz.^2)) ./ (d .* M0)
    ν = zeros(size(x_grid))
    ν .= ν_p

    #@show sum(M1x) * x_grid.x.dx

    return (ux, uy, uz), T, ν
end
