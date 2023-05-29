# Moments are computed using a "periodic" Trapezoid rule

function moments(f)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc(Float64, buffer, Nx, Ny, Nz)
    M1x = alloc(Float64, buffer, Nx, Ny, Nz)
    M1y = alloc(Float64, buffer, Nx, Ny, Nz)
    M1z = alloc(Float64, buffer, Nx, Ny, Nz)
    M2 = alloc(Float64, buffer, Nx, Ny, Nz)

    dv = grid.v.x.dx * grid.v.y.dx * grid.v.z.dx

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
end

function density(f, grid, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc(Float64, buffer, Nx, Ny, Nz)

    dv = grid.v.x.dx * grid.v.y.dx * grid.v.z.dx

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            M0[λxyz] += f[λxyz, λvx, λvy, λvz] * dv
        end
    end

    return M0
end

