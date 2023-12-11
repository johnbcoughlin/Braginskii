function ampere!(dE, f, sim, buffer)
    Nx, Ny, Nz = size(sim.xgrid)

    Jx = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    Jy = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    Jz = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    for (i, α) in enumerate(sim.species)
        fα = f.x[i]
        q = α.q
        _, M1, _ = moments(fα, α.discretization, α.v_dims, buffer)
        M1x, M1y, M1z = M1

        if :vx ∈ v_dims
            @. Jx = M1x * q
            Jx_avg = sum(Jx, dims=(1, 2, 3)) / (Nx*Ny*Nz)
            @. Jx -= Jx_avg
        end
        if :vy ∈ v_dims
            @. Jy += M1y * q
        end
        if :vz ∈ v_dims
            @. Jz += M1z * q
        end
    end

    Jx_avg = sum(Jx, dims=(1, 2, 3)) / (Nx*Ny*Nz)
    @. Jx -= Jx_avg
    Jy_avg = sum(Jy, dims=(1, 2, 3)) / (Nx*Ny*Nz)
    @. Jy -= Jy_avg

    dEx, dEy, dEz = dE
    @. dEx -= Jx
    @. dEy -= Jy
    @. dEz -= Jz
end


