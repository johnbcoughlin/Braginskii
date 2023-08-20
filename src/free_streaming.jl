function free_streaming!(df, f, species, buffer)
    @no_escape buffer begin
        df_fs = alloc_zeros(Float64, buffer, size(df)...)
        if :x ∈ species.x_dims
            df_x = alloc_zeros(Float64, buffer, size(species.discretization)...)
            @timeit "x" free_streaming_x!(df_x, f, species, buffer)
            df_fs .+= df_x
        end
        if :y ∈ species.x_dims
            df_y = alloc_zeros(Float64, buffer, size(species.discretization)...)
            free_streaming_y!(df_y, f, species, buffer)
            df_fs .+= df_y
        end
        if :z ∈ species.x_dims
            df_z = alloc_zeros(Float64, buffer, size(species.discretization)...)
            free_streaming_z!(df_z, f, species, buffer)
            df_fs .+= df_z
        end

        df .+= df_fs
        nothing
    end
end

function free_streaming_z!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    _, rest... = size(discretization)
    xgrid = discretization.x_grid
    dz = xgrid.z.dx
    vgrid = discretization.vdisc.grid

    @no_escape buffer begin
        f_with_boundaries = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx, Nvy, Nvz)
        f_with_boundaries[:, :, 4:Nz+3, :, :, :] .= f
        reflecting_wall_bcs!(f_with_boundaries, f, discretization)

        F⁻ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx, Nvy, Nvz÷2)
        F⁻ .= @view parent(f_with_boundaries)[:, :, :, :, :, 1:Nvz÷2]
        vz = Array(reshape(vgrid.VZ[1:Nvz÷2], (1, 1, 1, 1, 1, :)))
        F⁻ .*= vgrid.VZ[:, :, :, :, :, 1:Nvz÷2]

        F⁺ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx, Nvy, Nvz÷2)
        F⁺ .= @view parent(f_with_boundaries)[:, :, :, :, :, Nvz÷2+1:Nvz]
        vz = reshape(vgrid.VZ[Nvz÷2+1:Nvz], (1, 1, 1, 1, 1, :))
        F⁺ .*= vgrid.VZ[:, :, :, :, :, Nvz÷2+1:Nvz]

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dz)
        left_biased_stencil =  [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dz)

        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz÷2)

        convolve_z!(convolved, F⁻, right_biased_stencil, true, buffer)
        df[:, :, :, :, :, 1:Nvz÷2] .+= convolved
        convolve_z!(convolved, F⁺, left_biased_stencil, true, buffer)
        df[:, :, :, :, :, Nvz÷2+1:Nvz] .+= convolved

        nothing
    end

end

function broadcast_mul_over_vz(F, vz)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(F)
    F = reshape(F, (Nx*Ny*Nz*Nvx*Nvy, Nvz))
    for λxyzvxvy in axes(F, 1)
        for λvz in 1:Nvz
            F[λxyzvxvy, λvz] *= vz[λvz]
        end
    end
    F
end

function free_streaming_x!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    vgrid = discretization.vdisc.grid

    f = reshape(f, (Nx, Ny, :))
    df = reshape(df, (Nx, Ny, :))
    F = species.fft_plans.kxy_rfft

    @no_escape buffer begin
        Kx = (Nx ÷ 2 + 1)
        xy_modes = alloc_array(Complex{Float64}, buffer, Kx, Ny, Nz*Nvx*Nvy*Nvz)
        mul!(xy_modes, F, f)
        xy_modes = reshape(xy_modes, Kx, Ny, Nz, Nvx, Nvy*Nvz)
        kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1, Nvx)
        kxs .= (-im * 2π / xgrid.x.L) * ((0:Kx-1) .* vgrid.VX)
        xy_modes .*= kxs
        xy_modes = reshape(xy_modes, (Kx, Ny, :))
        mul!(df, species.fft_plans.kxy_irfft, xy_modes)
        nothing
    end
end

function free_streaming_y!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    vgrid = discretization.vdisc.grid

    f = reshape(f, (Nx, Ny, Nz, Nvx, Nvy, Nvz))
    df = reshape(df, (Nx, Ny, Nz, Nvx, Nvy, Nvz))
    F = plan_rfft(f, [1, 2])

    @no_escape buffer begin
        Kx = (Nx ÷ 2 + 1)
        Ky = Ny
        kys = alloc_array(Complex{Float64}, buffer, 1, Ky, 1, 1, Nvy)
        kys .= (-im * 2π / xgrid.y.L) * arraytype(buffer)(mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))') .* vgrid.VY
        xy_modes = alloc_zeros(Complex{Float64}, buffer, Kx, Ny, Nz, Nvx, Nvy, Nvz)
        mul!(xy_modes, F, f)
        xy_modes .*= kys
        #for λx in 1:Kx, ky in 1:Ky, λz in 1:Nz, λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            #vy = vgrid.VY[λvy]
            #xy_modes[λx, ky, λz, λvx, λvy, λvz] *= -im * kys[ky] * vy * 2π / xgrid.y.L
        #end
        mul!(df, inv(F), xy_modes)
        nothing
    end
end

function reflecting_wall_bcs!(f_with_boundaries, f, grid)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)

    # Left boundary
    f_with_boundaries[:, :, 1, :, :, 1:Nvz] .= f[:, :, 3, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, 2, :, :, 1:Nvz] .= f[:, :, 2, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, 3 , :, :, 1:Nvz] .= f[:, :, 1, :, :, Nvz:-1:1]

    # Right boundary
    f_with_boundaries[:, :, end, :, :, 1:Nvz] .= f[:, :, Nz-2, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, end-1, :, :, 1:Nvz] .= f[:, :, Nz-1, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, end-2, :, :, 1:Nvz] .= f[:, :, Nz, :, :, Nvz:-1:1]
end

