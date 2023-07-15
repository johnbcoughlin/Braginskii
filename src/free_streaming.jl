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

function free_streaming_x!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    _, rest... = size(discretization)
    xgrid = discretization.x_grid
    dx = xgrid.x.dx
    vgrid = discretization.vdisc.grid

    @no_escape buffer begin
        f_with_boundaries = alloc_array(Float64, buffer, Nx+6, rest...) |> Origin(-2, 1, 1, 1, 1, 1)
        f_with_boundaries[1:Nx, :, :, :, :, :] .= f
        reflecting_wall_bcs!(f_with_boundaries, f, discretization)

        F⁻ = alloc_array(Float64, buffer, Nx+6, Ny, Nz, Nvx÷2, Nvy, Nvz)
        F⁻ .= @view parent(f_with_boundaries)[:, :, :, 1:Nvx÷2, :, :]
        vx = Array(reshape(vgrid.VX[1:Nvx÷2], (1, 1, 1, :, 1, 1)))
        broadcast_mul_over_vx(F⁻, vx)

        F⁺ = alloc_array(Float64, buffer, Nx+6, Ny, Nz, Nvx÷2, Nvy, Nvz)
        F⁺ .= @view parent(f_with_boundaries)[:, :, :, Nvx÷2+1:Nvx, :, :]
        vx = reshape(vgrid.VX[Nvx÷2+1:Nvx], (1, 1, 1, :))
        broadcast_mul_over_vx(F⁺, vx)

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dx)
        left_biased_stencil =  [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dx)

        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx÷2, Nvy, Nvz)

        convolve_x!(convolved, F⁻, right_biased_stencil, true, buffer)
        df[:, :, :, 1:Nvx÷2, :, :] .+= convolved
        convolve_x!(convolved, F⁺, left_biased_stencil, true, buffer)
        df[:, :, :, Nvx÷2+1:Nvx, :, :] .+= convolved

        nothing
    end

end

function broadcast_mul_over_vx(F, vx)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(F)
    F = reshape(F, (Nx*Ny*Nz, Nvx, Nvy*Nvz))
    for λxyz in axes(F, 1)
        for λvx in 1:Nvx, λvyvz in axes(F, 3)
            F[λxyz, λvx, λvyvz] *= vx[λvx]
        end
    end
    F
end

function free_streaming_y!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    vgrid = discretization.vdisc.grid

    f = reshape(f, (Nx, Ny, Nz*Nvx, Nvy, Nvz))
    df = reshape(df, (Nx, Ny, Nz*Nvx, Nvy, Nvz))
    F = plan_rfft(f, (2,))

    @no_escape buffer begin
        Ky = (Ny ÷ 2 + 1)
        y_modes = alloc_zeros(Complex{Float64}, buffer, Nx, (Ny÷2+1), Nz*Nvx, Nvy, Nvz)
        mul!(y_modes, F, f)
        for λx in 1:Nx, ky in 1:Ky, λzvx in 1:(Nz*Nvx), λvy in 1:Nvy, λvz in 1:Nvz
            vy = vgrid.VY[λvy]
            y_modes[λx, ky, λzvx, λvy, λvz] *= -im * (ky-1) * vy * 2π / xgrid.y.L
        end
        mul!(df, inv(F), y_modes)
        nothing
    end
end

function free_streaming_z!(df, f, species::Species{WENO5}, buffer)
    (; grid) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)

    f = reshape(f, (Nx*Ny, Nz, Nvx*Nvy, Nvz))
    df = reshape(df, (Nx*Ny, Nz, Nvx*Nvy, Nvz))
    F = plan_rfft(f, (2,))

    @no_escape buffer begin
        Kz = (Nz ÷ 2 + 1)
        z_modes = alloc_array(Complex{Float64}, buffer, Nx*Ny, (Nz÷2+1), Nvx*Nvy, Nvz)
        mul!(z_modes, F, f)
        for λxy in 1:(Nx*Ny), kz in 1:Kz, λvxvy in 1:(Nvx*Nvy), λvz in 1:Nvz
            vz = grid.VZ[λvz]
            z_modes[λxy, kz, λvxvy, λvz] *= -im * (kz-1) * vz * 2π / grid.x.z.L
        end
        mul!(df, inv(F), f)
    end
end

function reflecting_wall_bcs!(f_with_boundaries, f, grid)
    Nx, _, _, Nvx, _, _ = size(grid)

    # Left boundary
    f_with_boundaries[-2, :, :, 1:Nvx, :, :] .= f[3, :, :, Nvx:-1:1, :, :]
    f_with_boundaries[-1, :, :, 1:Nvx, :, :] .= f[2, :, :, Nvx:-1:1, :, :]
    f_with_boundaries[0 , :, :, 1:Nvx, :, :] .= f[1, :, :, Nvx:-1:1, :, :]

    # Right boundary
    f_with_boundaries[Nx+3, :, :, 1:Nvx, :, :] .= f[Nx-2, :, :, Nvx:-1:1, :, :]
    f_with_boundaries[Nx+2, :, :, 1:Nvx, :, :] .= f[Nx-1, :, :, Nvx:-1:1, :, :]
    f_with_boundaries[Nx+1, :, :, 1:Nvx, :, :] .= f[Nx  , :, :, Nvx:-1:1, :, :]
end

