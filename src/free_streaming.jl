function free_streaming!(df, f, species, buffer)
    @no_escape buffer begin
        df_fs = alloc(Float64, buffer, size(df)...)
        df_fs .= 0
        if :x ∈ species.x_dims
            df_x = alloc(Float64, buffer, size(species.grid)...)
            df_x .= 0
            @timeit "x" free_streaming_x!(df_x, f, species, buffer)
            df_fs .+= df_x
        end
        if :y ∈ species.x_dims
            df_y = alloc(Float64, buffer, size(species.grid)...)
            df_y .= 0
            free_streaming_y!(df_y, f, species, buffer)
            df_fs .+= df_y
        end
        if :z ∈ species.x_dims
            df_z = alloc(Float64, buffer, size(species.grid)...)
            df_z .= 0
            free_streaming_z!(df_z, f, species, buffer)
            df_fs .+= df_z
        end

        df .+= df_fs
        nothing
    end
end

function free_streaming_x!(df, f, species, buffer)
    (; grid) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)
    _, rest... = size(grid)
    dx = grid.x.x.dx

    @no_escape buffer begin
        f_with_boundaries = alloc(Float64, buffer, Nx+6, rest...) |> Origin(-2, 1, 1, 1, 1, 1)
        f_with_boundaries[1:Nx, :, :, :, :, :] .= f
        reflecting_wall_bcs!(f_with_boundaries, f, grid)

        F⁻ = alloc(Float64, buffer, Nx+6, Ny, Nz, Nvx÷2, Nvy, Nvz)
        F⁻ .= @view parent(f_with_boundaries)[:, :, :, 1:Nvx÷2, :, :]
        vx = Array(reshape(grid.VX[1:Nvx÷2], (1, 1, 1, :, 1, 1)))
        broadcast_mul_over_vx(F⁻, vx)

        F⁺ = alloc(Float64, buffer, Nx+6, Ny, Nz, Nvx÷2, Nvy, Nvz)
        F⁺ .= @view parent(f_with_boundaries)[:, :, :, Nvx÷2+1:Nvx, :, :]
        vx = reshape(grid.VX[Nvx÷2+1:Nvx], (1, 1, 1, :))
        broadcast_mul_over_vx(F⁺, vx)

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dx)
        left_biased_stencil =  [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dx)

        convolved = alloc(Float64, buffer, Nx, Ny, Nz, Nvx÷2, Nvy, Nvz)

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

function free_streaming_y!(df, f, species, buffer)
    (; grid) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)

    f = reshape(f, (Nx, Ny, Nz*Nvx, Nvy, Nvz))
    df = reshape(df, (Nx, Ny, Nz*Nvx, Nvy, Nvz))
    F = plan_rfft(f, (2,))

    @no_escape buffer begin
        Ky = (Ny ÷ 2 + 1)
        y_modes = alloc(Complex{Float64}, buffer, Nx, (Ny÷2+1), Nz*Nvx, Nvy, Nvz)
        y_modes .= 0
        mul!(y_modes, F, f)
        for λx in 1:Nx, ky in 1:Ky, λzvx in 1:(Nz*Nvx), λvy in 1:Nvy, λvz in 1:Nvz
            vy = grid.VY[λvy]
            y_modes[λx, ky, λzvx, λvy, λvz] *= -im * (ky-1) * vy * 2π / grid.x.y.L
        end
        mul!(df, inv(F), y_modes)
        nothing
    end
end

function free_streaming_z!(df, f, species, buffer)
    (; grid) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)

    f = reshape(f, (Nx*Ny, Nz, Nvx*Nvy, Nvz))
    df = reshape(df, (Nx*Ny, Nz, Nvx*Nvy, Nvz))
    F = plan_rfft(f, (2,))

    @no_escape buffer begin
        Kz = (Nz ÷ 2 + 1)
        z_modes = alloc(Complex{Float64}, buffer, Nx*Ny, (Nz÷2+1), Nvx*Nvy, Nvz)
        mul!(z_modes, F, f)
        for λxy in 1:(Nx*Ny), kz in 1:Kz, λvxvy in 1:(Nvx*Nvy), λvz in 1:Nvz
            vz = grid.VZ[λvz]
            z_modes[λxy, kz, λvxvy, λvz] *= -im * (kz-1) * vz * 2π / grid.x.z.L
        end
        mul!(df, inv(F), f)
    end
end

function free_streaming_x_boundaries!(df, f, species, buffer)
    (; grid) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)
    dx = grid.x.x.dx

    @no_escape buffer begin
        left_boundary = alloc(Float64, 3, Ny, Nz, Nvx, Nvy, Nvz) |> Origin(-2, 1, 1, 1, 1, 1)
        left_boundary .= 0
        right_boundary = alloc(Float64, 3, Ny, Nz, Nvx, Nvy, Nvz) |> Origin(Nx+1, 1, 1, 1, 1, 1)
        right_boundary .= 0
        reflecting_wall_bcs!(left_boundary, right_boundary, f, grid)

        # Negative vx
        for λ in CartesianIndices((Ny, Nz, 1:(Nvx÷2), Nvy, Nvz))
            df[1, λ] -= 1/dx * (
                + left_boundary[1-2, λ] / 20
                - left_boundary[1-1, λ] / 2
                - f[1, λ] / 3
                + f[1+1, λ]
                - f[1+2, λ] / 4 
                + f[1+3, λ] / 30)
            df[2, λ] -= 1/dx * (
                + left_boundary[2-2, λ] / 20
                - f[2-1, λ] / 2
                - f[2, λ] / 3
                + f[2+1, λ]
                - f[2+2, λ] / 4 
                + f[2+3, λ] / 30)


            df[Nx-2, λ] -= 1/dx * (
                + f[Nx-4, λ] / 20
                - f[Nx-3, λ] / 2
                - f[Nx-2, λ] / 3
                + f[Nx-1, λ]
                - f[Nx, λ] / 4 
                + right_boundary[Nx+1, λ] / 30)
            df[Nx-1, λ] -= 1/dx * (
                + f[Nx-3, λ] / 20
                - f[Nx-2, λ] / 2
                - f[Nx-1, λ] / 3
                + f[Nx, λ]
                - right_boundary[Nx+1, λ] / 4 
                + right_boundary[Nx+2, λ] / 30)
            df[Nx, λ] -= 1/dx * (
                + f[Nx-2, λ] / 20
                - f[Nx-1, λ] / 2
                - f[Nx, λ] / 3
                + right_boundary[Nx+1, λ]
                - right_boundary[Nx+2, λ] / 4 
                + right_boundary[Nx+3, λ] / 30)
        end
        # Positive vx
        for λ in CartesianIndices((Ny, Nz, (Nvx÷2+1):Nvx, Nvy, Nvz))
            df[1, λ] -= 1/dx * (
                - left_boundary[1-3, λ] / 30
                + left_boundary[1-2, λ] / 4
                - left_boundary[1-1, λ]
                + f[1, λ] / 3
                + f[1+1, λ] / 2 
                - f[1+2, λ] / 20)
            df[2, λ] -= 1/dx * (
                - left_boundary[2-3, λ] / 30
                + left_boundary[2-2, λ] / 4
                - f[2-1, λ]
                + f[2, λ] / 3
                + f[2+1, λ] / 2 
                - f[2+2, λ] / 20)
            df[3, λ] -= 1/dx * (
                - left_boundary[3-3, λ] / 30
                + f[3-2, λ] / 4
                - f[3-1, λ]
                + f[3, λ] / 3
                + f[3+1, λ] / 2 
                - f[3+2, λ] / 20)
            df[Nx-1, λ] -= 1/dx * (
                - f[Nx-4, λ] / 30
                + f[Nx-3, λ] / 4
                - f[Nx-2, λ]
                + f[Nx-1, λ] / 3
                + f[Nx, λ] / 2 
                - right_boundary[Nx+1, λ] / 20)
            df[Nx, λ] -= 1/dx * (
                - f[Nx-3, λ] / 30
                + f[Nx-2, λ] / 4
                - f[Nx-1, λ]
                + f[Nx, λ] / 3
                + right_boundary[Nx+1, λ] / 2 
                - right_boundary[Nx+2, λ] / 20)
        end
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

