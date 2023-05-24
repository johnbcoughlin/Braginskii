function free_streaming!(df, f, species, buffer)
    df .= 0
    @no_escape buffer begin
        if :x ∈ species.x_dims
            df_x = alloc(Float64, buffer, size(species.grid)...)
            df_x .= 0
            free_streaming_x!(df_x, f, species, buffer)
            df .+= df_x
        end
        if :y ∈ species.x_dims
            df_y = alloc(Float64, buffer, size(species.grid)...)
            df_y .= 0
            free_streaming_y!(df_y, f, species, buffer)
            df .+= df_y
        end
        if :z ∈ species.x_dims
            df_z = alloc(Float64, buffer, size(species.grid)...)
            df_z .= 0
            free_streaming_z!(df_z, f, species, buffer)
            df .+= df_z
        end
    end
end

function free_streaming_x!(df, f, species, buffer)
    (; grid) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)
    dx = grid.x.x.dx

    # Negative vx
    for λx in 3:Nx-3, λyz in CartesianIndices((Ny, Nz))
        for λvx in 1:(Nvx ÷ 2), λvyvz in CartesianIndices((Nvy, Nvz))
            vx = grid.VX[λvx]
            fm = (
                + f[λx-2, λyz, λvx, λvyvz] / 20
                - f[λx-1, λyz, λvx, λvyvz] / 2
                - f[λx, λyz, λvx, λvyvz] / 3
                + f[λx+1, λyz, λvx, λvyvz]
                - f[λx+2, λyz, λvx, λvyvz] / 4 
                + f[λx+3, λyz, λvx, λvyvz] / 30)
            df[λx, λyz, λvx, λvyvz] -= 1/dx * fm
        end
    end

    # Positive vx
    for λx in 4:Nx-2, λyz in CartesianIndices((Ny, Nz))
        for λvx in (Nvx ÷ 2 + 1):Nvx, λvyvz in CartesianIndices((Nvy, Nvz))
            vx = grid.VX[λvx]
            fm = (
                - f[λx-3, λyz, λvx, λvyvz] / 30
                + f[λx-2, λyz, λvx, λvyvz] / 4
                - f[λx-1, λyz, λvx, λvyvz]
                + f[λx, λyz, λvx, λvyvz] / 3
                + f[λx+1, λyz, λvx, λvyvz] / 2 
                - f[λx+2, λyz, λvx, λvyvz] / 20)
            df[λx, λyz, λvx, λvyvz] -= 1/dx * fm
        end
    end

    free_streaming_x_boundaries!(df, f, species, buffer)

    # Multiply by VX
    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvyvz in CartesianIndices((Nvy, Nvz))
            df[λxyz, λvx, λvyvz] *= grid.VX[λvx]
        end
    end
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
            y_modes[λx, ky, λzvx, λvy, λvz] *= -im * (ky-1) * vy
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
            z_modes[λxy, kz, λvxvy, λvz] *= im * kz * vz
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

function reflecting_wall_bcs!(left_boundary, right_boundary, f, grid)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(grid)
    for λyz in CartesianIndices((Ny, Nz))
        for λvyvz in CartesianIndices((Nvy, Nvz))
            for λvx in 1:Nvx
                vx = grid.VX[λvx]

                if vx > 0
                    # Inflow
                    left_boundary[-2, λyz, λvx, λvyvz] = f[3, λyz, Nvx-λvx+1, λvyvz]
                    left_boundary[-1, λyz, λvx, λvyvz] = f[2, λyz, Nvx-λvx+1, λvyvz]
                    left_boundary[0, λyz, λvx, λvyvz] = f[1, λyz, Nvx-λvx+1, λvyvz]
                    # Outflow, copy out
                    right_boundary[Nx+1, λyz, λvx, λvyvz] = f[Nx, λyz, Nvx-λvx+1, λvyvz]
                    right_boundary[Nx+2, λyz, λvx, λvyvz] = f[Nx-1, λyz, Nvx-λvx+1, λvyvz]
                    right_boundary[Nx+3, λyz, λvx, λvyvz] = f[Nx-2, λyz, Nvx-λvx+1, λvyvz]
                else
                    # Outflow, copy out
                    left_boundary[-2, λyz, λvx, λvyvz] = f[3, λyz, Nvx-λvx+1, λvyvz]
                    left_boundary[-1, λyz, λvx, λvyvz] = f[2, λyz, Nvx-λvx+1, λvyvz]
                    left_boundary[0, λyz, λvx, λvyvz] = f[1, λyz, Nvx-λvx+1, λvyvz]
                    # Inflow
                    right_boundary[Nx+1, λyz, λvx, λvyvz] = f[Nx, λyz, Nvx-λvx+1, λvyvz]
                    right_boundary[Nx+2, λyz, λvx, λvyvz] = f[Nx-1, λyz, Nvx-λvx+1, λvyvz]
                    right_boundary[Nx+3, λyz, λvx, λvyvz] = f[Nx-2, λyz, Nvx-λvx+1, λvyvz]
                end
            end
        end
    end
end

