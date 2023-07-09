function electrostatic!(df, f, Ex, Ey, Bz, species, buffer)
    @no_escape buffer begin
        df_es = alloc_zeros(Float64, buffer, size(df)...)

        if :vx ∈ species.v_dims
            electrostatic_x!(df_es, f, Ex, Bz, species, buffer)
        end
        if :vy ∈ species.v_dims
            electrostatic_y!(df_es, f, Ey, Bz, species, buffer)
        end

        df .+= df_es
        nothing
    end
end

function electrostatic_x!(df, f, Ex, Bz, species::Species{WENO5}, buffer)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid
    dvx = vgrid.x.dx

    @no_escape buffer begin
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvy)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvy)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvy)
        for λxyz in CartesianIndices((Nx, Ny, Nz))
            for λvy in 1:Nvy
                vy = vgrid.VY[λvy]
                C[λxyz, λvy] = q / m * (Ex[λxyz] + vy * Bz[λxyz])
            end
        end

        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        F⁺ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        for λxyz in CartesianIndices((Nx, 2Ny-1, 2Nz-1))
            for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
                F⁺[λxyz, λvx, λvy, λvz] = max(C[λxyz, λvy], 0) * f̂[λxyz, λvx, λvy, λvz]
                F⁻[λxyz, λvx, λvy, λvz] = min(C[λxyz, λvy], 0) * f̂[λxyz, λvx, λvy, λvz]
            end
        end

        F⁺ = reverse_quadratic_dealias(F⁺, buffer)
        F⁻ = reverse_quadratic_dealias(F⁻, buffer)

        # Negative
        for λxyz in CartesianIndices((Nx, Ny, Nz))
            for λvx in 3:Nvx-3, λvyvz in CartesianIndices((Nvy, Nvz))
                fm = (
                    + F⁻[λxyz, λvx-2, λvyvz] / 20
                    - F⁻[λxyz, λvx-1, λvyvz] / 2
                    - F⁻[λxyz, λvx, λvyvz] / 3
                    + F⁻[λxyz, λvx+1, λvyvz]
                    - F⁻[λxyz, λvx+2, λvyvz] / 4
                    + F⁻[λxyz, λvx+3, λvyvz] / 30)
                df[λxyz, λvx, λvyvz] -= 1 / dvx * fm
            end
        end
        # Positive
        for λxyz in CartesianIndices((Nx, Ny, Nz))
            for λvx in 4:Nvx-2, λvyvz in CartesianIndices((Nvy, Nvz))
                fm = (
                    - F⁺[λxyz, λvx-3, λvyvz] / 30
                    + F⁺[λxyz, λvx-2, λvyvz] / 4
                    - F⁺[λxyz, λvx-1, λvyvz]
                    + F⁺[λxyz, λvx, λvyvz] / 3
                    + F⁺[λxyz, λvx+1, λvyvz] / 2
                    - F⁺[λxyz, λvx+2, λvyvz] / 20)
                df[λxyz, λvx, λvyvz] -= 1 / dvx * fm
            end
        end

        electrostatic_x_boundary!(df, F⁺, F⁻, discretization)

    end
end

function electrostatic_y!(df, f, Ey, Bz, species::Species{WENO5}, buffer)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid

    dvy = vgrid.y.dx

    @no_escape buffer begin
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)
        for λxyz in CartesianIndices((Nx, Ny, Nz))
            for λvx in 1:Nvx
                vx = vgrid.VX[λvx]
                C[λxyz, λvx] = q / m * (Ey[λxyz] - vx * Bz[λxyz])
            end
        end

        begin
        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        end
        F⁺ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        for λxyz in CartesianIndices((Nx, 2Ny-1, 2Nz-1))
            for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
                F⁺[λxyz, λvx, λvy, λvz] = max(C[λxyz, λvx], 0) * f̂[λxyz, λvx, λvy, λvz]
                F⁻[λxyz, λvx, λvy, λvz] = min(C[λxyz, λvx], 0) * f̂[λxyz, λvx, λvy, λvz]
            end
        end

        begin
        F⁺ = reverse_quadratic_dealias(F⁺, buffer)
        F⁻ = reverse_quadratic_dealias(F⁻, buffer)
        end

        # Negative
        for λxyz in CartesianIndices((Nx, Ny, Nz, Nvx))
            for λvy in 3:Nvy-3, λvz in 1:Nvz
                fm = (
                    + F⁻[λxyz, λvy-2, λvz] / 20
                    - F⁻[λxyz, λvy-1, λvz] / 2
                    - F⁻[λxyz, λvy, λvz] / 3
                    + F⁻[λxyz, λvy+1, λvz]
                    - F⁻[λxyz, λvy+2, λvz] / 4
                    + F⁻[λxyz, λvy+3, λvz] / 30)
                df[λxyz, λvy, λvz] -= 1 / dvy * fm
            end
        end
        # Positive
        for λxyz in CartesianIndices((Nx, Ny, Nz, Nvx))
            for λvy in 4:Nvy-2, λvz in 1:Nvz
                fm = (
                    - F⁺[λxyz, λvy-3, λvz] / 30
                    + F⁺[λxyz, λvy-2, λvz] / 4
                    - F⁺[λxyz, λvy-1, λvz]
                    + F⁺[λxyz, λvy, λvz] / 3
                    + F⁺[λxyz, λvy+1, λvz] / 2
                    - F⁺[λxyz, λvy+2, λvz] / 20)
                df[λxyz, λvy, λvz] -= 1 / dvy * fm
            end
        end

        electrostatic_y_boundary!(df, F⁺, F⁻, discretization)
    end
end

# Do the round trip from physical space to Fourier space back to physical space on twice
# as many grid points
function quadratic_dealias(u, buffer)
    Nx, Ny, Nz, Nvs... = size(u)
    u = reshape(u, (Nx, Ny, Nz, :))

    u_modes = alloc_zeros(Complex{Float64}, buffer, Nx, Ny, Nz, prod(Nvs))
    @timeit "plan fft" F = plan_rfft(u, (2, 3))
    u_modes_tmp = alloc_array(Complex{Float64}, buffer, Nx, Ny÷2+1, Nz÷2+1, prod(Nvs))
    mul!(u_modes_tmp, F, u)

    #u_modes[:, 1:(Ny÷2+1), 1:(Nz÷2+1), :] .= u_modes_tmp
    copy_to_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, Nx, 2Ny-1, 2Nz-1, prod(Nvs))
    @timeit "plan fft" F⁻¹ = plan_irfft(u_modes, 2Ny-1, (2, 3))
    mul!(U, F⁻¹, u_modes)

    return reshape(U, (Nx, 2Ny-1, 2Nz-1, Nvs...)) * (2Ny-1)/Ny * (2Nz-1)/Nz
end

function copy_to_first_half!(u_modes, u_modes_tmp)
    _, Ny, Nz, _ = size(u_modes)
    for λx in axes(u_modes, 1)
        for λy in 1:(Ny÷2+1), λz in 1:(Nz÷2+1), λ in axes(u_modes, 4)
            u_modes[λx, λy, λz, λ] = u_modes_tmp[λx, λy, λz, λ]
        end
    end
end

function reverse_quadratic_dealias(u, buffer)
    Nx, Nŷ, Nẑ, Nvs... = size(u)
    Ny = Nŷ÷2+1
    Nz = Nẑ÷2+1

    u = reshape(u, (Nx, Nŷ, Nẑ, :))

    u_modes_tmp = alloc_zeros(Complex{Float64}, buffer, Nx, Ny, Nz, prod(Nvs))
    @timeit "plan fft" F = plan_rfft(u, (2, 3))
    mul!(u_modes_tmp, F, u)

    u_modes = alloc_array(Complex{Float64}, buffer, Nx, Ny÷2+1, Nz÷2+1, prod(Nvs))
    copy_from_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, Nx, Ny, Nz, prod(Nvs))
    @timeit "plan fft" F⁻¹ = plan_irfft(u_modes, Ny, (2, 3))
    mul!(U, F⁻¹, u_modes)

    return reshape(U, (Nx, Ny, Nz, Nvs...)) / ( (2Ny-1)/Ny * (2Nz-1)/Nz )
end
function copy_from_first_half!(u_modes, u_modes_tmp)
    _, Ny, Nz, _ = size(u_modes_tmp)
    for λx in axes(u_modes, 1)
        for λy in 1:(Ny÷2+1), λz in 1:(Nz÷2+1), λ in axes(u_modes, 4)
            u_modes[λx, λy, λz, λ] = u_modes_tmp[λx, λy, λz, λ]
        end
    end
end


function electrostatic_x_boundary!(df, F⁺, F⁻, discretization::XVDiscretization{WENO5})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid
    dvx = vgrid.x.dx

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvyvz in CartesianIndices((Nvy, Nvz))
            λvx = 1
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                - F⁻[λxyz, λvx, λvyvz] / 3
                + F⁻[λxyz, λvx+1, λvyvz]
                - F⁻[λxyz, λvx+2, λvyvz] / 4
                + F⁻[λxyz, λvx+3, λvyvz] / 30)
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                + F⁺[λxyz, λvx, λvyvz] / 3
                + F⁺[λxyz, λvx+1, λvyvz] / 2
                - F⁺[λxyz, λvx+2, λvyvz] / 20)

            λvx = 2
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                - F⁻[λxyz, λvx-1, λvyvz] / 2
                - F⁻[λxyz, λvx, λvyvz] / 3
                + F⁻[λxyz, λvx+1, λvyvz]
                - F⁻[λxyz, λvx+2, λvyvz] / 4
                + F⁻[λxyz, λvx+3, λvyvz] / 30)
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                - F⁺[λxyz, λvx-1, λvyvz]
                + F⁺[λxyz, λvx, λvyvz] / 3
                + F⁺[λxyz, λvx+1, λvyvz] / 2
                - F⁺[λxyz, λvx+2, λvyvz] / 20)

            λvx = 3
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                + F⁺[λxyz, λvx-2, λvyvz] / 4
                - F⁺[λxyz, λvx-1, λvyvz]
                + F⁺[λxyz, λvx, λvyvz] / 3
                + F⁺[λxyz, λvx+1, λvyvz] / 2
                - F⁺[λxyz, λvx+2, λvyvz] / 20)

            λvx = Nvx
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                + F⁻[λxyz, λvx-2, λvyvz] / 20
                - F⁻[λxyz, λvx-1, λvyvz] / 2
                - F⁻[λxyz, λvx, λvyvz] / 3)
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                - F⁺[λxyz, λvx-3, λvyvz] / 30
                + F⁺[λxyz, λvx-2, λvyvz] / 4
                - F⁺[λxyz, λvx-1, λvyvz]
                + F⁺[λxyz, λvx, λvyvz] / 3)

            λvx = Nvx-1
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                + F⁻[λxyz, λvx-2, λvyvz] / 20
                - F⁻[λxyz, λvx-1, λvyvz] / 2
                - F⁻[λxyz, λvx, λvyvz] / 3
                + F⁻[λxyz, λvx+1, λvyvz])
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                - F⁺[λxyz, λvx-3, λvyvz] / 30
                + F⁺[λxyz, λvx-2, λvyvz] / 4
                - F⁺[λxyz, λvx-1, λvyvz]
                + F⁺[λxyz, λvx, λvyvz] / 3
                + F⁺[λxyz, λvx+1, λvyvz] / 2)

            λvx = Nvx-2
            df[λxyz, λvx, λvyvz] -= 1 / dvx * (
                + F⁻[λxyz, λvx-2, λvyvz] / 20
                - F⁻[λxyz, λvx-1, λvyvz] / 2
                - F⁻[λxyz, λvx, λvyvz] / 3
                + F⁻[λxyz, λvx+1, λvyvz]
                - F⁻[λxyz, λvx+2, λvyvz] / 4)
        end
    end
end

function electrostatic_y_boundary!(df, F⁺, F⁻, discretization::XVDiscretization{WENO5})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid
    dvy = vgrid.y.dx

    for λxyz in CartesianIndices((Nx, Ny, Nz, Nvx))
        for λvz in 1:Nvz
            λvy = 1
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                - F⁻[λxyz, λvy, λvz] / 3
                + F⁻[λxyz, λvy+1, λvz]
                - F⁻[λxyz, λvy+2, λvz] / 4
                + F⁻[λxyz, λvy+3, λvz] / 30)
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                + F⁺[λxyz, λvy, λvz] / 3
                + F⁺[λxyz, λvy+1, λvz] / 2
                - F⁺[λxyz, λvy+2, λvz] / 20)

            λvy = 2
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                - F⁻[λxyz, λvy-1, λvz] / 2
                - F⁻[λxyz, λvy, λvz] / 3
                + F⁻[λxyz, λvy+1, λvz]
                - F⁻[λxyz, λvy+2, λvz] / 4
                + F⁻[λxyz, λvy+3, λvz] / 30)
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                - F⁺[λxyz, λvy-1, λvz]
                + F⁺[λxyz, λvy, λvz] / 3
                + F⁺[λxyz, λvy+1, λvz] / 2
                - F⁺[λxyz, λvy+2, λvz] / 20)

            λvy = 3
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                + F⁺[λxyz, λvy-2, λvz] / 4
                - F⁺[λxyz, λvy-1, λvz]
                + F⁺[λxyz, λvy, λvz] / 3
                + F⁺[λxyz, λvy+1, λvz] / 2
                - F⁺[λxyz, λvy+2, λvz] / 20)

            λvy = Nvy
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                + F⁻[λxyz, λvy-2, λvz] / 20
                - F⁻[λxyz, λvy-1, λvz] / 2
                - F⁻[λxyz, λvy, λvz] / 3)
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                - F⁺[λxyz, λvy-3, λvz] / 30
                + F⁺[λxyz, λvy-2, λvz] / 4
                - F⁺[λxyz, λvy-1, λvz]
                + F⁺[λxyz, λvy, λvz] / 3)

            λvy = Nvy-1
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                + F⁻[λxyz, λvy-2, λvz] / 20
                - F⁻[λxyz, λvy-1, λvz] / 2
                - F⁻[λxyz, λvy, λvz] / 3
                + F⁻[λxyz, λvy+1, λvz])
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                - F⁺[λxyz, λvy-3, λvz] / 30
                + F⁺[λxyz, λvy-2, λvz] / 4
                - F⁺[λxyz, λvy-1, λvz]
                + F⁺[λxyz, λvy, λvz] / 3
                + F⁺[λxyz, λvy+1, λvz] / 2)

            λvy = Nvy-2
            df[λxyz, λvy, λvz] -= 1 / dvy * (
                + F⁻[λxyz, λvy-2, λvz] / 20
                - F⁻[λxyz, λvy-1, λvz] / 2
                - F⁻[λxyz, λvy, λvz] / 3
                + F⁻[λxyz, λvy+1, λvz]
                - F⁻[λxyz, λvy+2, λvz] / 4)
        end
    end
end
