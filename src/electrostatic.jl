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

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dvx)
        left_biased_stencil = [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dvx)

        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        convolve_vx!(convolved, F⁻, right_biased_stencil, false, buffer)
        df .+= convolved
        convolve_vx!(convolved, F⁺, left_biased_stencil, false, buffer)
        df .+= convolved

        return df
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

        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        F⁺ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, Nx, 2Ny-1, 2Nz-1, Nvx, Nvy, Nvz)
        for λxyz in CartesianIndices((Nx, 2Ny-1, 2Nz-1))
            for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
                F⁺[λxyz, λvx, λvy, λvz] = max(C[λxyz, λvx], 0) * f̂[λxyz, λvx, λvy, λvz]
                F⁻[λxyz, λvx, λvy, λvz] = min(C[λxyz, λvx], 0) * f̂[λxyz, λvx, λvy, λvz]
            end
        end

        F⁺ = reverse_quadratic_dealias(F⁺, buffer)
        F⁻ = reverse_quadratic_dealias(F⁻, buffer)

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dvy)
        left_biased_stencil = [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dvy)

        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        convolve_vy!(convolved, F⁻, right_biased_stencil, false, buffer)
        df .+= convolved
        convolve_vy!(convolved, F⁺, left_biased_stencil, false, buffer)
        df .+= convolved

        return df
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

