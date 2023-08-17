function electrostatic!(df, f, Ex, Ey, Ez, By, species, buffer)
    @no_escape buffer begin
        df_es = alloc_zeros(Float64, buffer, size(df)...)

        if :vx ∈ species.v_dims
            electrostatic_x!(df_es, f, Ex, By, species, buffer)
        end
        if :vy ∈ species.v_dims
            electrostatic_y!(df_es, f, Ey, By, species, buffer)
        end
        if :vz ∈ species.v_dims
            electrostatic_z!(df_es, f, Ez, By, species, buffer)
        end

        df .+= df_es
        nothing
    end
end

function electrostatic_x!(df, f, Ex, By, species::Species{WENO5}, buffer)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid

    dvx = vgrid.x.dx

    @no_escape buffer begin
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, 1, 1, Nvz)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvz)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvz)

        @. C = q / m * (Ex - vgrid.VZ * By)

        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        F⁺ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)

        @. F⁺ = max(C, 0) * f̂
        @. F⁻ = min(C, 0) * f̂

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

function electrostatic_y!(df, f, Ey, By, species::Species{WENO5}, buffer)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid

    dvy = vgrid.y.dx

    @no_escape buffer begin
        C = alloc_array(Float64, buffer, Nx, Ny, Nz)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

        @. C = q / m * Ey

        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        F⁺ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)

        @. F⁺ = max(C, 0) * f̂
        @. F⁻ = min(C, 0) * f̂

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

function electrostatic_z!(df, f, Ez, By, species::Species{WENO5}, buffer)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid
    dvz = vgrid.z.dx

    @no_escape buffer begin
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)

        @. C = q / m * (Ez + vgrid.VX * By)

        C = quadratic_dealias(C, buffer)
        f̂ = quadratic_dealias(f, buffer)
        F⁺ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)

        @. F⁺ = max(C, 0) * f̂
        @. F⁻ = min(C, 0) * f̂

        F⁺ = reverse_quadratic_dealias(F⁺, buffer)
        F⁻ = reverse_quadratic_dealias(F⁻, buffer)

        right_biased_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30] * (-1 / dvz)
        left_biased_stencil = [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0] * (-1 / dvz)

        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        convolve_vz!(convolved, F⁻, right_biased_stencil, false, buffer)
        df .+= convolved
        convolve_vz!(convolved, F⁺, left_biased_stencil, false, buffer)
        df .+= convolved

        return df
    end
end

# Do the round trip from physical space to Fourier space back to physical space on twice
# as many grid points
function quadratic_dealias(u, buffer)
    Nx, Ny, Nz, Nvs... = size(u)
    u = reshape(u, (Nx, Ny, :))

    u_modes = alloc_zeros(Complex{Float64}, buffer, Nx, 2Ny, Nz*prod(Nvs))
    @timeit "plan fft" F = plan_rfft(u, [1, 2])
    u_modes_tmp = alloc_array(Complex{Float64}, buffer, Nx÷2+1, Ny, Nz*prod(Nvs))
    mul!(u_modes_tmp, F, u)

    copy_to_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, 2Nx-1, 2Ny, Nz*prod(Nvs))
    @timeit "plan fft" F⁻¹ = plan_irfft(u_modes, 2Nx-1, [1, 2])
    mul!(U, F⁻¹, u_modes)

    return reshape(U, (2Nx-1, 2Ny, Nz, Nvs...)) * (2Nx-1)/Nx * (2Ny)/Ny
end

function copy_to_first_half!(u_modes, u_modes_tmp)
    Nx, Ny, _ = size(u_modes_tmp)
    for λx in 1:Nx, λy in 1:Ny, λ in axes(u_modes_tmp, 3)
        u_modes[λx, λy, λ] = u_modes_tmp[λx, λy, λ]
    end
end

function reverse_quadratic_dealias(u, buffer)
    Nx̂, Nŷ, Nz, Nvs... = size(u)
    Nx = Nx̂÷2+1
    Ny = Nŷ÷2

    u = reshape(u, (Nx̂, Nŷ, :))

    u_modes_tmp = alloc_zeros(Complex{Float64}, buffer, Nx̂÷2+1, Nŷ, Nz*prod(Nvs))
    @timeit "plan fft" F = plan_rfft(u, [1, 2])
    mul!(u_modes_tmp, F, u)

    u_modes = alloc_array(Complex{Float64}, buffer, Nx÷2+1, Ny, Nz*prod(Nvs))
    copy_from_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, Nx, Ny, Nz*prod(Nvs))
    @timeit "plan fft" F⁻¹ = plan_irfft(u_modes, Nx, [1, 2])
    mul!(U, F⁻¹, u_modes)

    return reshape(U, (Nx, Ny, Nz, Nvs...)) / ( (2Nx-1)/Nx * (2Ny)/Ny )
end

function copy_from_first_half!(u_modes, u_modes_tmp)
    Nx, Ny, _ = size(u_modes)
    for λx in 1:Nx, λy in 1:Ny, λ in axes(u_modes, 3)
        u_modes[λx, λy, λ] = u_modes_tmp[λx, λy, λ]
    end
end

