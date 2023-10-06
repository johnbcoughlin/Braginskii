function electrostatic!(df, f, Ex, Ey, Ez, By, species, buffer, xgrid_fft_plans)
    no_escape(buffer) do
        df_es = alloc_zeros(Float64, buffer, size(df)...)

        if :vx ∈ species.v_dims
            @timeit "x" electrostatic_x!(df_es, f, Ex, By, species, buffer, xgrid_fft_plans)
        end
        if :vy ∈ species.v_dims
            electrostatic_y!(df_es, f, Ey, By, species, buffer, xgrid_fft_plans)
        end
        if :vz ∈ species.v_dims
            @timeit "z" electrostatic_z!(df_es, f, Ez, By, species, buffer, xgrid_fft_plans)
        end

        df .+= df_es
        nothing
    end
end

function electrostatic_x!(df, f, Ex, By, species::Species{WENO5}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid

    dvx = vgrid.x.dx

    no_escape(buffer) do
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, 1, 1, Nvz)
        @. C = q / m * (Ex - vgrid.VZ * By)

        F⁺ = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
        @. F⁺ = max(C, 0) * f
        @. F⁻ = min(C, 0) * f

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

function electrostatic_x!(df, f, Ex, By, species::Species{<:Hermite}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    no_escape(buffer) do
        vzf = alloc_array(Float64, buffer, size(f)...)
        @timeit "mul by vz" mul_by_vz!(vzf, f, discretization)

        F = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        @. F = q / m * (Ex * f - vzf * By)

        df = reshape(df, (:, Nvx*Nvy*Nvz))
        F = reshape(F, (:, Nvx*Nvy*Nvz))
        mul!(df, F, discretization.vdisc.Dvx', -1 / discretization.vdisc.vth, 1.0)
    end
end

function electrostatic_y!(df, f, Ey, By, species::Species{WENO5}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid

    dvy = vgrid.y.dx

    no_escape(buffer) do
        C = alloc_array(Float64, buffer, Nx, Ny, Nz)
        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

        @. C = q / m * Ey

        C = quadratic_dealias(C, buffer, xgrid_fft_plans)
        f̂ = quadratic_dealias(f, buffer, species.fft_plans)
        F⁺ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)

        @. F⁺ = max(C, 0) * f̂
        @. F⁻ = min(C, 0) * f̂

        F⁺ = reverse_quadratic_dealias(F⁺, buffer, species.fft_plans)
        F⁻ = reverse_quadratic_dealias(F⁻, buffer, species.fft_plans)

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

function electrostatic_y!(df, f, Ey, By, species::Species{<:Hermite}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    no_escape(buffer) do
        f̂ = quadratic_dealias(f, buffer, species.fft_plans)
        F = alloc_array(Float64, buffer, 2Nx-1, 2Ny, Nz, Nvx, Nvy, Nvz)

        Ey = quadratic_dealias(Ey, buffer, xgrid_fft_plans)

        @. F = q / m * Ey * f̂

        F = reverse_quadratic_dealias(F, buffer, species.fft_plans)

        df = reshape(df, (:, Nvx*Nvy*Nvz))
        F = reshape(F, (:, Nvx*Nvy*Nvz))
        mul!(df, F, discretization.vdisc.Dvy', -1 / discretization.vdisc.vth, 1.0)
    end
end

function electrostatic_z!(df, f, Ez, By, species::Species{WENO5}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vgrid = discretization.vdisc.grid
    dvz = vgrid.z.dx

    no_escape(buffer) do
        C = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx)

        @. C = q / m * (Ez + vgrid.VX * By)

        F⁺ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)
        F⁻ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx)

        F⁺ = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
        F⁻ = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        @. F⁺ = max(C, 0) * f
        @. F⁻ = min(C, 0) * f

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

function electrostatic_z!(df, f, Ez, By, species::Species{<:Hermite}, buffer, xgrid_fft_plans)
    (; discretization, q, m) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    no_escape(buffer) do
        vxf = alloc_array(Float64, buffer, size(f)...)
        mul_by_vx!(vxf, f, discretization)

        F = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        @. F = q / m * (Ez * f + vxf * By)

        df = reshape(df, (:, Nvx*Nvy*Nvz))
        F = reshape(F, (:, Nvx*Nvy*Nvz))
        mul!(df, F, discretization.vdisc.Dvz', -1 / discretization.vdisc.vth, 1.0)
    end
end

# Do the round trip from physical space to Fourier space back to physical space on twice
# as many grid points
function quadratic_dealias(u, buffer, fft_plans)
    Nx, Ny, Nz, Nvs... = size(u)
    u = reshape(u, (Nx, Ny, :))

    u_modes = alloc_zeros(Complex{Float64}, buffer, Nx, 2Ny, Nz*prod(Nvs))
    F = fft_plans.kxy_rfft
    #@timeit "plan fft" F = plan_rfft(u, [1, 2])
    u_modes_tmp = alloc_array(Complex{Float64}, buffer, Nx÷2+1, Ny, Nz*prod(Nvs))
    mul!(u_modes_tmp, F, u)

    copy_to_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, 2Nx-1, 2Ny, Nz*prod(Nvs))
    F⁻¹ = fft_plans.kxy_double_irfft
    mul!(U, F⁻¹, u_modes)
    U .*= (2Nx-1)/Nx * (2Ny)/Ny

    return reshape(U, (2Nx-1, 2Ny, Nz, Nvs...))
end

function copy_to_first_half!(u_modes, u_modes_tmp)
    Nx, Ny, _ = size(u_modes_tmp)
    for λx in 1:Nx, λy in 1:Ny, λ in axes(u_modes_tmp, 3)
        u_modes[λx, λy, λ] = u_modes_tmp[λx, λy, λ]
    end
end

function copy_to_first_half!(u_modes::CuArray, u_modes_tmp)
    Nx, Ny, _ = size(u_modes_tmp)
    u_modes[1:Nx, 1:Ny, :] .= u_modes_tmp[1:Nx, 1:Ny, :]
end

function reverse_quadratic_dealias(u, buffer, fft_plans)
    Nx̂, Nŷ, Nz, Nvs... = size(u)
    Nx = Nx̂÷2+1
    Ny = Nŷ÷2

    u = reshape(u, (Nx̂, Nŷ, :))

    u_modes_tmp = alloc_zeros(Complex{Float64}, buffer, Nx̂÷2+1, Nŷ, Nz*prod(Nvs))
    F = fft_plans.kxy_double_rfft
    mul!(u_modes_tmp, F, u)

    u_modes = alloc_array(Complex{Float64}, buffer, Nx÷2+1, Ny, Nz*prod(Nvs))
    copy_from_first_half!(u_modes, u_modes_tmp)

    U = alloc_zeros(Float64, buffer, Nx, Ny, Nz*prod(Nvs))
    F⁻¹ = fft_plans.kxy_irfft
    mul!(U, F⁻¹, u_modes)
    U ./= (2Nx-1)/Nx * (2Ny)/Ny

    return reshape(U, (Nx, Ny, Nz, Nvs...))
end

function copy_from_first_half!(u_modes, u_modes_tmp)
    Nx, Ny, _ = size(u_modes)
    for λx in 1:Nx, λy in 1:Ny, λ in axes(u_modes, 3)
        u_modes[λx, λy, λ] = u_modes_tmp[λx, λy, λ]
    end
end

function copy_from_first_half!(u_modes::CuArray, u_modes_tmp)
    Nx, Ny, _ = size(u_modes)
    u_modes[1:Nx, 1:Ny, :] .= u_modes_tmp[1:Nx, 1:Ny, :]
end

