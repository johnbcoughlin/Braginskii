function drift_velocity(α, E, By, gz, buffer)
    Ex, Ey, Ez = E

    ux = alloc_zeros(Float64, buffer, size(By)...)
    uy = alloc_zeros(Float64, buffer, size(By)...)
    uz = alloc_zeros(Float64, buffer, size(By)...)

    # ExB drifts
    @. ux -= Ez*By / By^2
    @. uz += Ex*By / By^2

    # Gravitational drift
    @. ux -= α.m * gz * By / (α.q * By^2)

    # TODO other guiding center drifts
    
    return (ux, uy, uz)
end

function drifting!(df, f, u, α, buffer)
    ux, uy, uz = u

    no_escape(buffer) do
        df_drifts = alloc_zeros(Float64, buffer, size(df)...)
        if :x ∈ α.x_dims
            df_x = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "x" drifting_x!(df_x, f, ux, α, buffer)
            df_drifts .+= df_x
        end
        if :y ∈ α.x_dims
            df_y = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "y" drifting_y!(df_y, f, uy, α, buffer)
            df_drifts .+= df_y
        end
        if :z ∈ α.x_dims
            df_z = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "z" drifting_z!(df_z, f, uz, α, buffer)
            df_drifts .+= df_z
        end

        df .+= df_drifts
        nothing
    end
end

function drifting_x!(df, f, ux, α, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)
    xgrid = discretization.x_grid

    f = reshape(f, (Nx, Ny, :))
    df_size = size(df)
    df = reshape(df, (Nx, Ny, :))
    F = α.fft_plans.kxy_rfft

    no_escape(buffer) do
        Kx = (Nx ÷ 2 + 1)
        xy_modes = alloc_array(Complex{Float64}, buffer, Kx, Ny, Nz*Nμ*Nvy)
        mul!(xy_modes, F, f)
        kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1)
        kxs .= (-im * 2π / xgrid.x.L) * ((0:Kx-1))
        xy_modes .*= kxs
        df2 = alloc_array(Float64, buffer, size(df)...)
        mul!(df2, α.fft_plans.kxy_irfft, xy_modes)
        df = reshape(df, df_size)
        df2 = reshape(df2, df_size)
        @. df = ux * df2
        nothing
    end
end

function drifting_y!(df, f, uy, α, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)
    xgrid = discretization.x_grid

    f = reshape(f, (Nx, Ny, :))
    df_size = size(df)
    df = reshape(df, (Nx, Ny, :))
    F = α.fft_plans.kxy_rfft

    no_escape(buffer) do
        Kx = (Nx ÷ 2 + 1)
        Ky = Ny
        xy_modes = alloc_zeros(Complex{Float64}, buffer, Kx, Ny, Nz*Nμ*Nvy)
        mul!(xy_modes, F, f)
        xy_modes = reshape(xy_modes, Kx, Ny, Nz, Nμ, Nvy)
        kys = alloc_array(Complex{Float64}, buffer, 1, Ky)
        kys .= (-im * 2π / xgrid.y.L) * arraytype(buffer)(mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))')
        xy_modes .*= kys
        xy_modes = reshape(xy_modes, (Kx, Ny, :))
        df2 = alloc_array(Float64, buffer, size(df)...)
        mul!(df2, α.fft_plans.kxy_irfft, xy_modes)
        df = reshape(df, df_size)
        df2 = reshape(df2, df_size)
        @. df = uy * df2
        nothing
    end
end

function drifting_z!(df, f, uz, α::Species{<:HermiteLaguerre}, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)
    xgrid = discretization.x_grid


    no_escape(buffer) do
        f_with_boundaries, uz_with_boundaries = z_drifting_bcs(f, uz, α, buffer)

        uz⁺ = alloc_array(Float64, buffer, size(uz_with_boundaries)...)
        uz⁻ = alloc_array(Float64, buffer, size(uz_with_boundaries)...)
        @. uz⁺ = max(uz_with_boundaries, 0)
        @. uz⁻ = min(uz_with_boundaries, 0)

        F⁻ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nμ, Nvy)
        @. F⁻ = f_with_boundaries * uz⁻

        F⁺ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nμ, Nvy)
        @. F⁺ = f_with_boundaries * uz⁺

        right_biased_stencil, left_biased_stencil = xgrid.z_fd_stencils
        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nμ, Nvy)

        @timeit "conv" convolve_z!(convolved, reshape(F⁻, (Nx, Ny, Nz+6, Nμ, Nvy)), right_biased_stencil, true, buffer)
        df .+= convolved
        @timeit "conv" convolve_z!(convolved, reshape(F⁺, (Nx, Ny, Nz+6, Nμ, Nvy)), left_biased_stencil, true, buffer)
        df .+= convolved
    end
end

function z_drifting_bcs(f, uz, α, buffer)
    (; discretization) = α
    Nx, Ny, Nz, Nμ, Nvy = size(discretization)

    if isa(α.z_bcs, ReservoirBC)
        f_with_boundaries = α.z_bcs.f_with_bcs
        uz_with_boundaries = alloc_zeros(Float64, buffer, Nx, Ny, Nz+6)
        uz_with_boundaries[:, :, 1:3] .= uz[:, :, 1]
        uz_with_boundaries[:, :, end-2:end] .= uz[:, :, end]
    else
        error("Unknown or no BCs specified for z free streaming")
    end
    f_with_boundaries[:, :, 4:Nz+3, :, :] .= f
    uz_with_boundaries[:, :, 4:Nz+3] .= uz
    return f_with_boundaries
end
