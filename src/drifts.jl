function max_drift_eigenvalue(α, E, By, gz, buffer)
    xgrid = α.discretization.x_grid

    Ex, Ey, Ez = E

    ux = alloc_zeros(Float64, buffer, size(By)...)
    uy = alloc_zeros(Float64, buffer, size(By)...)
    uz = alloc_zeros(Float64, buffer, size(By)...)

    # ExB drifts
    @. ux += abs(Ez*By / By^2)
    @. uz += abs(Ex*By / By^2)

    # Gravitational drift
    @. ux += abs(α.m * gz * By / (α.q * By^2))

    # Grad-B drift
    grad_By_z = alloc_array(Float64, buffer, size(By)...)
    mul!(vec(grad_By_z), xgrid.Dz, vec(By))
    μ0 = α.discretization.vdisc.μ0
    @. ux += 3μ0 * max(grad_By_z / By)

    # TODO other guiding center drifts

    λmax = 0.0
    if :x ∈ α.x_dims
        λx = maximum(abs, ux) / xgrid.x.dx 
        λmax = max(λx, λmax)
    end
    if :y ∈ α.x_dims
        λy = maximum(abs, uy) / xgrid.y.dx 
        λmax = max(λy, λmax)
    end
    if :z ∈ α.x_dims
        λz = maximum(abs, uz) / xgrid.z.dx 
        λmax = max(λz, λmax)
    end

    #@show max_drift_eigenvalue = λmax
    
    return λmax
end

function drifting!(df, f, α, E, sim, buffer)
    no_escape(buffer) do
        df_drifts = alloc_zeros(Float64, buffer, size(df)...)
        if :x ∈ α.x_dims
            df_x = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "x" drifting_x!(df_x, f, α, E, sim, buffer)
            df_drifts .+= df_x
        end
        if :y ∈ α.x_dims
            df_y = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "y" drifting_y!(df_y, f, uy, α, buffer)
            df_drifts .+= df_y
        end
        if :z ∈ α.x_dims
            df_z = alloc_zeros(Float64, buffer, size(α.discretization)...)
            @timeit "z" drifting_z!(df_z, f, α, E, sim, buffer)
            df_drifts .+= df_z
        end

        df .+= df_drifts
        nothing
    end

    check_divergence_of_ExB(E, sim, buffer)

    #return max_drift_eigenvalue(α, E, sim.By, sim.gz, buffer)
    return 0.0
end

function check_divergence_of_ExB(E, sim, buffer)
    xgrid = sim.x_grid
    Nx, Ny, Nz = size(xgrid)

    Ex, _, Ez = E

    ExB_x = -Ez
    ExB_z = Ex

    Kx = Nx÷2+1
    dx_ExB_x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    xy_modes = alloc_zeros(Complex{Float64}, buffer, Kx, Ny, Nz)
    mul!(xy_modes, sim.fft_plans.kxy_rfft, ExB_x)
    kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1)
    kxs .= (im * 2π / xgrid.x.L) * ((0:Kx-1))
    xy_modes .*= kxs
    mul!(dx_ExB_x, sim.fft_plans.kxy_irfft, xy_modes)

    dz_ExB_z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    mul!(vec(dz_ExB_z), xgrid.Dz, vec(ExB_z))

    div_ExB = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    div_ExB += dx_ExB_x + dz_ExB_z

    #@show norm(div_ExB) / sum(norm(dx_ExB_x) + norm(dz_ExB_z))
end

function drift_ux_F(F, α, E, sim, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(F)

    Ξμ = discretization.vdisc.Ξμ

    Ex, _, Ez = E
    (; By, ωpτ, ωcτ, gz) = sim

    ux = alloc_zeros(Float64, buffer, size(By)...)

    # ExB drifts
    @. ux -= (ωpτ/ωcτ) * Ez*By / By^2

    # Gravitational drift
    if gz != 0
        @. ux -= (1/ωcτ) * α.m * gz * By / (α.q * By^2)
    end

    ux_F = alloc_array(Float64, buffer, size(F)...)
    @. ux_F = ux * F
    
    # Grad-B drift
    μ_F = alloc_array(Float64, buffer, size(F)...)
    mul!(reshape(μ_F, (:, Nμ*Nvy)), reshape(F, (:, Nμ*Nvy)), Ξμ')

    grad_By_z = alloc_array(Float64, buffer, size(By)...)
    mul!(vec(grad_By_z), sim.x_grid.Dz, vec(By))

    @. ux_F += (1 / ωcτ) * μ_F / α.q * grad_By_z * By / By^2

    #@show maximum(abs, ux_F ./ F)

    return ux_F
end

function drifting_x!(dF, F, α, E, sim, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)
    xgrid = discretization.x_grid

    dF = reshape(dF, (Nx, Ny, :))
    transform = α.fft_plans.kxy_rfft

    no_escape(buffer) do
        Kx = (Nx ÷ 2 + 1)
        ux_F = drift_ux_F(F, α, E, sim, buffer)
        xy_modes = alloc_array(Complex{Float64}, buffer, Kx, Ny, Nz*Nμ*Nvy)
        mul!(xy_modes, transform, reshape(ux_F, (Nx, Ny, :)))

        kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1)
        kxs .= (-im * 2π / xgrid.x.L) * ((0:Kx-1))
        xy_modes .*= kxs
        mul!(dF, α.fft_plans.kxy_irfft, xy_modes)
        nothing
    end
end

function drifting_y!(dF, F, α, sim, buffer)
    error("Not yet implemented")
end

function drift_uz_F(F, α, E, sim, buffer)
    Ex, _, _ = E

    (; By, ωpτ, ωcτ, gz) = sim

    uz = alloc_zeros(Float64, buffer, size(By)...)

    # ExB drift
    @. uz += (ωpτ/ωcτ) * Ex * By / By^2

    F_with_bcs, uz_with_bcs = z_drifting_bcs(F, uz, α, buffer)

    uz_F⁺ = alloc_array(Float64, buffer, size(F_with_bcs)...)
    uz_F⁻ = alloc_array(Float64, buffer, size(F_with_bcs)...)

    uz⁺ = alloc_array(Float64, buffer, size(uz_with_bcs)...)
    uz⁻ = alloc_array(Float64, buffer, size(uz_with_bcs)...)
    @. uz⁺ = max(uz_with_bcs, 0)
    @. uz⁻ = min(uz_with_bcs, 0)

    @. uz_F⁺ = uz⁺ * F_with_bcs
    @. uz_F⁻ = uz⁻ * F_with_bcs

    return uz_F⁺, uz_F⁻
end

function drifting_z!(dF, F, α::Species{<:HermiteLaguerre}, E, sim, buffer)
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)
    xgrid = discretization.x_grid

    no_escape(buffer) do
        uz_F⁺, uz_F⁻ = drift_uz_F(F, α, E, sim, buffer)

        right_biased_stencil, left_biased_stencil = xgrid.z_fd_stencils
        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nμ, Nvy)

        @timeit "conv" convolve_z!(convolved, reshape(uz_F⁻, (Nx, Ny, Nz+6, Nμ, Nvy)), right_biased_stencil, true, buffer)
        dF .+= convolved
        @timeit "conv" convolve_z!(convolved, reshape(uz_F⁺, (Nx, Ny, Nz+6, Nμ, Nvy)), left_biased_stencil, true, buffer)
        dF .+= convolved
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
    return f_with_boundaries, uz_with_boundaries
end

