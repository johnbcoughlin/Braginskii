include("free_streaming_kernels.jl")

function free_streaming!(df, f, species, buffer)
    no_escape(buffer) do
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
            @timeit "z" free_streaming_z!(df_z, f, species, buffer)
            df_fs .+= df_z
        end

        df .+= df_fs
        nothing
    end
    #species.name == "electrons" && @info "after free streaming" df[1, 1, 1:5, 1, 1, 1:2] df[1, 1, 1:5, 1:2, 1, 1]

    return estimate_max_freestreaming_eigenvalue(f, species)
end

function estimate_max_freestreaming_eigenvalue(f, α::Species{<:Hermite})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)
    (; vth) = α.discretization.vdisc
    x_grid = α.discretization.x_grid

    λ = 0.0
    if :x ∈ α.x_dims
        λ += vth * sqrt(Nvx) / x_grid.x.dx
    end
    if :y ∈ α.x_dims
        λ += vth * sqrt(Nvy) / x_grid.y.dx
    end
    if :z ∈ α.x_dims
        λ += vth * sqrt(Nvz) / x_grid.z.dx
    end
    return λ
end

function free_streaming_z!(df, f, species::Species{WENO5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    dz = xgrid.z.dx
    vgrid = discretization.vdisc.grid

    no_escape(buffer) do
        f_with_boundaries = z_free_streaming_bcs(f, species, buffer)

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

function free_streaming_z!(df, f, species::Species{<:Hermite}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    dz = xgrid.z.dx
    Ξz⁻ = discretization.vdisc.Ξz⁻
    Ξz⁺ = discretization.vdisc.Ξz⁺

    no_escape(buffer) do
        f_with_boundaries = z_free_streaming_bcs(f, species, buffer)
        #species.name == "electrons" && @info "" f_with_boundaries[1, 1, 1:10, 1, 1, 1:3]

        F⁻ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx*Nvy*Nvz)
        @timeit "mul" mul!(reshape(F⁻, (:, Nvx*Nvy*Nvz)), reshape(f_with_boundaries, (:, Nvx*Nvy*Nvz)), (Ξz⁻)')

        F⁺ = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx*Nvy*Nvz)
        @timeit "mul" mul!(reshape(F⁺, (:, Nvx*Nvy*Nvz)), reshape(f_with_boundaries, (:, Nvx*Nvy*Nvz)), (Ξz⁺)')

        right_biased_stencil, left_biased_stencil = xgrid.z_fd_stencils
        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        @timeit "conv" convolve_z!(convolved, reshape(F⁻, (Nx, Ny, Nz+6, Nvx, Nvy, Nvz)), right_biased_stencil, true, buffer)
        df .+= convolved
        @timeit "conv" convolve_z!(convolved, reshape(F⁺, (Nx, Ny, Nz+6, Nvx, Nvy, Nvz)), left_biased_stencil, true, buffer)
        df .+= convolved
    end
end

function z_free_streaming_bcs(f, species, buffer)
    (; discretization) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    if isa(species.z_bcs, ReflectingWallBCs)
        f_with_boundaries = alloc_array(Float64, buffer, Nx, Ny, Nz+6, Nvx, Nvy, Nvz)
        @timeit "bcs" reflecting_wall_bcs!(f_with_boundaries, f, discretization)
    elseif isa(species.z_bcs, ReservoirBC)
        f_with_boundaries = species.z_bcs.f_with_bcs
    elseif isnothing(species.z_bcs)
        error("No BCs specified for z free streaming")
    end

    #@info "" f_with_boundaries[1, 1, 1:10, 1, 1, 1]
    f_with_boundaries[:, :, 4:Nz+3, :, :, :] .= f
    return f_with_boundaries
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

function free_streaming_x!(df, f, species::Species{<:Hermite, <:FD5}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid
    Ξx⁻ = discretization.vdisc.Ξx⁻
    Ξx⁺ = discretization.vdisc.Ξx⁺

    no_escape(buffer) do
        f_with_boundaries = x_free_streaming_bcs(f, species, buffer)

        F⁻ = alloc_array(Float64, buffer, Nx+6, Ny, Nz, Nvx*Nvy*Nvz)
        @timeit "mul" mul!(reshape(F⁻, (:, Nvx*Nvy*Nvz)), reshape(f_with_boundaries, (:, Nvx*Nvy*Nvz)), (Ξx⁻)')

        F⁺ = alloc_array(Float64, buffer, Nx+6, Ny, Nz, Nvx*Nvy*Nvz)
        @timeit "mul" mul!(reshape(F⁺, (:, Nvx*Nvy*Nvz)), reshape(f_with_boundaries, (:, Nvx*Nvy*Nvz)), (Ξx⁺)')

        #right_biased_dx, left_biased_dx = xgrid.x_fd_sparsearrays
        right_biased_stencil, left_biased_stencil = xgrid.x_fd_stencils
        convolved = alloc_array(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

        @timeit "conv" convolve_x!(convolved, reshape(F⁻, (Nx+6, Ny, Nz, Nvx, Nvy, Nvz)), right_biased_stencil, true, buffer)
        df .+= convolved
        @timeit "conv" convolve_x!(convolved, reshape(F⁺, (Nx+6, Ny, Nz, Nvx, Nvy, Nvz)), left_biased_stencil, true, buffer)
        df .+= convolved

        #@timeit "conv-mul" mul!(reshape(result, (Nx, :)), right_biased_dx, reshape(F⁻, (Nx, :)))
        #@timeit "conv-mul" mul!(reshape(result, (Nx, :)), left_biased_dx, reshape(F⁺, (Nx, :)))
        #df .+= result
        nothing
    end
end

function x_free_streaming_bcs(f, α, buffer)
    (; discretization) = α
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    f_with_boundaries = alloc_array(Float64, buffer, Nx+6, Ny, Nz, Nvx, Nvy, Nvz)
    f_with_boundaries[4:Nx+3, :, :, :, :, :] .= f
    f_with_boundaries[1:3, :, :, :, :, :] .= @view f[Nx-2:Nx, :, :, :, :, :]
    f_with_boundaries[Nx+4:Nx+6, :, :, :, :, :] .= @view f[1:3, :, :, :, :, :]
    f_with_boundaries
end

function free_streaming_x!(df, f, species::Species{<:Any, <:PSFourier}, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid

    f = reshape(f, (Nx, Ny, :))
    df_size = size(df)
    df = reshape(df, (Nx, Ny, :))
    F = species.fft_plans.kxy_rfft

    no_escape(buffer) do
        Kx = (Nx ÷ 2 + 1)
        xy_modes = alloc_array(Complex{Float64}, buffer, Kx, Ny, Nz*Nvx*Nvy*Nvz)
        mul!(xy_modes, F, f)
        xy_modes = reshape(xy_modes, Kx, Ny, Nz, Nvx, Nvy*Nvz)
        kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1, Nvx)
        kxs .= (-im * 2π / xgrid.x.L) * ((0:Kx-1))
        xy_modes .*= kxs
        xy_modes = reshape(xy_modes, (Kx, Ny, :))
        df2 = alloc_array(Float64, buffer, size(df)...)
        mul!(df2, species.fft_plans.kxy_irfft, xy_modes)
        df = reshape(df, df_size)
        df2 = reshape(df2, df_size)
        mul_by_vx!(df, df2, discretization)
        nothing
    end
end

function free_streaming_y!(df, f, species::Species, buffer)
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    xgrid = discretization.x_grid

    f = reshape(f, (Nx, Ny, :))
    df_size = size(df)
    df = reshape(df, (Nx, Ny, :))
    F = species.fft_plans.kxy_rfft

    no_escape(buffer) do
        Kx = (Nx ÷ 2 + 1)
        Ky = Ny
        xy_modes = alloc_zeros(Complex{Float64}, buffer, Kx, Ny, Nz*Nvx*Nvy*Nvz)
        mul!(xy_modes, F, f)
        xy_modes = reshape(xy_modes, Kx, Ny, Nz, Nvx, Nvy, Nvz)
        kys = alloc_array(Complex{Float64}, buffer, 1, Ky, 1, 1, Nvy)
        kys .= (-im * 2π / xgrid.y.L) * arraytype(buffer)(mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))')
        xy_modes .*= kys
        xy_modes = reshape(xy_modes, (Kx, Ny, :))
        df2 = alloc_array(Float64, buffer, size(df)...)
        mul!(df2, species.fft_plans.kxy_irfft, xy_modes)
        df = reshape(df, df_size)
        df2 = reshape(df2, df_size)
        mul_by_vy!(df, df2, discretization)
        nothing
    end
end

function reflecting_wall_bcs!(f_with_boundaries, f, discretization::XVDiscretization{WENO5})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    # Left boundary
    f_with_boundaries[:, :, 1, :, :, 1:Nvz] .= f[:, :, 3, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, 2, :, :, 1:Nvz] .= f[:, :, 2, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, 3 , :, :, 1:Nvz] .= f[:, :, 1, :, :, Nvz:-1:1]

    # Right boundary
    f_with_boundaries[:, :, end, :, :, 1:Nvz] .= f[:, :, Nz-2, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, end-1, :, :, 1:Nvz] .= f[:, :, Nz-1, :, :, Nvz:-1:1]
    f_with_boundaries[:, :, end-2, :, :, 1:Nvz] .= f[:, :, Nz, :, :, Nvz:-1:1]
end

function reflecting_wall_bcs!(f_with_boundaries, f, discretization::XVDiscretization{<:Hermite})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    flips = discretization.vdisc.vz_flips_array

    @. f_with_boundaries[:, :, 1, :, :, :] = (@view f[:, :, 3, :, :, :]) * flips
    @. f_with_boundaries[:, :, 2, :, :, :] = (@view f[:, :, 2, :, :, :]) * flips
    @. f_with_boundaries[:, :, 3, :, :, :] = (@view f[:, :, 1, :, :, :]) * flips

    @. f_with_boundaries[:, :, end, :, :, :] = (@view f[:, :, Nz-2, :, :, :]) * flips
    @. f_with_boundaries[:, :, end-1, :, :, :] = (@view f[:, :, Nz-1, :, :, :]) * flips
    @. f_with_boundaries[:, :, end-2, :, :, :] = (@view f[:, :, Nz, :, :, :]) * flips
end
