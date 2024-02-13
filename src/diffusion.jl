function apply_hyperdiffusion!(dF, F, sim, α::Species{<:Any, <:FD5}, buffer)
    # no-op
end

function apply_hyperdiffusion!(dF, F, sim, α::Species{<:Any, <:PSFourier}, buffer)
    x_dims = sim.x_dims
    helper = sim.x_grid.poisson_helper
    Nx, Ny, Nz = size(sim.x_grid)

    hyperdiffusion = alloc_zeros(Float64, buffer, size(dF)...)

    dz = sim.x_grid.z.dx
    grid_scale_coef = sim.grid_scale_hyperdiffusion_coef
    K_cutoff = Nx ÷ 3
    η_x = grid_scale_coef * (2pi * K_cutoff)^(-4)
    η_z = grid_scale_coef * dz^4

    no_escape(buffer) do
        F_with_bcs = z_diffusion_bcs(F, α)

        @timeit "z" if :z ∈ x_dims
            stencil = η_z * helper.centered_fourth_derivative_stencil_fourth_order
            @timeit "conv" convolve_z!(hyperdiffusion, F_with_bcs, stencil, true, buffer)
        end

        @timeit "x" if :x ∈ x_dims
            F_xxxx = alloc_zeros(Float64, buffer, size(F)...)
            F_xxxx .= F
            in_kxy_domain!(F_xxxx, buffer, α.fft_plans) do F̂
                kxs = 0:Nx÷2
                @timeit "kxs" @. F̂ *= kxs^4 * (2π / sim.x_grid.x.L)^4
            end
            @. hyperdiffusion += η_x * F_xxxx
        end

        @timeit "xz" if false && :x ∈ x_dims && :z ∈ x_dims
            F_zz = alloc_zeros(Float64, buffer, size(F)...)
            stencil = sqrt(η_z) * helper.centered_second_derivative_stencil_sixth_order
            convolve_z!(F_zz, F_with_bcs, stencil, true, buffer)

            F_xxzz = F_zz
            in_kxy_domain!(F_xxzz, buffer, α.fft_plans) do F̂
                kxs = 0:Nx÷2
                @. F̂ *= -kxs^2 * (2π / sim.x_grid.x.L)^2
            end
            sqrt_eta_x = sqrt(η_x)
            @. hyperdiffusion += 2 * sqrt_eta_x * F_xxzz
        end

        @timeit "y" if :y ∈ x_dims
            error("Don't support y yet")
        end
    end

    #@show norm(hyperdiffusion)

    dF .-= hyperdiffusion
end

function z_diffusion_bcs(f, α::Species{<:Hermite})
    (; discretization) = α

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    if isa(α.z_bcs, ReservoirBC)
        f_with_boundaries = α.z_bcs.f_with_bcs
    else
        error("Unknown or no BCs specified for z free streaming")
    end
    f_with_boundaries[:, :, 4:Nz+3, :, :, :] .= f
    return f_with_boundaries
end

function z_diffusion_bcs(F, α::Species{<:HermiteLaguerre})
    (; discretization) = α

    Nx, Ny, Nz, Nμ, Nvy = size(discretization)

    if isa(α.z_bcs, ReservoirBC)
        f_with_boundaries = α.z_bcs.f_with_bcs
    else
        error("Unknown or no BCs specified for z free streaming")
    end
    f_with_boundaries[:, :, 4:Nz+3, :, :] .= F
    return f_with_boundaries
end

function apply_x_diffusion!(df, f, sim, α::Species{<:Any, <:FD5}, buffer)
    # no-op
end

function apply_x_diffusion!(df, f, sim, α::Species{<:Any, <:PSFourier}, buffer)
    x_dims = sim.x_dims
    if :x ∉ x_dims
        return
    end

    Nx, Ny, Nz = size(sim.x_grid)

    diffusion_df = alloc_zeros(Float64, buffer, size(df)...)

    no_escape(buffer) do
        @timeit "x" begin
            f_xx = alloc_array(Float64, buffer, size(f)...)
            f_xx .= f
            in_kxy_domain!(f_xx, buffer, α.fft_plans) do f̂
                kxs = 0:Nx÷2
                @. f̂ *= -kxs^2 * (2π / sim.x_grid.x.L)^2
            end
            diffusion_df .+= f_xx
        end
    end

    @. df += diffusion_df * sim.x_diffusion_profile
end
