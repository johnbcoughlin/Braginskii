function apply_hyperdiffusion!(dF, F, sim, α, buffer)
    x_dims = sim.x_dims
    helper = sim.x_grid.poisson_helper
    Nx, Ny, Nz = size(sim.x_grid)

    hyperdiffusion_df = alloc_zeros(Float64, buffer, size(dF)...)

    no_escape(buffer) do
        @timeit "z" if :z ∈ x_dims
            F_with_bcs = z_diffusion_bcs(F, α)

            stencil = -helper.centered_fourth_derivative_stencil_fourth_order
            @timeit "conv" convolve_z!(hyperdiffusion_df, F_with_bcs, stencil, true, buffer)
        end

        @timeit "x" if :x ∈ x_dims
            F_xxxx = alloc_array(Float64, buffer, size(F)...)
            F_xxxx .= F
            in_kxy_domain!(F_xxxx, buffer, α.fft_plans) do F̂
                kxs = 0:Nx÷2
                @timeit "kxs" @. F̂ *= kxs^4 * (2π / sim.x_grid.x.L)^4
            end
            hyperdiffusion_df .+= F_xxxx
        end

        @timeit "y" if :y ∈ x_dims
            error("Don't support y yet")
        end
    end

    #@show norm(hyperdiffusion_df)

    dF .+= 1e-10 * hyperdiffusion_df
end

function z_diffusion_bcs(F, α)
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
