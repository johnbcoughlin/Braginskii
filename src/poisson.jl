import LinearAlgebra: mul!, ldiv!
import Base: eltype, size, *

function charge_density(sim, f, buffer)
    Nx, Ny, Nz = size(sim.x_grid)
    ρ_c = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @timeit "charge density" for i in eachindex(sim.species)
        α = sim.species[i]
        fi = f.x[i]
        species_density = density(fi, α, sim.By, buffer)
        ρ_c .+= density(fi, α, sim.By, buffer) .* α.q
    end

    @timeit "ρ_c" if length(sim.species) == 1
        @timeit "sum" ρ_sum = sum(ρ_c, dims=(1, 2, 3))
        #ρ0 = ρ_sum / (Nx*Ny*Nz)
        @. ρ_c -= ρ_sum / (Nx*Ny*Nz)
    end
    @. ρ_c *= sim.ωpτ
    ρ_c
end

function poisson(sim, f, buffer)
    grid = sim.x_grid

    ρ_c = charge_density(sim, f, buffer)

    @timeit "direct" poisson_direct(ρ_c, sim.Δ_lu, sim.ϕ_left, sim.ϕ_right, grid, 
        sim.x_dims, buffer, sim.ϕ, sim.fft_plans, grid.poisson_helper)
end

function poisson(ρ_c, ϕ_left, ϕ_right, grid, x_dims, buffer, ϕ, fft_plans, helper)
    Nx, Ny, Nz = size(grid)

    Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ey = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ez = alloc_array(Float64, buffer, Nx, Ny, Nz)

    poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper)

    return (Ex, Ey, Ez)
end

function poisson_direct(ρ_c, Δ_lu, ϕ_left, ϕ_right, grid, x_dims, buffer, ϕ, fft_plans, helper)
    Nx, Ny, Nz = size(grid)

    Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ey = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ez = alloc_array(Float64, buffer, Nx, Ny, Nz)

    poisson_direct!(ϕ, (Ex, Ey, Ez), ρ_c, Δ_lu, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper)

    return (Ex, Ey, Ez)
end

function poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid::XGrid, x_dims, buffer, fft_plans, helper)
    Δ = LaplacianOperator(grid, x_dims, buffer, ϕ_left, ϕ_right, fft_plans, helper)

    minus_ρ_c = -ρ_c
    @timeit "minres" minres!(reshape(ϕ, (:,)), Δ, minus_ρ_c, abstol=1e-12, maxiter=10)

    @timeit "potential gradient" potential_gradient!(Ex, Ey, Ez, ϕ, 
        ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper) 
end

function poisson_direct!(ϕ, (Ex, Ey, Ez), ρ_c, Δ_lu, ϕ_left, ϕ_right,
        grid, x_dims, buffer, fft_plans, helper)
    Nx, Ny, Nz = size(grid) 

    ϕ .= do_poisson_solve(Δ_lu, ρ_c, grid, ϕ_left, ϕ_right, helper, x_dims, fft_plans, buffer)

    @timeit "potential gradient" potential_gradient!(Ex, Ey, Ez, ϕ, 
        ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper) 
end

function prepare_poisson_rhs(ρ, grid, ϕ_left, ϕ_right, helper, x_dims, fft_plans, buffer)
    Nx, Ny, Nz = size(grid) 

    ρ_modified = alloc_array(Float64, buffer, Nx, Ny, Nz)
    ρ_modified .= ρ
    if :z ∈ x_dims
        ρ_modified[:, :, 1] .-= ϕ_left * helper.centered_second_derivative_stencil[1, :] 
        ρ_modified[:, :, end] .-= ϕ_right * helper.centered_second_derivative_stencil[3, :]
    end

    Kx = Nx ÷ 2 + 1
    rhs = alloc_array(Complex{Float64}, buffer, Kx, Ny, Nz)
    mul!(rhs, fft_plans.kxy_rfft, ρ_modified)
    if :z ∈ x_dims
        return rhs |> vec
    else
        # Set (0, 0) Fourier mode to zero
        rhs[1, 1, :] .= 0.0
        return rhs |> vec
    end
end

function postprocess_poisson_soln(ϕ̂, grid, fft_plans, buffer)
    Nx, Ny, Nz = size(grid) 
    ϕ = alloc_array(Float64, buffer, Nx, Ny, Nz)
    mul!(reshape(ϕ, (Nx, Ny, Nz)), fft_plans.kxy_irfft, reshape(ϕ̂, (:, Ny, Nz)))
    return ϕ
end

struct LaplacianOperator{BUF, LEFT, RIGHT, FFTPLANS, HELPER}
    grid::XGrid
    x_dims::Vector{Symbol}
    buffer::BUF
    ϕ_left::LEFT
    ϕ_right::RIGHT
    fft_plans::FFTPLANS
    poisson_helper::HELPER
end

size(Δ::LaplacianOperator, d) = begin
    # It's a square operator
    return prod(size(Δ.grid))
end

*(Δ::LaplacianOperator, ϕ) = begin
    y = alloc_zeros(Float64, Δ.buffer, size(Δ, 1))
    mul!(y, Δ, ϕ)
    return y
end

function mul!(y, Δ::LaplacianOperator, ϕ)
    no_escape(Δ.buffer) do
        @timeit "laplacian" apply_laplacian!(y, ϕ, Δ.ϕ_left, Δ.ϕ_right, Δ.grid, Δ.x_dims, Δ.buffer, Δ.fft_plans, Δ.poisson_helper)
    end
    return y
end

eltype(Δ::LaplacianOperator) = Float64

function apply_laplacian!(dest, ϕ, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper)
    Nx, Ny, Nz = size(grid)
    ϕ = reshape(ϕ, (Nx, Ny, Nz))
    dest = reshape(dest, (Nx, Ny, Nz))

    dz = grid.z.dx

    no_escape(buffer) do
        @timeit "z" if :z ∈ x_dims
            ϕ_with_z_bdy = alloc_array(Float64, buffer, Nx, Ny, Nz+2)
            ϕ_with_z_bdy[:, :, 2:Nz+1] .= ϕ
            ϕ_with_z_bdy[:, :, 1] .= ϕ_left
            ϕ_with_z_bdy[:, :, end] .= ϕ_right
            #@timeit "bcs" apply_poisson_bcs!(ϕ_with_z_bdy, ϕ_left, ϕ_right, helper)

            stencil = helper.centered_second_derivative_stencil
            @timeit "conv" convolve_z!(dest, ϕ_with_z_bdy, stencil, true, buffer)
            #@timeit "conv" convolve_z!(dest, ϕ, stencil, false, buffer)
        else
            dest .= 0
        end

        @timeit "x" if :x ∈ x_dims
            ϕ_xx = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_xx .= ϕ
            in_kxy_domain!(ϕ_xx, buffer, fft_plans) do ϕ̂
                #kxs = arraytype(buffer)(0:Nx÷2)
                kxs = 0:Nx÷2
                @timeit "kxs" @. ϕ̂ *= -kxs^2 * (2π / grid.x.L)^2
            end
            dest .+= ϕ_xx
        end

        @timeit "y" if :y ∈ x_dims
            ϕ_yy = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_yy .= ϕ
            in_kxy_domain!(ϕ_yy, buffer, fft_plans) do ϕ̂
                kys = arraytype(buffer)(mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2)))
                ϕ̂ .*= -(kys').^2 * (2π / grid.y.L)^2
            end
            dest .+= ϕ_yy
        end

        nothing
    end
end

function potential_gradient!(Ex, Ey, Ez, ϕ, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans, helper)
    Nx, Ny, Nz = size(grid)

    no_escape(buffer) do
        if :z ∈ x_dims
            #=
            ϕ_with_z_bdy = alloc_array(Float64, buffer, Nx, Ny, Nz+2)
            ϕ_with_z_bdy[:, :, 2:Nz+1] .= ϕ
            ϕ_with_z_bdy[:, :, 1] .= ϕ_left
            ϕ_with_z_bdy[:, :, end] .= ϕ_right
            #apply_poisson_bcs!(ϕ_with_z_bdy, ϕ_left, ϕ_right, helper)
            
            stencil = helper.centered_first_derivative_stencil
            ϕ_z = alloc_array(Float64, buffer, Nx, Ny, Nz)
            convolve_z!(ϕ_z, ϕ_with_z_bdy, stencil, true, buffer)
            Ez .= -arraytype(Ez)(ϕ_z)
            =#
            ϕ_z = alloc_array(Float64, buffer, Nx, Ny, Nz)
            mul!(vec(ϕ_z), grid.Dz_3rd_order, vec(ϕ))
            Ez .= -arraytype(Ez)(ϕ_z)
        else
            Ez .= 0
        end

        if :x ∈ x_dims
            ϕ_x = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_x .= ϕ
            in_kxy_domain!(ϕ_x, buffer, fft_plans) do ϕ̂
                kxs = (0:Nx÷2)
                ϕ̂ .*= im * kxs * (2π / grid.x.L)
            end
            Ex .= -arraytype(Ex)(ϕ_x)
        else
            Ex .= 0
        end

        if :y ∈ x_dims
            ϕ_y = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_y .= ϕ
            in_kxy_domain!(ϕ_y, buffer, fft_plans) do ϕ̂
                kys = alloc_array(Float64, buffer, Ny)
                kys .= mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))
                ϕ̂ .*= im  * (2π / grid.y.L) .* kys'
            end
            Ey .= -arraytype(Ey)(ϕ_y)
        else
            Ey .= 0
        end

        #@warn "zeroing out E field"
        #Ex .= Ey .= Ez .= 0

        nothing
    end
end

function eliminate_curl!(E, sim, buffer)
    Nx, Ny, Nz = size(sim.x_grid)

    Ex, Ey, Ez = E

    high_wave_number = norm(Ex[:, :, 1:2:end] .- Ex[:, :, 2:2:end])
    #@show high_wave_number

    # y component
    # (∇ × E)_y = ∂x Ez - ∂z Ex.
    # We'll fix the defect by modifying Ex.
    ∂x_Ez = ∂x(Ez, sim, buffer)
    ∂z_Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    mul!(vec(∂z_Ex), sim.x_grid.Dz, vec(Ex)) 
    
    defect = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @. defect = ∂x_Ez - ∂z_Ex
    #display(as_xz(defect))
    #Ex .+= ∂z_inv(defect, sim, buffer)

    ∂z_Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    mul!(vec(∂z_Ex), sim.x_grid.Dz, vec(Ex)) 
    
    defect = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @. defect = ∂x_Ez - ∂z_Ex
    #@show norm(defect)
    #println()

    #error()
end

function apply_poisson_bcs!(ϕ, ϕ_left, ϕ_right, helper)
    Nx, Ny, Nz6 = size(ϕ)
    
    # Do left side first
    rhs = -reshape(ϕ[:, :, 4:7], (:, 4)) * helper.Q_left' 
    rhs .+= 128 * helper.S1' .* vec(ϕ_left)

    ϕ_ghosts = reshape(rhs * helper.M_inv_left', (Nx, Ny, 3))
    ϕ[:, :, 1:3] .= ϕ_ghosts

    # Right side
    rhs = -reshape(ϕ[:, :, end-6:end-3], (:, 4)) * helper.Q_right'
    rhs .+= 128 * helper.S1' .* vec(ϕ_right)

    ϕ_ghosts = reshape(rhs * helper.M_inv_right', (Nx, Ny, 3))
    ϕ[:, :, end-2:end] .= ϕ_ghosts
end

function factorize_poisson_operator(Δ::SparseMatrixCSC)
    return lu(Δ)
end

function factorize_poisson_operator(Δ::CuSparseMatrixCSR)
    return CUSOLVERRF.RFLU(Δ)
end

function do_poisson_solve(Δ_lu, ρ_c, grid, ϕ_left, ϕ_right, helper, x_dims, fft_plans, buffer)
    @timeit "prepare rhs" ρ̂ = prepare_poisson_rhs(-ρ_c, grid, ϕ_left, ϕ_right, helper, x_dims, fft_plans, buffer)
    ρ̂_re = alloc_array(Float64, buffer, size(ρ̂)...)
    ρ̂_re .= real.(ρ̂)
    ρ̂_im = alloc_array(Float64, buffer, size(ρ̂)...)
    ρ̂_im .= imag.(ρ̂)

    Nx, Ny, Nz = size(grid)

    # Do in-place solves
    @timeit "ldiv" ldiv!(Δ_lu, ρ̂_re)
    ϕ̂_re = ρ̂_re
    @timeit "ldiv" ldiv!(Δ_lu, ρ̂_im)
    ϕ̂_im = ρ̂_im

    ϕ̂ = alloc_array(ComplexF64, buffer, size(ρ̂)...)
    @. ϕ̂ = ϕ̂_re + im * ϕ̂_im
    @timeit "postprocess" ϕ = postprocess_poisson_soln(ϕ̂, grid, fft_plans, buffer)
    #@warn "zeroing out electric potential"
    return ϕ
end
