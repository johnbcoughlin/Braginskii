import LinearAlgebra: mul!
import Base: eltype, size, *

function poisson(sim, f, buffer)
    grid = sim.x_grid
    Nx, Ny, Nz = size(grid)

    ρ_c = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @timeit "charge density" for i in eachindex(sim.species)
        α = sim.species[i]
        fi = f.x[i]
        ρ_c .+= density(fi, α.discretization, α.v_dims, buffer) .* α.q
    end

    if length(sim.species) == 1
        ρ_sum = sum(ρ_c)
        ρ_c .-= ρ_sum / length(ρ_c)
    end
    @assert sum(ρ_c) < sqrt(eps())

    poisson(ρ_c, sim.ϕ_left, sim.ϕ_right, grid, 
        sim.x_dims, buffer, sim.ϕ, sim.cpu_fft_plans, grid.poisson_helper)
end

function poisson(ρ_c, ϕ_left, ϕ_right, grid, x_dims, buffer, ϕ, fft_plans, helper)
    Nx, Ny, Nz = size(grid)

    Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ey = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ez = alloc_array(Float64, buffer, Nx, Ny, Nz)

    cpu_buffer = allocator(:cpu)
    no_escape(cpu_buffer) do
        poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid, x_dims, cpu_buffer, fft_plans, helper)
    end

    return (Ex, Ey, Ez)
end

function poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid::XGrid, x_dims, cpu_buffer, fft_plans, helper)
    Δ = LaplacianOperator(grid, x_dims, cpu_buffer, 
        Array(ϕ_left), Array(ϕ_right), fft_plans, helper)

    ϕ = Array(ϕ)

    @timeit "minres" minres!(vec(ϕ), Δ, -Array(ρ_c), abstol=1e-12, maxiter=10)

    @timeit "potential gradient" potential_gradient!(Ex, Ey, Ez, ϕ, 
        Array(ϕ_left), Array(ϕ_right), grid, x_dims, cpu_buffer, fft_plans, helper) 
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
            ϕ_with_z_bdy = alloc_array(Float64, buffer, Nx, Ny, Nz+6)
            ϕ_with_z_bdy[:, :, 4:Nz+3] .= ϕ
            apply_poisson_bcs!(ϕ_with_z_bdy, ϕ_left, ϕ_right, helper)

            stencil = helper.centered_second_derivative_stencil
            convolve_z!(dest, ϕ_with_z_bdy, stencil, true, buffer)
        else
            dest .= 0
        end

        @timeit "x" if :x ∈ x_dims
            ϕ_xx = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_xx .= ϕ
            in_kxy_domain!(ϕ_xx, buffer, fft_plans) do ϕ̂
                kxs = arraytype(buffer)(0:Nx÷2)
                ϕ̂ .*= -kxs.^2 * (2π / grid.x.L)^2
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
    dz = grid.z.dx

    no_escape(buffer) do
        if :z ∈ x_dims
            ϕ_with_z_bdy = alloc_array(Float64, buffer, Nx, Ny, Nz+6)
            ϕ_with_z_bdy[:, :, 4:Nz+3] .= ϕ
            apply_poisson_bcs!(ϕ_with_z_bdy, ϕ_left, ϕ_right, helper)

            stencil = helper.centered_first_derivative_stencil
            ϕ_z = alloc_array(Float64, buffer, Nx, Ny, Nz)
            convolve_z!(ϕ_z, ϕ_with_z_bdy, stencil, true, buffer)
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

        nothing
    end
end

function apply_ik!(ϕ̂, factor=1.0, dim=1)
    N1, N2 = size(ϕ̂)
    for i in axes(ϕ̂, 1), k in axes(ϕ̂, 2), j in axes(ϕ̂, 3)
        c = dim == 1 ? i : k
        ϕ̂[i, k, j] *= im * (c-1) * factor
    end
end

function apply_k²!(ϕ̂, factor=1.0, dim=1)
    N1, N2 = size(ϕ̂)
    ϕ̂ = reshape(reinterpret(reshape, Float64, ϕ̂), (2N1, N2, :))
    for i in axes(ϕ̂, 1), k in axes(ϕ̂, 2), j in axes(ϕ̂, 3)
        c = dim == 1 ? i : k
        ϕ̂[i, k, j] *= -(c-1)^2 * factor
    end
end

#=
function apply_poisson_bcs!(ϕ, ϕ_left, ϕ_right, buffer)
    Nx, Ny, Nz6 = size(ϕ)
    Nz = Nz6-6
    
    # Do left side first
    M = arraytype(buffer)(
         [3  -20  90;
          0   -5  60;
          0    0  35]);
    Q = arraytype(buffer)(
        [60   -5  0  0;
         90  -20  3  0;
         140 -70 28 -5]);
    S1 = alloc_array(Float64, buffer, 3);
    S1 .= 1.0;

    #@show typeof(ϕ[:, :, 4:7])
    rhs = -reshape(ϕ[:, :, 4:7], (:, 4)) * Q' 
    #@show typeof(rhs)
    rhs .+= 128 * S1' .* vec(ϕ_left)

    ϕ_ghosts = reshape(rhs / M', (Nx, Ny, 3))
    ϕ[:, :, 1:3] .= ϕ_ghosts

    # Right side
    M = arraytype(buffer)(
         [90 -20   3;
          60  -5   0;
          35   0   0]);
    Q = arraytype(buffer)(
        [ 0  0  -5  60;
           0  3 -20  90;
           -5 28 -70 140]);
    rhs = -reshape(ϕ[:, :, end-6:end-3], (:, 4)) * Q'
    rhs .+= 128 * S1' .* vec(ϕ_right)

    ϕ_ghosts = reshape(rhs / M', (Nx, Ny, 3))
    ϕ[:, :, end-2:end] .= ϕ_ghosts
end
=#

function apply_poisson_bcs!(ϕ, ϕ_left, ϕ_right, helper)
    Nx, Ny, Nz6 = size(ϕ)
    Nz = Nz6-6
    
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
