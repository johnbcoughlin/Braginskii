import LinearAlgebra: mul!
import Base: eltype, size, *

function poisson(sim, f, buffer)
    grid = sim.x_grid
    Nx, Ny, Nz = size(grid)

    ρ_c = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    for i in eachindex(sim.species)
        α = sim.species[i]
        fi = f.x[i]
        ρ_c .+= density(fi, α.discretization, α.v_dims, buffer) .* α.q
    end

    if length(sim.species) == 1
        ρ_sum = sum(ρ_c)
        ρ_c .-= ρ_sum / length(ρ_c)
    end
    @assert sum(ρ_c) < sqrt(eps())

    poisson(ρ_c, sim.ϕ_left, sim.ϕ_right, grid, sim.x_dims, buffer, sim.ϕ, sim.fft_plans)
end

function poisson(ρ_c, ϕ_left, ϕ_right, grid, x_dims, buffer, ϕ, fft_plans)
    Nx, Ny, Nz = size(grid)

    Ex = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ey = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Ez = alloc_array(Float64, buffer, Nx, Ny, Nz)

    poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans)

    return (Ex, Ey, Ez)
end

function poisson!(ϕ, (Ex, Ey, Ez), ρ_c, ϕ_left, ϕ_right, grid::XGrid, x_dims, buffer, fft_plans)
    Δ = LaplacianOperator(grid, x_dims, buffer, ϕ_left, ϕ_right, fft_plans)

    minres!(vec(ϕ), Δ, -ρ_c, abstol=1e-12)

    potential_gradient!(Ex, Ey, Ez, ϕ, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans) 
end

struct LaplacianOperator{BUF, LEFT, RIGHT, FFTPLANS}
    grid::XGrid
    x_dims::Vector{Symbol}
    buffer::BUF
    ϕ_left::LEFT
    ϕ_right::RIGHT
    fft_plans::FFTPLANS
end

size(Δ::LaplacianOperator, d) = begin
    # It's a square operator
    return prod(size(Δ.grid))
end

*(Δ::LaplacianOperator, ϕ) = begin
    y = zeros(size(Δ, 1))
    mul!(y, Δ, ϕ)
    return y
end

function mul!(y, Δ::LaplacianOperator, ϕ)
    @no_escape Δ.buffer begin
        apply_laplacian!(y, ϕ, Δ.ϕ_left, Δ.ϕ_right, Δ.grid, Δ.x_dims, Δ.buffer, Δ.fft_plans)
    end
    return y
end

eltype(Δ::LaplacianOperator) = Float64

function apply_laplacian!(dest, ϕ, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans)
    Nx, Ny, Nz = size(grid)
    ϕ = reshape(ϕ, (Nx, Ny, Nz))
    dest = reshape(dest, (Nx, Ny, Nz))

    dx = grid.x.dx

    @no_escape buffer begin
        if :x ∈ x_dims
            ϕ_with_x_bdy = alloc_array(Float64, buffer, Nx+6, Ny, Nz)
            ϕ_with_x_bdy[4:Nx+3, :, :] .= ϕ
            apply_poisson_bcs!(ϕ_with_x_bdy, ϕ_left, ϕ_right)

            stencil = SVector[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / dx^2;
            convolve_x!(dest, ϕ_with_x_bdy, stencil, true, buffer)
        else
            dest .= 0
        end

        if :y ∈ x_dims
            ϕ_yy = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_yy .= ϕ
            in_ky_domain!(ϕ_yy, buffer, fft_plans) do ϕ̂
                apply_k²!(ϕ̂, (2π / grid.y.L)^2)
            end
            dest .+= ϕ_yy
        end

        if :z ∈ x_dims
            ϕ_zz = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_zz .= ϕ
            in_kz_domain!(ϕ_zz, buffer) do ϕ̂
                apply_k²!(ϕ̂, (2π / grid.z.L)^2)
            end
            dest .+= ϕ_zz
        end

        nothing
    end
end

function potential_gradient!(Ex, Ey, Ez, ϕ, ϕ_left, ϕ_right, grid, x_dims, buffer, fft_plans)
    Nx, Ny, Nz = size(grid)
    dx = grid.x.dx

    @no_escape buffer begin
        if :x ∈ x_dims
            ϕ_with_x_bdy = alloc_array(Float64, buffer, Nx+6, Ny, Nz)
            ϕ_with_x_bdy[4:Nx+3, :, :] .= ϕ
            apply_poisson_bcs!(ϕ_with_x_bdy, ϕ_left, ϕ_right)

            stencil = -1 * SVector[-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / dx
            convolve_x!(Ex, ϕ_with_x_bdy, stencil, true, buffer)
        else
            Ex .= 0
        end

        if :y ∈ x_dims
            ϕ_y = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_y .= ϕ
            in_ky_domain!(ϕ_y, buffer, fft_plans) do ϕ̂
                apply_ik!(ϕ̂, 2π / grid.y.L)
            end
            Ey .= -ϕ_y
        else
            Ey .= 0
        end

        if :z ∈ x_dims
            ϕ_z = alloc_array(Float64, buffer, Nx, Ny, Nz)
            ϕ_z .= ϕ
            in_kz_domain!(ϕ_z, buffer) do ϕ̂
                apply_ik!(ϕ̂, 2π / grid.z.L)
            end
            Ez .= -ϕ_z
        else
            Ez .= 0
        end

        nothing
    end
end

function apply_ik!(ϕ̂, factor=1.0)
    N1, N2 = size(ϕ̂)
    for i in axes(ϕ̂, 1), k in axes(ϕ̂, 2), j in axes(ϕ̂, 3)
        ϕ̂[i, k, j] *= im * (k-1) * factor
    end
end

function apply_k²!(ϕ̂, factor=1.0)
    N1, N2 = size(ϕ̂)
    ϕ̂ = reshape(reinterpret(reshape, Float64, ϕ̂), (2N1, N2, :))
    for i in axes(ϕ̂, 1), k in axes(ϕ̂, 2), j in axes(ϕ̂, 3)
        ϕ̂[i, k, j] *= -(k-1)^2 * factor
    end
end

function apply_poisson_bcs!(ϕ, ϕ_left, ϕ_right)
    Nx6, Ny, Nz = size(ϕ)
    Nx = Nx6-6
    
    # Do left side first
    M = @SMatrix [3  -20  90;
                  0   -5  60;
                  0    0  35];
    Q = @SMatrix [60   -5  0  0;
                  90  -20  3  0;
                  140 -70 28 -5];
    S1 = @SVector ones(3);

    rhs = -Q * reshape(ϕ[4:7, :, :], (4, :)) 
    rhs .+= 128 * S1 .* vec(ϕ_left)'

    ϕ_ghosts = reshape(M \ rhs, (3, Ny, Nz))
    ϕ[1:3, :, :] .= ϕ_ghosts

    # Right side
    M = @SMatrix [90 -20   3;
                  60  -5   0;
                  35   0   0];
    Q = @SMatrix [ 0  0  -5  60;
                   0  3 -20  90;
                  -5 28 -70 140];
    rhs = -Q * reshape(ϕ[end-6:end-3, :, :], (4, :)) 
    rhs .+= 128 * S1 .* vec(ϕ_right)'

    ϕ_ghosts = reshape(M \ rhs, (3, Ny, Nz))
    ϕ[end-2:end, :, :] .= ϕ_ghosts
end
