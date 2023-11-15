#==============================
# Physical space discretization
===============================#

struct Grid1D
    N::Int
    dx::Float64
    min::Float64
    max::Float64
    nodes::Vector{Float64}
end

# A periodic grid from [0, L), with nodes at 0, Δx, 2Δx, ...
struct PeriodicGrid1D
    N::Int
    dx::Float64
    L::Float64
    nodes::Vector{Float64}
end

# Create a non-periodic uniform grid, with points at cell centers.
grid1d(N, min, max) = begin
    dx = (max - min) / N
    cell_centers = collect(LinRange(min+dx/2, max-dx/2, N))
    Grid1D(N, dx, min, max, cell_centers)
end

periodic_grid1d(N, L) = begin
    dx = L / N
    nodes = collect((0:N-1) * dx)
    PeriodicGrid1D(N, dx, L, nodes)
end

function hou_li_filter(N, buffer)
    res = ones(N)
    β = 36.0
    for i in 1:N
        s = i / N
        if i >= 4 && s >= 2/3
            res[i] = exp(-β * s^β)
        end
    end
    return arraytype(buffer)(res)
end

function hou_li_filter(buffer, mode_numbers::Vector{Int64})
    res = ones(length(mode_numbers))
    N = maximum(abs, mode_numbers)
    β = 36.0
    for (i, k) in enumerate(mode_numbers)
        s = abs(k) / N
        if s >= 2/3
            res[i] = exp(-β * s^β)
        end
    end
    return arraytype(buffer)(res)
end

struct PoissonHelper{A1, A2}
    centered_first_derivative_stencil::A1
    centered_second_derivative_stencil::A1

    M_inv_left::A2
    M_inv_right::A2

    Q_left::A2
    Q_right::A2

    S1::A1
end

function poisson_helper(dz, buffer)
    T = arraytype(buffer)

    centered_first_derivative_stencil = T([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / dz)
    centered_second_derivative_stencil = T([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / dz^2)

    M_inv_left = T([3  -20  90;
                    0   -5  60;
                    0    0  35] |> inv);
    M_inv_right = T([90 -20   3;
                     60  -5   0;
                     35   0   0] |> inv);

    Q_left = T([60   -5  0  0;
                90  -20  3  0;
                140 -70 28 -5]) .|> Float64
    Q_right = T([ 0  0  -5  60;
                  0  3 -20  90;
                 -5 28 -70 140]) .|> Float64;

    S1 = T(ones(3))

    return PoissonHelper(
        centered_first_derivative_stencil,
        centered_second_derivative_stencil,
        M_inv_left,
        M_inv_right,
        Q_left,
        Q_right,
        S1
    )
end

struct XGrid{XA, YA, ZA, FILTERS, STENCILS, POISSON}
    x::PeriodicGrid1D
    y::PeriodicGrid1D
    z::Grid1D

    X::XA
    Y::YA
    Z::ZA

    xy_hou_li_filters::FILTERS
    z_fd_stencils::STENCILS
    poisson_helper::POISSON

    XGrid(xgrid, ygrid, zgrid, buffer) = begin
        X = alloc_zeros(Float64, buffer, length(xgrid.nodes), 1, 1)
        copyto!(X, reshape(xgrid.nodes, (:, 1, 1)))

        Y = alloc_zeros(Float64, buffer, 1, length(ygrid.nodes), 1)
        copyto!(Y, reshape(ygrid.nodes, (1, :, 1)))

        Z = alloc_zeros(Float64, buffer, 1, 1, length(zgrid.nodes))
        copyto!(Z, reshape(zgrid.nodes, (1, 1, :)))

        dz = zgrid.dx
        right_biased_stencil = arraytype(buffer)([0, 1/20, -1/2, -1/3, 1, -1/4, 1/30]) * (-1 / dz)
        left_biased_stencil =  arraytype(buffer)([-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0]) * (-1 / dz)
        stencils = (right_biased_stencil, left_biased_stencil)

        helper = poisson_helper(dz, buffer)

        Nx = xgrid.N
        Kx = Nx÷2+1
        Ny = ygrid.N
        Ky = Ny
        σx = hou_li_filter(Kx, buffer)

        ky_mode_numbers = mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))
        σy = hou_li_filter(buffer, ky_mode_numbers)

        filters = (σx, σy)

        new{typeof(X), typeof(Y), typeof(Z), typeof(filters), typeof(stencils), typeof(helper)}(
            xgrid, ygrid, zgrid, X, Y, Z, filters, stencils, helper)
    end
end

function min_dx(grid::XGrid)
    result = Inf
    if grid.x.N != 1
        result = min(grid.x.dx, result)
    end
    if grid.y.N != 1
        result = min(grid.y.dx, result)
    end
    if grid.z.N != 1
        result = min(grid.z.dx, result)
    end
    return result
end

function form_fourier_domain_poisson_operator(ϕ_left, ϕ_right, grid, x_dims, buffer)
    Nx, Ny, Nz = size(grid)

    helper = grid.poisson_helper

    T = arraytype(buffer)
    ST = sparsearraytype(buffer)

    @assert ϕ_left ≈ ϕ_left[1] * T(ones(size(ϕ_left)))
    @assert ϕ_right ≈ ϕ_right[1] * T(ones(size(ϕ_right)))

    if :z ∈ x_dims
        Dzz = spzeros(Nz, Nz)
        u = T(zeros(1, 1, Nz))
        b = T(zeros(1, 1, Nz))
        z_grid = Helpers.z_grid_1d(Nz, grid.z.min, grid.z.max, buffer)
        for i in 1:Nz
            u .= 0.0
            u[i] = 1.0
            apply_laplacian!(b, u, T([ϕ_left[1]]), T([ϕ_right[1]]), z_grid, [:z], buffer, plan_ffts(z_grid, buffer), helper)
            Dzz[:, i] .= sparse(vec(b))
        end
    else
        Dzz = 0*I(1)
    end

    Kx = Nx÷2+1
    kxs = collect(0:Nx÷2)
    kys = mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))

    # If it's just x and y, we need to make sure the overall matrix is invertible.
    if :z ∉ x_dims
        kxs[1] = 1.0
        kys[1] = 1.0
    end

    if :x ∈ x_dims
        Dxx = Diagonal(-kxs.^2 * (2π / grid.x.L)^2)
    else
        Dxx = 0*I(1)
    end
    if :y ∈ x_dims
        Dyy = Diagonal(-kys.^2 * (2π / grid.y.L)^2)
    else
        Dyy = 0*I(1)
    end

    if isempty(x_dims)
        return sparse(I(1))
    end

    return ST(sparse(kron(I(Nz), I(Ny), Dxx) + kron(I(Nz), Dyy, I(Kx)) + kron(Dzz, I(Ny), I(Kx))))
end

size(grid::XGrid) = (grid.x.N, grid.y.N, grid.z.N)

#==============================
# Velocity space discretizations
===============================#

export approximate_f, vgrid_of, expand_f

struct VGrid{XA, YA, ZA}
    x::Grid1D
    y::Grid1D
    z::Grid1D

    VX::XA
    VY::YA
    VZ::ZA

    VGrid(dims, x, y, z, buffer) = begin
        @assert dims ⊆ [:vx, :vy, :vz]
        # Check that it's suitable for reflecting wall BCs
        if :vx ∈ dims
            @assert iseven(x.N) && x.max == -x.min
        end

        X = alloc_zeros(Float64, buffer, 1, 1, 1, length(x.nodes), 1, 1)
        copyto!(X, reshape(x.nodes, (1, 1, 1, :, 1, 1)))

        Y = alloc_zeros(Float64, buffer, 1, 1, 1, 1, length(y.nodes), 1)
        copyto!(Y, reshape(y.nodes, (1, 1, 1, 1, :, 1)))

        Z = alloc_zeros(Float64, buffer, 1, 1, 1, 1, 1, length(z.nodes))
        copyto!(Z, reshape(z.nodes, (1, 1, 1, 1, 1, :)))

        new{typeof(X), typeof(Y), typeof(Z)}(x, y, z, X, Y, Z)
    end
end

size(grid::VGrid) = (grid.x.N, grid.y.N, grid.z.N)

struct WENO5
    grid::VGrid
end

size(fd::WENO5) = size(fd.grid)

struct Hermite{SPARSE, FILTERS, FLIPS}
    Nvx::Int
    Nvy::Int
    Nvz::Int

    vth::Float64

    Ξx::SPARSE
    Ξy::SPARSE
    Ξz::SPARSE

    Ξx⁻::SPARSE
    Ξy⁻::SPARSE
    Ξz⁻::SPARSE

    Ξx⁺::SPARSE
    Ξy⁺::SPARSE
    Ξz⁺::SPARSE

    Dvx::SPARSE
    Dvy::SPARSE
    Dvz::SPARSE

    filters::FILTERS
    vz_flips_array::FLIPS
end

Hermite(Nvx, Nvy, Nvz, vth, device) = begin
    N = max(Nvx, Nvy, Nvz)

    Ξ = spdiagm(-1 => sqrt.(1:N-1), 1 => sqrt.(1:N-1)) * vth
    
    Ξx = Ξ[1:Nvx, 1:Nvx]
    Ξy = Ξ[1:Nvy, 1:Nvy]
    Ξz = Ξ[1:Nvz, 1:Nvz]

    #Λx, Rx = eigen(Array(Ξx))
    #Ξx⁻ = kron(I(Nvz), I(Nvy), sparsify(Rx * Diagonal(min.(Λx, 0.0)) / Rx))
    #Ξx⁺ = kron(I(Nvz), I(Nvy), sparsify(Rx * Diagonal(max.(Λx, 0.0)) / Rx))
    Ξx⁻ = kron(I(Nvz), I(Nvy), 0.5*Ξx - 0.5 * I * sqrt(Nvx) * vth)
    Ξx⁺ = kron(I(Nvz), I(Nvy), 0.5*Ξx + 0.5 * I * sqrt(Nvx) * vth)

    #Λy, Ry = eigen(Array(Ξy))
    #Ξy⁻ = kron(I(Nvz), sparsify(Ry * Diagonal(min.(Λy, 0.0)) / Ry), I(Nvx))
    #Ξy⁺ = kron(I(Nvz), sparsify(Ry * Diagonal(max.(Λy, 0.0)) / Ry), I(Nvx))
    Ξy⁻ = kron(I(Nvz), 0.5*Ξy - 0.5 * I * sqrt(Nvy) * vth, I(Nvx))
    Ξy⁺ = kron(I(Nvz), 0.5*Ξy + 0.5 * I * sqrt(Nvy) * vth, I(Nvx))

    #Λz, Rz = eigen(Array(Ξz))
    #Ξz⁻ = kron(sparsify(Rz * Diagonal(min.(Λz, 0.0)) / Rz), I(Nvy), I(Nvx))
    #Ξz⁺ = kron(sparsify(Rz * Diagonal(max.(Λz, 0.0)) / Rz), I(Nvy), I(Nvx))
    Ξz⁻ = kron(0.5*Ξz - 0.5 * I * sqrt(Nvz) * vth, I(Nvy), I(Nvx))
    Ξz⁺ = kron(0.5*Ξz + 0.5 * I * sqrt(Nvz) * vth, I(Nvy), I(Nvx))

    D = spdiagm(-1 => -sqrt.(1:N-1))

    Ξx = kron(I(Nvz), I(Nvy), Ξ[1:Nvx, 1:Nvx])
    Ξy = kron(I(Nvz), Ξ[1:Nvy, 1:Nvy], I(Nvx))
    Ξz = kron(Ξ[1:Nvz, 1:Nvz], I(Nvy), I(Nvx))

    Dvx = kron(I(Nvz), I(Nvy), D[1:Nvx, 1:Nvx])
    Dvy = kron(I(Nvz), D[1:Nvy, 1:Nvy], I(Nvx))
    Dvz = kron(D[1:Nvz, 1:Nvz], I(Nvy), I(Nvx))

    cx = if device == :cpu
        identity
    elseif device == :gpu
        CuSparseMatrixCSC
    end

    flip = i -> iseven(i-1) ? 1 : -1
    flips = reshape(arraytype(device)(flip.(1:Nvz)), (1, 1, 1, 1, Nvz))

    buffer = allocator(device)
    σvx = reshape(hou_li_filter(Nvx, buffer), (1, 1, 1, :))
    σvy = reshape(hou_li_filter(Nvy, buffer), (1, 1, 1, 1, :))
    σvz = reshape(hou_li_filter(Nvz, buffer), (1, 1, 1, 1, 1, :))
    filters = (σvx, σvy, σvz)

    Hermite(Nvx, Nvy, Nvz, vth, cx(Ξx), cx(Ξy), cx(Ξz), cx(Ξx⁻), cx(Ξy⁻), cx(Ξz⁻), 
        cx(Ξx⁺), cx(Ξy⁺), cx(Ξz⁺), cx(Dvx), cx(Dvy), cx(Dvz),
        filters, flips)
end

function sparsify(A)
    A[abs.(A) .< 1e-15] .= 0.0
    sparse(A)
end

size(hd::Hermite) = (hd.Nvx, hd.Nvy, hd.Nvz)

function vgrid_of(fd::WENO5, args...)
    return fd.grid
end

function vgrid_of(hd::Hermite, K=50, vmax=5.0, buffer=default_buffer())
    vmax = vmax*hd.vth

    dims = Symbol[]
    if hd.Nvx == 1
        vx_grid = grid1d(1, 0.0, 0.0)
    else
        vx_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vx)
    end
    if hd.Nvy == 1
        vy_grid = grid1d(1, 0.0, 0.0)
    else
        vy_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vy)
    end
    if hd.Nvz == 1
        vz_grid = grid1d(1, 0.0, 0.0)
    else
        vz_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vz)
    end

    return VGrid(dims, vx_grid, vy_grid, vz_grid, buffer)
end

struct XVDiscretization{VDISC}
    x_grid::XGrid
    vdisc::VDISC
end

average_vth(disc::XVDiscretization{<:Hermite}) = disc.vdisc.vth

size(disc::XVDiscretization) = tuple(size(disc.x_grid)..., size(disc.vdisc)...)

approximate_f(f, disc::XVDiscretization, dims, buffer) = begin
    result = alloc_zeros(Float64, buffer, size(disc)...)
    approximate_f!(result, f, disc.x_grid, disc.vdisc, dims)
    result
end

approximate_f!(result, f, x_grid, vdisc::WENO5, dims) = begin
    all_dimensions = [x_grid.X, x_grid.Y, x_grid.Z, vdisc.grid.VX, vdisc.grid.VY, vdisc.grid.VZ]
    dimensions = all_dimensions[[dims...]]
    result .= f.(dimensions...)
end

approximate_f!(result, f, x_grid, vdisc::Hermite, dims) = begin
    vdims = sum(dims .>= 4)
    f_all(args...) = f((args[dim] for dim in dims)...)
    (; Nvx, Nvy, Nvz, vth) = vdisc
    (; X, Y, Z) = x_grid
    result .= Float64.(bigfloat_weighted_hermite_expansion(f_all, Nvx-1, Nvy-1, Nvz-1, X, Y, Z, vth))
end

expand_f(f, disc::XVDiscretization{WENO5}, vgrid) = begin
    return copy(f)
end

expand_f(coefs, disc::XVDiscretization{<:Hermite}, vgrid) = begin
    expand_bigfloat_hermite_f(coefs, vgrid, disc.vdisc.vth)
end
