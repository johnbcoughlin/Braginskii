#==============================
# Physical space discretization
===============================#

# Pseudospectral Fourier discretization
struct PSFourier end

# Fifth-order finite difference discretization
struct FD5 end

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

orszag_two_thirds_filter(N::Int, buffer) = orszag_two_thirds_filter(buffer, collect(0:N-1))

function orszag_two_thirds_filter(buffer, mode_numbers::Vector{Int64})
    res = ones(length(mode_numbers))
    N = maximum(abs, mode_numbers)
    for (i, k) in enumerate(mode_numbers)
        s = abs(k) / N
        if s >= 2/3
            res[i] = 0.0
        end
    end
    return arraytype(buffer)(res)
end

struct PoissonHelper{A1, A2}
    centered_first_derivative_stencil::A1
    centered_second_derivative_stencil::A1

    centered_second_derivative_stencil_sixth_order::A1
    centered_fourth_derivative_stencil_fourth_order::A1

    M_inv_left::A2
    M_inv_right::A2

    Q_left::A2
    Q_right::A2

    S1::A1
end

function poisson_helper(dz, buffer)
    T = arraytype(buffer)

    #centered_first_derivative_stencil = T([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / dz)
    centered_fourth_derivative_stencil_fourth_order = T([-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6] / dz^4)
    centered_second_derivative_stencil_sixth_order = T([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / dz^2)
    centered_first_derivative_stencil = T([-1/2, 0, 1/2] / dz)
    centered_second_derivative_stencil = T([1, -2, 1] / dz^2)

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
        centered_second_derivative_stencil_sixth_order,
        centered_fourth_derivative_stencil_fourth_order,
        M_inv_left,
        M_inv_right,
        Q_left,
        Q_right,
        S1
    )
end

struct XGrid{XDISC, XA, YA, ZA, FILTERS, STENCILS, POISSON, SPARSE, DENSE, WENO_STENCILS}
    x::PeriodicGrid1D
    y::PeriodicGrid1D
    z::Grid1D

    X::XA
    Y::YA
    Z::ZA

    Dz::SPARSE
    Dz_3rd_order::SPARSE
    Dz_inv::DENSE

    xy_hou_li_filters::FILTERS
    z_fd_stencils::STENCILS
    x_fd_stencils::STENCILS
    x_fd_sparsearrays::Tuple{SPARSE, SPARSE}
    poisson_helper::POISSON

    z_weno_left::WENO_STENCILS
    z_weno_right::WENO_STENCILS

    XGrid(xgrid, ygrid, zgrid, buffer) = begin
        X = alloc_zeros(Float64, buffer, length(xgrid.nodes), 1, 1)
        copyto!(X, reshape(xgrid.nodes, (:, 1, 1)))

        Y = alloc_zeros(Float64, buffer, 1, length(ygrid.nodes), 1)
        copyto!(Y, reshape(ygrid.nodes, (1, :, 1)))

        Z = alloc_zeros(Float64, buffer, 1, 1, length(zgrid.nodes))
        copyto!(Z, reshape(zgrid.nodes, (1, 1, :)))

        dz = zgrid.dx

        Nx = length(xgrid.nodes)
        Ny = length(ygrid.nodes)
        Nz = length(zgrid.nodes)

        if Nz >= 4
            Dz = spdiagm(-1 => -ones(Nz-1), 1 => ones(Nz-1))
            Dz[1, 1:3] .= [-3, 4, -1]
            Dz[end, end-2:end] .= [1, -4, 3]
            Dz = Dz ./ (2dz)

            Dz_invertible = copy(Dz)
            Dz_invertible[2, :] .+= Dz_invertible[1, :]
            Dz_invertible[1, :] .= 1 / dz
            Dz_inv = inv(Array(Dz_invertible))

            Dz_3rd_order = spdiagm(-1 => -2*ones(Nz-1), 0 => -3*ones(Nz), 
                1 => 6*ones(Nz-1), 2=>-1*ones(Nz-2)) / (6dz)
            Dz_3rd_order[1, 1:4] .= [-11, 18, -9, 2] / (6dz)
            Dz_3rd_order[Nz-1, Nz-3:Nz] .= [1, -6, 3, 2] / (6dz)
            Dz_3rd_order[Nz, Nz-3:Nz] .= [-2, 9, -18, 11] / (6dz)
        else
            Dz = spdiagm(0 => ones(1))
            Dz_3rd_order = spdiagm(0 => ones(1))
            Dz_inv = ones(1, 1)
        end

        ST = sparsearraytype(buffer)
        Dz = kron(Dz, I(Ny), I(Nx)) |> ST
        Dz_3rd_order = kron(Dz_3rd_order, I(Ny), I(Nx)) |> ST
        Dz_inv = Dz_inv |> arraytype(buffer)


        dx = xgrid.dx
        #right_stencil = [0, 1/20, -1/2, -1/3, 1, -1/4, 1/30]
        #left_stencil = [-1/30, 1/4, -1, 1/3, 1/2, -1/20, 0]
        # Trying out first-order differencing to check stability
        right_stencil = [0, 0, 0, -1, 1, 0, 0]
        left_stencil = [0, 0, -1, 1, 0, 0, 0]
        right_biased_stencil = arraytype(buffer)(right_stencil)
        left_biased_stencil =  arraytype(buffer)(left_stencil)
        z_stencils = (right_biased_stencil * (-1 / dz), left_biased_stencil * (-1 / dz))
        x_stencils = (right_biased_stencil * (-1 / dx), left_biased_stencil * (-1 / dx))

        helper = poisson_helper(dz, buffer)

        if Nx > 1
            dx = xgrid.dx
            x_fd_right_biased = make_sparse_array_from_stencil(
                right_stencil * (-1/dx), Nx; periodic=true) |> ST
            x_fd_left_biased = make_sparse_array_from_stencil(
                left_stencil * (-1/dx), Nx; periodic=true) |> ST
        else
            x_fd_right_biased = 0 * I(1) |> ST
            x_fd_left_biased = 0 * I(1) |> ST
        end
        x_fd_sparsearrays = (x_fd_right_biased, x_fd_left_biased)

        Nx = xgrid.N
        Kx = Nx÷2+1
        Ny = ygrid.N
        Ky = Ny
        σx = orszag_two_thirds_filter(Kx, buffer)

        ky_mode_numbers = mod.(0:Ny-1, Ref(-Ny÷2:(Ny-1)÷2))
        σy = orszag_two_thirds_filter(buffer, ky_mode_numbers)

        filters = (σx, σy)

        z_weno_left = left_biased_weno5_stencil(arraytype)
        z_weno_right = right_biased_weno5_stencil(arraytype)

        new{PSFourier, typeof(X), typeof(Y), typeof(Z), typeof(filters), typeof(z_stencils), typeof(helper), typeof(Dz), typeof(Dz_inv), typeof(z_weno_left)}(
            xgrid, ygrid, zgrid, X, Y, Z, Dz, Dz_3rd_order, Dz_inv, filters, z_stencils, x_stencils, 
            x_fd_sparsearrays, helper, z_weno_left, z_weno_right)
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

function form_fourier_domain_poisson_operator(grid::XGrid{<:FD5}, x_dims, buffer)
    Nx, Ny, Nz = size(grid)

    helper = grid.poisson_helper

    T = arraytype(buffer)
    ST = sparsearraytype(buffer)

    stencil = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
    if :z ∈ x_dims
        dz = grid.z.dx
        Dzz = make_sparse_array_from_stencil(stencil / dz^2, Nz; periodic=false)
    else
        Dzz = 0*I(1)
    end

    if :x ∈ x_dims
        dx = grid.x.dx
        Dxx = make_sparse_array_from_stencil(stencil / dx^2, Nx; periodic=true)
    else
        Dxx = 0*I(1)
    end

    Dyy = 0*I(1)

    result = ST(sparse(kron(I(Nz), I(Ny), Dxx) + kron(I(Nz), Dyy, I(Nx)) + kron(Dzz, I(Ny), I(Nx))))
    result
end

function form_fourier_domain_poisson_operator(grid::XGrid{<:PSFourier}, x_dims, buffer)
    Nx, Ny, Nz = size(grid)

    helper = grid.poisson_helper

    T = arraytype(buffer)
    ST = sparsearraytype(buffer)

    stencil = [1, -2, 1]
    if :z ∈ x_dims
        dz = grid.z.dx
        Dzz = make_sparse_array_from_stencil(stencil / dz^2, Nz; periodic=false)
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

    result = ST(sparse(kron(I(Nz), I(Ny), Dxx) + kron(I(Nz), Dyy, I(Kx)) + kron(Dzz, I(Ny), I(Kx))))
    result
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

struct Hermite{SPARSE, DENSE, VECTOR, FILTERS, FLIPS}
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

    RΞz::DENSE
    RΞz_inv::DENSE
    Λvz::VECTOR

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

    Λz, Rz = eigen(Array(Ξz))
    Rzinv = inv(Rz)
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
    densecx = if device == :cpu
        identity
    elseif device == :gpu
        cuda
    end

    flip = i -> iseven(i-1) ? 1 : -1
    flips = reshape(arraytype(device)(flip.(1:Nvz)), (1, 1, 1, 1, Nvz))

    buffer = allocator(device)
    σvx = reshape(hou_li_filter(Nvx, buffer), (1, 1, 1, :))
    σvy = reshape(hou_li_filter(Nvy, buffer), (1, 1, 1, 1, :))
    σvz = reshape(hou_li_filter(Nvz, buffer), (1, 1, 1, 1, 1, :))
    filters = (σvx, σvy, σvz)

    Hermite(Nvx, Nvy, Nvz, vth, cx(Ξx), cx(Ξy), cx(Ξz), cx(Ξx⁻), cx(Ξy⁻), cx(Ξz⁻), 
        cx(Ξx⁺), cx(Ξy⁺), cx(Ξz⁺), 
        densecx(Rz), densecx(Rzinv), densecx(Λz),
        cx(Dvx), cx(Dvy), cx(Dvz),
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

struct HermiteLaguerre{SPARSE, DENSE, FILTERS}
    Nμ::Int
    Nvy::Int

    μ0::Float64
    vth::Float64

    # Parallel direction
    Ξy::SPARSE
    Dvy::SPARSE

    # Perpendicular direction
    Dμ::DENSE
    ΞμDμ::DENSE

    Ξμ::DENSE

    filters::FILTERS
end

HermiteLaguerre(Nμ, Nvy, μ0, vth, device, buffer) = begin
    # Notation from Olver et al.
    R = spdiagm(0 => ones(Nμ), 1 => -ones(Nμ-1)) |> Array
    L = spdiagm(0 => 0.0 .+ (1:Nμ), -1 => -(1:Nμ-1)) |> Array
    W = spdiagm(-1 => (1:Nμ-1)) |> Array

    Dμ = L \ (W - 0.5 * inv(R)) / μ0 |> arraytype(buffer)
    ΞμDμ = R*W - 0.5*I(Nμ) |> arraytype(buffer)

    L_minus_1 = spdiagm(0 => -1.0 .+ (1:Nμ), -1 => -(1:Nμ-1)) |> Array
    Ξμ = kron(I(Nvy), R * L_minus_1 * μ0) |> arraytype(buffer)

    cx = if device == :cpu
        identity
    elseif device == :gpu
        CuSparseMatrixCSC
    end

    Ξy = spdiagm(-1 => sqrt.(1:Nvy-1), 1 => sqrt.(1:Nvy-1)) * vth |> cx
    Dvy = spdiagm(-1 => -sqrt.(1:Nvy-1)) |> cx

    σμ = reshape(hou_li_filter(Nμ, buffer), (1, 1, 1, :))
    σvy = reshape(hou_li_filter(Nvy, buffer), (1, 1, 1, 1, :))

    filters = (σμ, σvy)

    HermiteLaguerre(Nμ, Nvy, μ0, vth, cx(Ξy), cx(Dvy), Dμ, ΞμDμ, Ξμ, filters)
end

size(hl::HermiteLaguerre) = (hl.Nμ, hl.Nvy)

struct GyroVGrid{μA, YA}
    μ::Grid1D
    y::Grid1D

    Vμ::μA
    VY::YA

    GyroVGrid(dims, μ, y, buffer) = begin
        @assert dims ⊆ [:μ, :vy]
        Vμ = alloc_zeros(Float64, buffer, 1, 1, 1, length(μ.nodes), 1)
        copyto!(Vμ, reshape(μ.nodes, (1, 1, 1, :, 1)))

        Y = alloc_zeros(Float64, buffer, 1, 1, 1, 1, length(y.nodes))
        copyto!(Y, reshape(y.nodes, (1, 1, 1, 1, :)))

        new{typeof(Vμ), typeof(Y)}(μ, y, Vμ, Y)
    end
end

function vgrid_of(hl::HermiteLaguerre; K=50, vmax=5.0, μmax=vmax^2/2, buffer=default_buffer())
    vmax = vmax*hl.vth
    μmax = μmax*hl.μ0

    dims = Symbol[]
    if hl.Nμ == 1
        μ_grid = grid1d(1, 0.0, 0.0)
    else
        μ_grid = grid1d(K, 0.0, μmax)
        push!(dims, :μ)
    end
    if hl.Nvy == 1
        vy_grid = grid1d(1, 0.0, 0.0)
    else
        vy_grid = grid1d(K, -vmax, vmax)
        push!(dims, :vy)
    end

    return GyroVGrid(dims, μ_grid, vy_grid, buffer)
end

size(grid::GyroVGrid) = (grid.μ.N, grid.y.N)

struct XVDiscretization{VDISC, XDISC}
    x_grid::XGrid{XDISC}
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
    f_all(args...) = f((args[dim] for dim in dims)...)
    (; Nvx, Nvy, Nvz, vth) = vdisc
    (; X, Y, Z) = x_grid
    result .= Float64.(bigfloat_weighted_hermite_expansion(f_all, Nvx-1, Nvy-1, Nvz-1, X, Y, Z, vth))
end

approximate_f!(result, f, x_grid, vdisc::HermiteLaguerre, dims) = begin
    f_all(args...) = f((args[dim] for dim in dims)...)
    (; Nμ, Nvy, μ0, vth) = vdisc
    (; X, Y, Z) = x_grid
    result .= Float64.(bigfloat_weighted_laguerre_expansion(f_all, Nμ-1, Nvy-1, X, Y, Z, μ0, vth))
end

expand_f(f, disc::XVDiscretization{WENO5}, vgrid) = begin
    return copy(f)
end

expand_f(coefs, disc::XVDiscretization{<:Hermite}, vgrid) = begin
    expand_bigfloat_hermite_f(coefs, vgrid, disc.vdisc.vth)
end
