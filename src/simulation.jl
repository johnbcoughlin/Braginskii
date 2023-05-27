import Base.size, Base.getproperty

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

struct XGrid{XA, YA, ZA}
    x::Grid1D
    y::PeriodicGrid1D
    z::PeriodicGrid1D

    X::XA
    Y::YA
    Z::ZA

    XGrid(xgrid, ygrid, zgrid) = begin
        X = xgrid.nodes
        Y = reshape(ygrid.nodes, (1, :))
        Z = reshape(zgrid.nodes, (1, 1, :))
        new{typeof(X), typeof(Y), typeof(Z)}(xgrid, ygrid, zgrid, X, Y, Z)
    end
end

size(grid::XGrid) = (grid.x.N, grid.y.N, grid.z.N)

struct VGrid
    x::Grid1D
    y::Grid1D
    z::Grid1D

    VGrid(dims, x, y, z) = begin
        # Check that it's suitable for reflecting wall BCs
        if :x ∈ dims
            @assert iseven(x.N) && x.max == -x.min
        end
        new(x, y, z)
    end
end

size(grid::VGrid) = (grid.x.N, grid.y.N, grid.z.N)

struct Grid{XA, YA, ZA, VXA, VYA, VZA}
    x::XGrid
    v::VGrid

    X::XA
    Y::YA
    Z::ZA
    VX::VXA
    VY::VYA
    VZ::VZA

    Grid(xgrid, vgrid) = begin
        X = xgrid.x.nodes
        Y = reshape(xgrid.y.nodes, (1, :))
        Z = reshape(xgrid.z.nodes, (1, 1, :))
        VX = reshape(vgrid.x.nodes, (1, 1, 1, :))
        VY = reshape(vgrid.y.nodes, (1, 1, 1, 1, :))
        VZ = reshape(vgrid.z.nodes, (1, 1, 1, 1, 1, :))
        new{typeof(X), typeof(Y), typeof(Z), typeof(VX), typeof(VY), typeof(VZ)}(xgrid, vgrid, X, Y, Z, VX, VY, VZ)
    end
end

size(grid::Grid) = (size(grid.x)..., size(grid.v)...)

struct Species{G<:Grid}
    name::String
    grid::G
    v_grid::VGrid
    x_dims::Vector{Symbol}
    v_dims::Vector{Symbol}
    q::Float64
    m::Float64
end

struct SimulationMetadata{SP}
    x_dims::Vector{Symbol}
    x_grid::XGrid

    species::SP
end

struct Simulation{BA, U, SP}
    metadata::SimulationMetadata{SP}
    Bz::BA
    u::U
end

getproperty(sim::Simulation, sym::Symbol) = begin
    if sym ∈ (:u, :metadata, :Bz)
        getfield(sim, sym)
    else
        getproperty(sim.metadata, sym)
    end
end

function vlasov_fokker_planck_step!(du, u, p, t)
    (; sim, Bz, λmax, buffer) = p
    vlasov_fokker_planck!(du, u, sim, Bz, λmax, buffer)
end

function vlasov_fokker_planck!(du, u, sim, Bz, λmax, buffer)
    λmax[] = 0.0

    Ex, Ey = poisson(u, sim, buffer)

    for i in eachindex(sim.species)
        α = sim.species[i]

        df = du.x[i]
        df .= 0

        @timeit "free streaming" free_streaming!(df, u.x[i], α, buffer)
        @timeit "electrostatic" electrostatic!(df, u.x[i], Ex, Ey, Bz, α, buffer)
    end
end

function runsim_lightweight!(sim, T, Δt)
    set_default_buffer_size!(100_000_000)

    buffer = default_buffer()

    t = 0.0
    λmax = Ref(0.0)
    u = sim.u
    while t < T
        stepdt = min(Δt, T-t)
        p = (; sim=sim.metadata, λmax, buffer, Bz=sim.Bz)
        success, sf = ssp_rk43(vlasov_fokker_planck_step!, u, p, t, stepdt, 1.0, buffer)

        if success
            t += stepdt
        else
            Δt *= sf
        end
    end
end
