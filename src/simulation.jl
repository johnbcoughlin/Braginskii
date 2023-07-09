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
        X = zeros(length(xgrid.nodes), 1, 1)
        X .= reshape(xgrid.nodes, (:, 1, 1))

        Y = zeros(1, length(ygrid.nodes), 1)
        Y .= reshape(ygrid.nodes, (1, :, 1))

        Z = zeros(1, 1, length(zgrid.nodes))
        Z .= reshape(zgrid.nodes, (1, 1, :))

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
        X = zeros(length(xgrid.x.nodes), 1, 1, 1, 1, 1)
        X .= reshape(xgrid.x.nodes, (:, 1, 1, 1, 1, 1))

        Y = zeros(1, length(xgrid.y.nodes), 1, 1, 1, 1)
        Y .= reshape(xgrid.y.nodes, (1, :, 1, 1, 1, 1))

        Z = zeros(1, 1, length(xgrid.z.nodes), 1, 1, 1)
        Z .= reshape(xgrid.z.nodes, (1, 1, :, 1, 1, 1))

        VX = zeros(1, 1, 1, length(vgrid.x.nodes), 1, 1)
        VX .= reshape(vgrid.x.nodes, (1, 1, 1, :, 1, 1))

        VY = zeros(1, 1, 1, 1, length(vgrid.y.nodes), 1)
        VY .= reshape(vgrid.y.nodes, (1, 1, 1, 1, :, 1))

        VZ = zeros(1, 1, 1, 1, 1, length(vgrid.z.nodes))
        VZ .= reshape(vgrid.z.nodes, (1, 1, 1, 1, 1, :))

        new{typeof(X), typeof(Y), typeof(Z), typeof(VX), typeof(VY), typeof(VZ)}(xgrid, vgrid, X, Y, Z, VX, VY, VZ)
    end
end

size(grid::Grid) = (size(grid.x)..., size(grid.v)...)

struct Species{G<:Grid, FFTPLANS}
    name::String
    grid::G
    v_grid::VGrid
    x_dims::Vector{Symbol}
    v_dims::Vector{Symbol}
    q::Float64
    m::Float64

    fft_plans::FFTPLANS
end

struct CollisionalMoments{uA, TA, νA}
    ux::uA
    uy::uA
    uz::uA
    T::TA
    ν::νA
end

struct SimulationMetadata{BA, PHI_L, PHI_R, PHI, SP, FFTPLANS, CM_DICT}
    x_dims::Vector{Symbol}
    x_grid::XGrid

    Bz::BA
    ϕ_left::PHI_L
    ϕ_right::PHI_R
    ϕ::PHI

    free_streaming::Bool

    ν_p::Float64
    collisional_moments::CM_DICT

    species::SP

    fft_plans::FFTPLANS
end

struct Simulation{SM<:SimulationMetadata, U}
    metadata::SM
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
    (; sim, λmax, buffer) = p
    vlasov_fokker_planck!(du, u, sim, λmax, buffer)
end

function vlasov_fokker_planck!(du, f, sim, λmax, buffer)
    λmax[] = 0.0

    @no_escape buffer begin
        @timeit "poisson" Ex, Ey, _ = poisson(sim, f, buffer)

        @timeit "collisional moments" collisional_moments!(sim, f, buffer)

        for i in eachindex(sim.species)
            α = sim.species[i]

            df = du.x[i]
            df .= 0

            if sim.free_streaming
                @timeit "free streaming" free_streaming!(df, f.x[i], α, buffer)
            end
            @timeit "electrostatic" electrostatic!(df, f.x[i], Ex, Ey, sim.Bz, α, buffer)
            @timeit "dfp" dfp!(df, f.x[i], α, sim, buffer)
        end
    end
end

function runsim_lightweight!(sim, T, Δt; diagnostic=nothing)
    set_default_buffer_size!(100_000_000)

    buffer = default_buffer()

    prog = Progress(Int(ceil(T / Δt)))
    t = 0.0
    λmax = Ref(0.0)
    u = sim.u
    if !isnothing(diagnostic)
        diagnostics = diagnostic.init()
    end
    while t < T
        stepdt = min(Δt, T-t)
        p = (; sim=sim.metadata, λmax, buffer)
        success, sf = ssp_rk43(vlasov_fokker_planck_step!, u, p, t, stepdt, 1.0, buffer)

        if success
            t += stepdt
            next!(prog)
        else
            error("Failed timestep")
        end

        if !isnothing(diagnostic)
            push!(diagnostics, diagnostic.run(sim, t))
        end
    end

    if !isnothing(diagnostic)
        return diagnostics
    end
end

function runsim!(sim, d, t_end; kwargs...)
    buffer = default_buffer()
    rk_step!(sim, t, dt) = begin
        λmax = Ref(0.0)
        p = (; sim=sim.metadata, λmax, buffer)
        ssp_rk43(vlasov_fokker_planck_step!, sim.u, p, t, dt, 1.0, buffer)
    end

    integrate_stably(rk_step!, sim, t_end, d; run_diagnostics=core_diagnostics, kwargs...)
end

function lightweight_diagnostics()
    init() = DataFrame(t=Float64[], electric_energy=Float64[])
    run(sim, t) = begin
        tup = core_diagnostics(sim, t)
        values(tup)
    end

    (; init, run=core_diagnostics)
end

