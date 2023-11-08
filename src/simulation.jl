
struct Species{DISC, FFTPLANS, Z_BCS}
    name::String
    x_dims::Vector{Symbol}
    v_dims::Vector{Symbol}
    q::Float64
    m::Float64

    fft_plans::FFTPLANS
    discretization::XVDiscretization{DISC}
    z_bcs::Z_BCS
end

struct CollisionalMoments{uA, TA, νA}
    ux::uA
    uy::uA
    uz::uA
    T::TA
    ν::νA
end

struct SimulationMetadata{BA, PHI_L, PHI_R, PHI, SP, FFTPLANS, CPUFFTPLANS, CM_DICT, POISSON_LU, BUF}
    x_dims::Vector{Symbol}
    x_grid::XGrid

    By::BA
    ϕ_left::PHI_L
    ϕ_right::PHI_R
    ϕ::PHI

    # The gravitational force in the z direction
    gz::Float64

    free_streaming::Bool

    ν_p::Float64
    collisional_moments::CM_DICT

    species::SP

    fft_plans::FFTPLANS
    cpu_fft_plans::CPUFFTPLANS

    Δ_lu::POISSON_LU

    device::Symbol
    buffer::BUF
end

function collisional_moments(xgrid, species, buffer)
    result = Dict{Tuple{String, String}, CollisionalMoments}()
    for α in species
        for β in species
            ux = alloc_zeros(Float64, buffer, size(xgrid)...)
            uy = alloc_zeros(Float64, buffer, size(xgrid)...)
            uz = alloc_zeros(Float64, buffer, size(xgrid)...)
            T = alloc_zeros(Float64, buffer, size(xgrid)...)
            ν = alloc_zeros(Float64, buffer, size(xgrid)...)
            cm = CollisionalMoments(ux, uy, uz, T, ν)
            push!(result, (α, β) => cm)
        end
    end
    result
end

function construct_sim_metadata(
    x_dims, x_grid, species::Tuple, free_streaming, By, ϕl, ϕr, ν_p, gz,
    device, buffer)
    ϕ = alloc_zeros(Float64, buffer, size(x_grid)...)

    cms = collisional_moments(x_grid, [α.name for α in species], buffer)

    SimulationMetadata(
        x_dims, x_grid, By, ϕl, ϕr, ϕ, gz, free_streaming,
        ν_p, cms, species,
        plan_ffts(x_grid, buffer),
        plan_ffts(x_grid, allocator(:cpu)),
        factorize_poisson_operator(form_fourier_domain_poisson_operator(
            ϕl, ϕr, x_grid, x_dims, buffer
        )),
        device, buffer)
end

struct Simulation{SM<:SimulationMetadata, U}
    metadata::SM
    u::U
end

getproperty(sim::Simulation, sym::Symbol) = begin
    if sym ∈ (:u, :metadata, :By)
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

    no_escape(buffer) do
        @timeit "poisson" Ex, Ey, Ez = poisson(sim, f, buffer)

        @timeit "collisional moments" collisional_moments!(sim, f, buffer)

        for i in eachindex(sim.species)
            α = sim.species[i]

            df = du.x[i]
            df .= 0

            if sim.free_streaming
                @timeit "free streaming" free_streaming!(df, f.x[i], α, buffer)
            end
            if α.q != 0.0 || sim.gz != 0.0
                @timeit "electrostatic" electrostatic!(df, f.x[i], Ex, Ey, Ez, sim.By, sim.gz, 
                    α, buffer, sim.fft_plans)
            end
            if sim.ν_p != 0.0
                @timeit "dfp" dfp!(df, f.x[i], α, sim, buffer)
            end
        end
    end
end

function filter!(f, sim, buffer)
    for i in eachindex(sim.species)
        α = sim.species[i]

        @timeit "filter" filter!(f.x[i], α, buffer)
    end
end

function filter!(f, species::Species, buffer)
    (; discretization) = species

    # Need to filter fourier coefficients first
    in_kxy_domain!(f, buffer, species.fft_plans) do modes
        σx, σy = discretization.x_grid.xy_hou_li_filters
        @. modes *= σx * σy'
    end

    filter_v!(f, species, buffer)
end

# No-op for WENO disc.
filter_v!(f, species::Species{WENO5}, buffer) = begin end

filter_v!(f, species::Species{<:Hermite}, buffer) = begin
    (; discretization) = species

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    σvx, σvy, σvz = discretization.vdisc.filters

    @. f *= σvx * σvy * σvz
end

function runsim_lightweight!(sim, T, Δt; diagnostic=nothing)
    buffer = sim.buffer

    prog = Progress(Int(ceil(T / Δt)))
    t = 0.0
    λmax = Ref(0.0)
    u = sim.u
    if !isnothing(diagnostic)
        diagnostics = diagnostic.init()
    end
    while t < T
        no_escape(buffer) do 
            stepdt = min(Δt, T-t)
            p = (; sim=sim.metadata, λmax, buffer)
            success, sf = ssp_rk43(vlasov_fokker_planck_step!, u, p, t, stepdt, 1.0, buffer)

            filter!(sim.u, sim, buffer)

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
    end

    if !isnothing(diagnostic)
        return diagnostics
    end
end

function runsim!(sim, d, t_end; kwargs...)
    buffer = sim.buffer
    rk_step!(sim, t, dt) = begin
        no_escape(buffer) do
            λmax = Ref(0.0)
            p = (; sim=sim.metadata, λmax, buffer)
            filter!(sim.u, sim, buffer)
            ssp_rk43(vlasov_fokker_planck_step!, sim.u, p, t, dt, 1.0, buffer)
        end
    end

    integrate_stably(rk_step!, sim, t_end, d; run_diagnostics=core_diagnostics, kwargs...)
end

function lightweight_diagnostics()
    init() = DataFrame(t=Float64[], electric_energy=Float64[], kinetic_energy_z=Float64[])
    run(sim, t) = begin
        tup = core_diagnostics(sim, t)
        values(tup)
    end

    (; init, run=core_diagnostics)
end

