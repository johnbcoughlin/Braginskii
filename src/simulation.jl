import PDEHarness.frame_writeout
import PDEHarness.load_from_frame!

using LoopVectorization: initialize_outer_reductions!
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

    νpτ::Float64
    ωpτ::Float64
    ωcτ::Float64
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
    x_dims, x_grid, species::Tuple, free_streaming, By, ϕl, ϕr, ν_p, ωpτ, ωcτ, gz,
    device, buffer)
    ϕ = alloc_zeros(Float64, buffer, size(x_grid)...)

    cms = collisional_moments(x_grid, [α.name for α in species], buffer)

    poisson_operator = CUDA.@allowscalar form_fourier_domain_poisson_operator(
        x_grid, x_dims, buffer)
    SimulationMetadata(
        x_dims, x_grid, By, ϕl, ϕr, ϕ, gz, free_streaming,
        ν_p, ωpτ, ωcτ, cms, species,
        plan_ffts(x_grid, buffer),
        plan_ffts(x_grid, allocator(:cpu)),
        factorize_poisson_operator(poisson_operator),
        device, buffer)
end

struct Simulation{SM<:SimulationMetadata, U}
    metadata::SM
    u::U
end

getproperty(sim::Simulation, sym::Symbol) = begin
    if sym ∈ (:u, :metadata)
        getfield(sim, sym)
    else
        getproperty(sim.metadata, sym)
    end
end

function vlasov_fokker_planck_step!(du, u, p, t)
    (; sim, λmax, buffer) = p
    vlasov_fokker_planck!(du, u, sim, λmax, buffer)
end

function vlasov_fokker_planck!(du, fs, sim, λmax, buffer)
    λmax[] = 0.0

    no_escape(buffer) do
        @timeit "poisson" E = poisson(sim, fs, buffer)
        #eliminate_curl!(E, sim, buffer)

        if sim.νpτ != 0.0
            @timeit "collisional moments" collisional_moments!(sim, fs, buffer)
        end

        for i in eachindex(sim.species)
            α = sim.species[i]

            df = du.x[i]
            df .= 0
            f = fs.x[i]

            λ = kinetic_rhs!(df, f, E, sim, α, buffer)
            λmax[] = max(λ, λmax[])
        end
    end
end

kinetic_rhs!(df, f, E, sim, α::Species{<:HermiteLaguerre}, buffer) = drift_kinetic_species_rhs!(df, f, E, sim, α, buffer)
kinetic_rhs!(df, f, E, sim, α::Union{Species{<:Hermite}, Species{<:WENO5}}, buffer) = vlasov_species_rhs!(df, f, E, sim, α, buffer)

function vlasov_species_rhs!(df, f, E, sim, α, buffer)
    λ_fs = λ_es = 0.0
    if sim.free_streaming
        @timeit "free streaming" λ_fs = free_streaming!(df, f, α, buffer)
    end
    if α.q != 0.0 || sim.gz != 0.0
        Ex, Ey, Ez = E
        @timeit "electrostatic" λ_es = electrostatic!(df, f, Ex, Ey, Ez, sim,
            α, buffer, sim.fft_plans)
    end
    if sim.νpτ != 0.0
        @timeit "dfp" dfp!(df, f, α, sim, buffer)
    end

    return 5 * (λ_es + λ_fs)
end

function drift_kinetic_species_rhs!(df, f, E, sim, α, buffer)
    @timeit "drifting" λ = drifting!(df, f, α, E, sim, buffer)
    @timeit "hyperdiffusion" apply_hyperdiffusion!(df, f, sim, α, buffer)
    return λ
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

filter_v!(f, species::Species{<:HermiteLaguerre}, buffer) = begin end

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
    snapshot_takers = make_snapshot_takers(sim, d; kwargs...)
    rk_step!(sim, t, dt) = begin
        no_escape(buffer) do
            λmax = Ref(0.0)
            p = (; sim=sim.metadata, λmax, buffer)
            filter!(sim.u, sim, buffer)
            success, safety_factor = ssp_rk43(vlasov_fokker_planck_step!, sim.u, p, t, dt, 1.0, buffer)
            success && take_snapshots(sim, snapshot_takers, t)
            return (success, safety_factor)
        end
    end

    @info "All set up and ready to integrate"
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

function frame_writeout(sim::Simulation, t)
    result = Dict{String, Any}("t" => t)
    fs = sim.u
    no_escape(sim.buffer) do
        result["ρ_c"] = charge_density(sim, fs, sim.buffer) |> hostarray
        Ex, Ey, Ez = poisson(sim, fs, sim.buffer)
        #result["ϕ"] = do_poisson_solve(sim.Δ_lu, result["ρ_c"], sim.x_grid, 
            #sim.ϕ_left, sim.ϕ_right, sim.x_grid.poisson_helper,
        #sim.x_dims, sim.fft_plans, sim.buffer) |> copy
        result["Ex"] = Ex |> hostarray
        #result["Ey"] = Ey |> copy
        result["Ez"] = Ez |> hostarray
    end
    for (i, α) in enumerate(sim.species)
        result[α.name] = Dict("f" => hostarray(sim.u.x[i]))
    end
    return result
end

function make_snapshot_takers(sim, d; snapshot_interval_dt=Inf, kwargs...)
    result = []
    if !isfinite(snapshot_interval_dt)
        return result
    end

    for i in 1:length(sim.species)
        α = sim.species[i]
        if isa(α.discretization.vdisc, HermiteLaguerre)
            continue
        end

        # Δx / vth
        halfwidth = min(0.5*snapshot_interval_dt, min_dx(sim.x_grid) / (1*average_vth(α.discretization)))

        # 3D arrays of the buffer type
        arraytype = typeof(alloc_array(Float64, sim.buffer, 1, 1, 1))

        latest_snap_index, latest_snap_t = existing_snapshots_count(d, α)

        snapshot_taker = SnapshotTaker(
            i,
            snapshot_interval_dt,
            halfwidth,
            arraytype,
            size(sim.x_grid),
            (args...) -> process_snapshot(args..., α, d),
            latest_snap_index + 1,
            latest_snap_t
        )
        push!(result, snapshot_taker)
    end
    #initialize_snapshot_file(sim, d)
    return result
end

function take_snapshots(sim, snapshot_takers, t)
    for st in snapshot_takers
        st(sim, t)
    end
end

function initialize_snapshot_file(sim, d)
    snapshot_file = joinpath(PDEHarness.mksimpath(d), "snapshots.jld2")
    jldopen(snapshot_file, "w") do file
        @info "initializing?"
        for α in sim.species
            file[α.name] = Dict{String, Any}()
        end
    end
end

function existing_snapshots_count(d, α::Species)
    snapshot_file = joinpath(PDEHarness.mksimpath(d), "snapshots.jld2")
    if isfile(snapshot_file)
        return jldopen(snapshot_file, "r") do file
            all_keys = collect(keys(file))
            r = Regex("snapshot_(\\d+)_$(α.name)")
            maxsnap = maximum(all_keys) do k
                m = match(r, k)
                return isnothing(m) ? 0 : parse(Int, m[1])
            end
            t = file["snapshot_$(maxsnap)_$(α.name)"]["t"]
            return maxsnap, t
        end
    else
        return 0, 0.0
    end
end

function process_snapshot(t, snapshot, snapshot_index, α::Species, d)
    snapshot_file = joinpath(PDEHarness.mksimpath(d), "snapshots.jld2")
    jldopen(snapshot_file, "a") do file
        @info "Writing snapshot" snapshot_index
        file["snapshot_$(snapshot_index)_$(α.name)"] = Dict{String, Any}(
            "t" => t,
            "n" => hostarray(snapshot.n),
            "u_x" => hostarray(snapshot.u_x),
            "u_y" => hostarray(snapshot.u_y),
            "u_z" => hostarray(snapshot.u_z),
            "T" => hostarray(snapshot.T),
            "q_x" => hostarray(snapshot.q_x),
            "q_y" => hostarray(snapshot.q_y),
            "q_z" => hostarray(snapshot.q_z),
        )
    end
end

function load_from_frame!(sim, frame)
    for i in 1:length(sim.species)
        α = sim.species[i]
        f = frame[α.name]
        sim.u.x[i] .= f
    end
end
