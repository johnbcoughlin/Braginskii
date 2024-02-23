import Base.zero

struct SnapshotSpec
    """
    Snapshots will be taken centered every `interval_dt` in time.
    """
    interval_dt::Float64
end

struct SnapshotSample{A}
    # Number density
    n::A

    # The velocity
    u_x::A
    u_y::A
    u_z::A

    # The isotropic temperature
    T::A

    # Deviatoric part of pressure tensor
    Pi_xx::A
    Pi_yy::A
    Pi_zz::A
    Pi_xy::A
    Pi_xz::A
    Pi_yz::A

    # The heat flux
    q_x::A
    q_y::A
    q_z::A
end

empty_snapshot_sample(::Type{A}, sz) where A = SnapshotSample(
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...),
    alloc_zeros(Float64, A, sz...)
)

mutable struct Snapshot{A}
    center::Float64
    sample_sum::SnapshotSample{A}
    weight_sum::Float64
end

mutable struct SnapshotTaker{A, F}
    # Different species take snapshot samples at different intervals.
    species_index::Int64

    # The interval between centers of successive snapshots
    interval_dt::Float64

    # The width of the sample window of each snapshot
    halfwidth::Float64

    last_called_at::Union{Nothing, Float64}

    prior_snapshot::Snapshot{A}
    next_snapshot::Snapshot{A}

    process_snapshot::F

    snapshot_index::Int64
end

SnapshotTaker(species_index, interval_dt, halfwidth, arraytype::Type{A}, 
    sz, process_snapshot::F, starting_index, latest_snap_t) where {A, F} = begin
    SnapshotTaker{A, F}(
        species_index,
        interval_dt,
        halfwidth,
        nothing,
        new_snapshot(arraytype, sz, latest_snap_t),
        new_snapshot(arraytype, sz, latest_snap_t + interval_dt),
        process_snapshot,
        starting_index
    )
end

new_snapshot(::Type{A}, sz, center::Float64) where A = Snapshot{A}(center, 
    empty_snapshot_sample(A, sz), 0.0)

new_snapshot(s::Snapshot{A}, center) where {A} = new_snapshot(A, size(s.sample_sum.T), center)

(st::SnapshotTaker)(sim, t) = begin
    @assert st.halfwidth <= st.interval_dt

    if isnothing(st.last_called_at)
        st.last_called_at = t
    elseif t - st.last_called_at > st.interval_dt / 2
        error("The snapshot interval must be at least twice the interval between timesteps.")
    end
    st.last_called_at = t

    # If we step past the end of the prior snapshot, immediately process the
    # prior snapshot, and shift the snapshots up by one.
    if t > st.prior_snapshot.center + st.halfwidth
        st.process_snapshot(st.prior_snapshot.center, 
            average_out(st.prior_snapshot), st.snapshot_index)
        st.snapshot_index += 1

        st.prior_snapshot = st.next_snapshot
        st.next_snapshot = new_snapshot(st.prior_snapshot, st.prior_snapshot.center + st.interval_dt)
    end

    #@info "" st.prior_snapshot.sample_sum.u_y

    prior_start = st.prior_snapshot.center - st.halfwidth
    prior_end = st.prior_snapshot.center + st.halfwidth
    next_start = st.next_snapshot.center - st.halfwidth
    next_end = st.next_snapshot.center + st.halfwidth

    if (prior_start <= t <= prior_end) || (next_start <= t <= next_end)
        sample = snapshot_sample(sim, st.species_index)
        for snapshot in [st.prior_snapshot, st.next_snapshot]
            weight = max(0.0, st.halfwidth - abs(t - snapshot.center))
            if (weight != 0.0)
                add_onto!(snapshot.sample_sum, sample, weight)
                snapshot.weight_sum += weight
            end
        end
    end
end

function add_onto!(sample_sum::SnapshotSample, sample::SnapshotSample, weight)
    @. sample_sum.n += sample.n * weight
    @. sample_sum.u_x += sample.u_x * weight
    @. sample_sum.u_y += sample.u_y * weight
    @. sample_sum.u_z += sample.u_z * weight
    @. sample_sum.T += sample.T * weight
    @. sample_sum.Pi_xx += sample.Pi_xx * weight
    @. sample_sum.Pi_yy += sample.Pi_yy * weight
    @. sample_sum.Pi_zz += sample.Pi_zz * weight
    @. sample_sum.Pi_xy += sample.Pi_xy * weight
    @. sample_sum.Pi_xz += sample.Pi_xz * weight
    @. sample_sum.Pi_yz += sample.Pi_yz * weight
    @. sample_sum.q_x += sample.q_x * weight
    @. sample_sum.q_y += sample.q_y * weight
    @. sample_sum.q_z += sample.q_z * weight
end

function average_out(snapshot::Snapshot)
    return SnapshotSample(
        snapshot.sample_sum.n / snapshot.weight_sum,
        snapshot.sample_sum.u_x / snapshot.weight_sum,
        snapshot.sample_sum.u_y / snapshot.weight_sum,
        snapshot.sample_sum.u_z / snapshot.weight_sum,
        snapshot.sample_sum.T / snapshot.weight_sum,
        snapshot.sample_sum.Pi_xx / snapshot.weight_sum,
        snapshot.sample_sum.Pi_yy / snapshot.weight_sum,
        snapshot.sample_sum.Pi_zz / snapshot.weight_sum,
        snapshot.sample_sum.Pi_xy / snapshot.weight_sum,
        snapshot.sample_sum.Pi_xz / snapshot.weight_sum,
        snapshot.sample_sum.Pi_yz / snapshot.weight_sum,
        snapshot.sample_sum.q_x / snapshot.weight_sum,
        snapshot.sample_sum.q_y / snapshot.weight_sum,
        snapshot.sample_sum.q_z / snapshot.weight_sum
    )
end

function snapshot_sample(sim, species_index)
    α = sim.species[species_index]
    n, u_x, u_y, u_z, T, Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz, q_x, q_y, q_z = moments_for_wsindy(
        sim.u.x[species_index],
        α.discretization,
        α.v_dims,
        sim.buffer)
    return SnapshotSample(n, u_x, u_y, u_z, T, 
        Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz,
        q_x, q_y, q_z)
end
