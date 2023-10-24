import Base.zero

struct SnapshotSpec
    """
    Snapshots will be taken centered every `interval_dt` in time.
    """
    interval_dt::Float64
end

struct SnapshotSample{A}
    # The heat flux
    q_x::A
    q_y::A
    q_z::A

    # The velocity
    u_x::A
    u_y::A
    u_z::A

    # The temperature
    T::A
end

struct Snapshot{A}
    center::Float64
    sample_sum::SnapshotSample{A}
    weight_sum::Float64
end

mutable struct SnapshotTaker{A, F}
    interval_dt::Float64

    last_called_at::Float64

    prior_snapshot::Snapshot{A}
    next_snapshot::Snapshot{A}

    process_snapshot::F
end

new_snapshot(::SnapshotTaker{A}, center::Float64) where A = Snapshot{A}(center, 
    zero(SnapshotSample{A}), 0.0)

(st::SnapshotTaker)(sim, t, simpath) = begin
    if t - st.last_called_at > st.interval_dt / 2
        error("The snapshot interval must be at least twice the interval between timesteps.")
    end

    # If we step over the next snapshot center, immediately process the
    # prior snapshot, and shift the snapshots up by one.
    if t > st.next_snapshot.center
        st.process_snapshot(st.prior_snapshot)

        st.prior_snapshot = st.next_snapshot
        st.next_snapshot = new_snapshot(st, st.prior_snapshot.center + st.interval_dt)
    end

    sample = snapshot_sample(sim)
    for snapshot in [st.prior_snapshot, st.next_snapshot]
        add_onto!(snapshot.sample_sum, sample)
        snapshot.weight_sum += abs(t - snapshot.center)
    end
end

function snapshot_sample(sim)

end

function data_callback(interval_dt)
end

function take_snapshot(sim, t, simpath)

end

function write_out_data(species, t, simpath)

end
