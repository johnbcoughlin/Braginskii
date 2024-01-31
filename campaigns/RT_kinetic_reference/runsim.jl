include("module.jl")

using Braginskii
using PDEHarness
using Bumper

Bumper.reset_buffer!()

d, sim = RTKineticReference.make_sim_vlasov(Val(:cpu); id=1, Ae=1/25);
τ_g = 1 / d["ωg"]
@show τ_g
d = PDEHarness.normalize!(d)
dt = 1e-2 / 25
t_end = dt * 100

Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=0.02, log=true,
    snapshot_interval_dt=0.01)

0.0
