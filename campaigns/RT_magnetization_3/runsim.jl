include("module.jl")

using Braginskii
using TimerOutputs
using PDEHarness
using Bumper

Bumper.reset_buffer!()

ωcτ = 1.0
Ae = 1/25
d, sim = RTMagnetization3.make_sim_vlasov(Val(:gpu); ωcτ=1.0, Ae)
d = PDEHarness.normalize!(d)
@show dt = 0.01 / ωcτ * Ae
t_end = 3*dt
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=100*dt, log=true,
    snapshot_interval_dt=30*dt
    )
TimerOutputs.print_timer()
