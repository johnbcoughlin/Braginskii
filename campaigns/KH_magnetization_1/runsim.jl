include("module.jl")

using Braginskii
using TimerOutputs
using PDEHarness
using Bumper

Bumper.reset_buffer!()

ωcτ = 1.0
d, sim = KHMagnetization1.make_sim_hybrid(Val(:cpu); ωcτ=1.0)
d = PDEHarness.normalize!(d)
@show dt = 0.01 / ωcτ
t_end = 2.0*dt
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    initial_dt=dt, writeout_dt=100*dt, log=true,
    snapshot_interval_dt=30*dt
    )
TimerOutputs.print_timer()
