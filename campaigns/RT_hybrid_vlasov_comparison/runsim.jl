include("module.jl")

using Braginskii
using TimerOutputs
using PDEHarness

d, sim = RTHybridVlasovComparison.make_sim_vlasov(Val(:gpu))
d = PDEHarness.normalize!(d)
dt = 1e-2 / 100
t_end = 2.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=0.02, log=true,
    snapshot_interval_dt=0.01
    )
TimerOutputs.print_timer()
