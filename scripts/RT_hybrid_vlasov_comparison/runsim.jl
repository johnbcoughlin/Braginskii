include("module.jl")

using Braginskii
using TimerOutputs

d, sim = RTHybridVlasovComparison.make_sim_vlasov(Val(:gpu))
dt = 1e-2 / 50
t_end = 2.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=2.0, log=true,
    snapshot_interval_dt=0.1
    )
TimerOutputs.print_timer()
