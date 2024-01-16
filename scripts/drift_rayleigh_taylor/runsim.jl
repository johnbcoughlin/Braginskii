include("module.jl")

using Braginskii
using TimerOutputs

d, sim = RayleighTaylorDriftKinetic.make_sim(:cpu)
t_end = 50.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    initial_dt=2e-2, writeout_dt=.1, log=false)
TimerOutputs.print_timer()
