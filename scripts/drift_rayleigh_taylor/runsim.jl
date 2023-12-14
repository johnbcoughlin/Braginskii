include("module.jl")

using Braginskii
using TimerOutputs

d, sim = RayleighTaylorDriftKinetic.make_sim(:cpu)
t_end = 0.2
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    initial_dt=1e-3, log=false)
TimerOutputs.print_timer()
