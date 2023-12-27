include("module.jl")

using Braginskii
using TimerOutputs
using Cthulhu

TimerOutputs.reset_timer!()
d, sim = RayleighTaylor2D2V.make_sim(:gpu)
t_end = 0.05
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=t_end, initial_dt=0.001, log=false)
TimerOutputs.print_timer()
