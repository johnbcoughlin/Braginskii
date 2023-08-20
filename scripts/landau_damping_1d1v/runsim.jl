include("module.jl")

using Braginskii
using TimerOutputs

TimerOutputs.reset_timer!()
d, sim = LandauDamping1D1V.make_sim(:gpu)
t_end = 0.5
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=t_end, log=false)
TimerOutputs.print_timer()
