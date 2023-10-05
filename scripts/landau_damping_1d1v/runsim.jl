include("module.jl")

using Braginskii
using TimerOutputs

TimerOutputs.reset_timer!()
d, sim = LandauDamping1D1V.make_sim(:cpu)
t_end = 40.0
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=0.1, log=false)
TimerOutputs.print_timer()
