include("module.jl")

using Braginskii
using TimerOutputs

TimerOutputs.reset_timer!()
d, sim = LandauDamping1D1V.make_sim(:gpu)
t_end = 2.0
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=0.1, initial_dt=0.001, log=false)
TimerOutputs.print_timer()
