include("module.jl")

using Braginskii
using TimerOutputs

TimerOutputs.reset_timer!()
d, sim = LandauDamping1D1V.make_sim(:cpu)
t_end = 50.0
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=0.1, snapshot_interval_dt = 1.0, initial_dt=0.004, log=false)
TimerOutputs.print_timer()
