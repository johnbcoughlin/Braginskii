include("module.jl")

using Braginskii
using TimerOutputs

d, sim = ElectrostaticFlute2D2V.make_sim(:cpu)
dt = 3e-3
t_end = 1.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=1e-2, log=true)
TimerOutputs.print_timer()
