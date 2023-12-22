include("module.jl")

using Braginskii
using TimerOutputs

d, sim = ElectrostaticFlute2D2V.make_sim(:cpu)
dt = 1e-1
t_end = 60.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    initial_dt=dt, writeout_dt=0.1, log=false)
TimerOutputs.print_timer()
