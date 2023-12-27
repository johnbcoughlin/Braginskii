include("module.jl")

using Braginskii
using TimerOutputs

d, sim = HybridRayleighTaylor.make_sim_equilibrium_2d(:cpu)
dt = 1e-3
t_end = 1.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=0.5, log=true
    )
TimerOutputs.print_timer()
