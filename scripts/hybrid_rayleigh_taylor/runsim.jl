include("module.jl")

using Braginskii
using TimerOutputs

d, sim = HybridRayleighTaylor.make_sim_equilibrium_2d(Val(:gpu))
dt = 4e-3
t_end = 10.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=0.1, log=true
    )
TimerOutputs.print_timer()
