include("module.jl")

using Braginskii
using TimerOutputs

d, sim = HybridRayleighTaylor.make_sim_equilibrium_2d(Val(:gpu))
dt = 1e-2 / 2
t_end = 3.0
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=2.0, log=true
    )
TimerOutputs.print_timer()
