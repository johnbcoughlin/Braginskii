include("module.jl")

using Braginskii
using TimerOutputs

d, sim = ElectrostaticFlute2D2V.make_sim_equilibrium_2d(:cpu)
dt = 1e-3/2
t_end = 0.2
TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=4e-2, log=true,
    snapshot_interval_dt=0.01
    )
TimerOutputs.print_timer()
