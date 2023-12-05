include("module.jl")

using Braginskii
using TimerOutputs

d, sim = Shock1D1V.make_sim(:cpu);
t_end = 0.4
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false,
    adaptive_dt=false,
    diagnostics_dt=0.1, 
    writeout_dt=0.1,
    snapshot_interval_dt=0.05, initial_dt=0.001/4, log=false)

