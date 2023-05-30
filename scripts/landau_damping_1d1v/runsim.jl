include("module.jl")

using Braginskii

d, sim = LandauDamping1D1V.make_sim()
t_end = 20.0
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=false,
    diagnostics_dt=t_end/400, log=false)
