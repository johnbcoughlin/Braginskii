include("module.jl")

using Braginskii
using TimerOutputs
using PDEHarness

d, sim = LandauDamping1D1V.make_sim_vlasov(Val(:cpu))
d = PDEHarness.normalize!(d)
dt = 1e-2/10
t_end = 20.0
TimerOutputs.reset_timer!()
Braginskii.runsim_lightweight!(sim, d, t_end, adaptive_dt=true,
    initial_dt=dt, writeout_dt=1.0, log=true,
    snapshot_interval_dt=0.2)
TimerOutputs.print_timer()
