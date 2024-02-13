include("module.jl")

using Braginskii
using PDEHarness
using Bumper
using TimerOutputs

Bumper.reset_buffer!()

Ae = 1/25

d, sim = SpectralBlockingDebugging.make_sim_vlasov(Val(:cpu));
d = PDEHarness.normalize!(d)

dt_omega_p_tau = 0.1 / d["ωpτ"] * sqrt(Ae)
@show dt_omega_p_tau
dt_omega_c_tau = 0.04 / d["ωcτ"] * Ae
@show dt_omega_c_tau
dt = min(dt_omega_p_tau, dt_omega_c_tau)
#@show t_end = 6000 * dt_omega_p_tau
t_end = 6.0

TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=0.2, log=true,
    snapshot_interval_dt=0.1)
TimerOutputs.print_timer()

0.0
