include("module.jl")

using Braginskii
using PDEHarness
using Bumper
using TimerOutputs

Bumper.reset_buffer!()

Ae = 1/25

d, sim = RTKineticReference.make_sim_vlasov(Val(:gpu); pt=1, Ae);
d = PDEHarness.normalize!(d)

dt_omega_p_tau = 0.02 / d["ωpτ"] * sqrt(Ae)
@show dt_omega_p_tau
dt_omega_c_tau = 0.02 / d["ωcτ"] * Ae
@show dt_omega_c_tau
dt = min(dt_omega_p_tau, dt_omega_c_tau)
#@show t_end = 6000 * dt_omega_p_tau
t_end = 2.4

TimerOutputs.reset_timer!()
Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=40*dt_omega_p_tau, log=true,
    snapshot_interval_dt=20*dt_omega_p_tau)
TimerOutputs.print_timer()

0.0
