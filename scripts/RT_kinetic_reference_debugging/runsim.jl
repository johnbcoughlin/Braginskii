include("module.jl")

using Braginskii
using PDEHarness
using Bumper

Bumper.reset_buffer!()

Ae = 1/25
d, sim = RTKineticReferenceDebugging.make_sim_vlasov(Val(:cpu); ωpτ=20.0, ωcτ=1.0, Ae);
τ_g = 1 / d["ωg"]
@show τ_g
d = PDEHarness.normalize!(d)

dt_ωpτ = 0.1 / d["ωpτ"] * sqrt(Ae)
dt_ωcτ = 0.04 / d["ωcτ"] * Ae
dt = min(dt_ωpτ, dt_ωcτ)
@show dt

t_end = 40.0 / d["ωpτ"]
#t_end = 3dt

Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
    initial_dt=dt, writeout_dt=1 / d["ωpτ"], log=true)

0.0
