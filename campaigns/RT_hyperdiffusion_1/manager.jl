module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")
include("../RT_shared/module.jl")

function campaign_sims()
    coefs = [0.4, 0.8, 1.6, 3.2, 6.4]
    return [(; grid_scale_hyperdiffusion_coef=c) for c in coefs]
end

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    sims = campaign_sims()
    for (id, params) in enumerate(sims)
        (; grid_scale_hyperdiffusion_coef) = params
        (; d) = RTHyperDiffusion1.make_sim_hybrid(Val(:gpu); grid_scale_hyperdiffusion_coef)

        set_simpath(id, d)
        prepare_sim_workdir(id, d)
    end

    template = read("slurm_template.sh", String)
    slurm = replace(template,
        "@@WORKDIR@@" => abspath("."),
        "@@NTASKS@@" => length(sims)
    )

    write("RT-array.slurm", slurm)
end

function prepare_sim_workdir(id, d)
    path = joinpath("sims", "RT-$id")
    mkpath(path)

    params_path = joinpath(path, "params.toml")
    !isfile(params_path) && open(params_path, "w") do io
        TOML.print(io, d, sorted=true) do x
            x isa Symbol && return string(x)
            x isa NamedTuple && return "NamedTuple"
            return x
        end
    end
end

function run_sim(; id)
    (; grid_scale_hyperdiffusion_coef) = campaign_sims()[id]
    (; d, sim) = RTHyperDiffusion1.make_sim_hybrid(Val(:gpu); grid_scale_hyperdiffusion_coef)
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    display(d)
    τg = 1 / d["ωg"]
    ωcτ = d["ωcτ"]
    @show t_end = 50.0 * τg
    dt_omega_c_tau = 0.01 / ωcτ
    @show dt_omega_c_tau
    dt = dt_omega_c_tau

    Braginskii.runsim!(sim, d, t_end, 
        restart_from_latest=true, 
        adaptive_dt=false,
        initial_dt=dt, 
        writeout_dt=1.0*τg, 
        snapshot_interval_dt=0.1*τg,
        log=true)
end

function set_simpath(id, d)
    d["##simpath##"] = abspath(joinpath("sims", "RT-$id"))
end

function cleanup()
    rm("RT-array.slurm", force=true)
    rm("sims", recursive=true, force=true)
    mkdir("sims")
    rm("shared", recursive=true, force=true)
    mkdir("shared")
end

end

