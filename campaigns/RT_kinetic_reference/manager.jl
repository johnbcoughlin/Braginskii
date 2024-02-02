module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")

function campaign_sims()
    return tuple.([1, 2, 3], [1/25, 1/50, 1/100]') |> vec
end

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    sims = campaign_sims
    for (id, ps) in enumerate(sims)
        pt, Ae = ps
        (; d) = RTKineticReference.make_sim_vlasov(Val(:gpu); pt, Ae)

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
    pt, Ae = campaign_sims()[id]
    (; d, sim) = RTKineticReference.make_sim_vlasov(Val(:gpu); pt, Ae)
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    τg = 1 / d["ωg"]
    @show t_end = 10.0 * τg
    dt = 5e-3 * Ae
    Braginskii.runsim!(sim, d, t_end, 
        restart_from_latest=true, 
        adaptive_dt=false,
        initial_dt=dt, 
        writeout_dt=0.1*τg, 
        snapshot_interval_dt=0.05*τg,
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
