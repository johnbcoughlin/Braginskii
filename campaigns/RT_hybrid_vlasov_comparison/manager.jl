module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    (; d) = RTHybridVlasovComparison.make_sim_hybrid(Val(:gpu))
    set_simpath(1, d)
    prepare_sim_workdir(1, d)
    (; d) = RTHybridVlasovComparison.make_sim_vlasov(Val(:gpu))
    set_simpath(2, d)
    prepare_sim_workdir(2, d)

    template = read("slurm_template.sh", String)
    slurm = replace(template,
        "@@WORKDIR@@" => abspath("."),
        "@@NTASKS@@" => 2
    )

    write("RT-array.slurm", slurm)
end

#=
function get_or_save_hybrid_ic(id)
    if ispath("shared/f_ic_hybrid.jld2")
        file = load("shared/f_ic_hybrid.jld2")
        return file["fe"], file["fi"]
    end
    (; sim) = RayleighTaylor2D2V.make_sim(Kn=1.0, f_ic=nothing)
    save("shared/f_ic_hybrid.jld2", 
        Dict("fi" => Braginskii.hostarray(sim.u.x[1])))
    return sim.u.x[1]
end
=#

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
    hybrid = (id == 1)

    if hybrid
        (; d, sim) = RTHybridVlasovComparison.make_sim_hybrid(Val(:gpu))
    else
        (; d, sim) = RTHybridVlasovComparison.make_sim_vlasov(Val(:gpu))
    end
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    t_end = 250.0
    dt = hybrid ? 5e-3 : (5e-3 * d["Ae"])
    Braginskii.runsim!(sim, d, t_end, restart_from_latest=false, adaptive_dt=true,
        initial_dt=dt, writeout_dt=2.0, log=true, snapshot_interval_dt=0.5)
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
