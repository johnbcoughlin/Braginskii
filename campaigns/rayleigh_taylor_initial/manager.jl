module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")

knudsens() = 10.0 .^ (-2.4:0.2:-1)

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    Kns = knudsens()

    f_ic = get_or_save_ic()

    for (id, Kn) in enumerate(Kns)
        (; d) = RayleighTaylor2D2V.make_sim(; Kn, f_ic)
        set_simpath(id, d)
        prepare_sim_workdir(id, d)
    end

    template = read("slurm_template.sh", String)
    slurm = replace(template,
        "@@WORKDIR@@" => abspath("."),
        "@@NTASKS@@" => length(Kns)
    )

    write("RT-array.slurm", slurm)
end

function get_or_save_ic()
    if ispath("shared/f_ic.jld2")
        return load("shared/f_ic.jld2")["f"]
    end
    (; sim) = RayleighTaylor2D2V.make_sim(Kn=1.0, f_ic=nothing)
    save("shared/f_ic.jld2", Dict("f" => Braginskii.hostarray(sim.u.x[1])))
    return sim.u.x[1]
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
    Kns = knudsens()
    Kn = Kns[id]

    f_ic = get_or_save_ic()
    (; d, sim, tau, dt) = RayleighTaylor2D2V.make_sim(; Kn, f_ic)
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    t_end = tau * 10.0
    nframes = 50
    @show nsnapshots = min(2000, t_end/(.1))
    Braginskii.runsim!(
        sim, d, t_end, 
        restart_from_latest=true, 
        adaptive_dt=false,
        diagnostics_dt=(t_end / 100),
        writeout_dt=(t_end / nframes),
        snapshot_interval_dt=(t_end / nsnapshots),
        initial_dt=dt,
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
