module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")

function campaign_sims()
    α = KH2.interface_width()
    rLis = [0.2, 0.4, 0.65, 1.0, 1.5] .* α
    etas = [0.6, 0.4, .27, 0.2, 0.2]
    sizes = [1, 2, 3]

    T_ref = KH2.reference_temp()
    vti = sqrt(T_ref)
    return [(; ωcτ=vti/rLis[i], eta=etas[i], sz=sizes[j]) for j in 1:3, i in 1:5] |> vec
end

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    sims = campaign_sims()
    for (id, params) in enumerate(sims)
        (; ωcτ, eta, sz) = params
        (; d) = KH2.make_sim_hybrid(Val(:gpu); ωcτ, eta, sz, just_setup=true)

        set_simpath(id, d)
        prepare_sim_workdir(id, d)
    end

    template = read("slurm_template.sh", String)
    slurm = replace(template,
        "@@WORKDIR@@" => abspath("."),
        "@@NTASKS@@" => length(sims)
    )

    write("KH-array.slurm", slurm)
end

function prepare_sim_workdir(id, d)
    magnetization = (id - 1) ÷ 3 + 1
    #sz = (id - 1) % 3 + 1
    path = joinpath("sims", "KH-$id")
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
    magnetization = (id - 1) ÷ 3 + 1
    #sz = (id - 1) % 3 + 1
    (; ωcτ, eta, sz) = campaign_sims()[id]
    (; d, sim) = KH2.make_sim_hybrid(Val(:gpu); ωcτ, eta, sz)
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    display(d)
    @show t_end = 400.0
    dt_omega_c_tau = magnetization == 5
        0.006 / ωcτ
    else if magnetization == 4
        0.008 / ωcτ
    else
        0.01 / ωcτ
    end
    @show dt_omega_c_tau
    dt = dt_omega_c_tau

    Braginskii.runsim!(sim, d, t_end, 
        restart_from_latest=true, 
        adaptive_dt=false,
        initial_dt=dt, 
        writeout_dt=2.0, 
        snapshot_interval_dt=0.5,
        log=true)
end

function set_simpath(id, d)
    d["##simpath##"] = abspath(joinpath("sims", "KH-$id"))
end

function cleanup()
    rm("KH-array.slurm", force=true)
    rm("sims", recursive=true, force=true)
    mkdir("sims")
    rm("shared", recursive=true, force=true)
    mkdir("shared")
end

end
