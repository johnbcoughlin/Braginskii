module CampaignManager

using DrWatson
@quickactivate :Braginskii
using PDEHarness
using TOML

include("module.jl")

function campaign_sims()
    return Vorticity.simulations()
end

"""
This function's job is to setup the working directories for every
simulation in the campaign.
"""
function setup_campaign()
    sims = campaign_sims()
    for (id, params) in enumerate(sims)
        (; ωcτ, γ, u_s_fac, u_V_fac, ζ) = params
        (; d) = Vorticity.make_sim_hybrid(Val(:gpu); ωcτ, γ, u_s_fac, u_V_fac, sz=4, ζ, just_setup=true)

        set_simpath(id, d)
        prepare_sim_workdir(id, d)
    end

    template = read("slurm_template.sh", String)
    slurm = replace(template,
        "@@WORKDIR@@" => abspath("."),
        "@@NTASKS@@" => length(sims)
    )

    write("SF-array.slurm", slurm)
end

function prepare_sim_workdir(id, d)
    path = joinpath("sims", "SF-$id")
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
    (; ωcτ, γ, u_s_fac, u_V_fac, ζ) = campaign_sims()[id]
    (; d) = Vorticity.make_sim_hybrid(Val(:gpu); ωcτ, γ, u_s_fac, u_V_fac, sz=4, ζ, just_setup=true)
    set_simpath(id, d)
    d = PDEHarness.normalize!(d)
    display(d)

    Nv = d["Nvx"]
    dx = d["Lx"] / d["Nx"]
    dz = d["Lz"] / d["Nz"]
    vt = sqrt(d["T_ref"])
    # vx and vz terms from cyclotron rotation
    lambda_B = ωcτ * Nv * 2
    # Free streaming terms
    lambda_x = vt * sqrt(Nv) / dx
    lambda_z = vt * sqrt(Nv) / dz

    # Use a CFL number of 0.7
    dt = 0.7 / sum(lambda_B + lambda_x + lambda_z)

    @show t_end = 250.0
    @show dt

    flush(stdout)

    Braginskii.runsim!(sim, d, t_end, 
        restart_from_latest=true, 
        adaptive_dt=false,
        initial_dt=dt, 
        writeout_dt=10.0, 
        snapshot_interval_dt=2.0,
        log=true)
end


function set_simpath(id, d)
    d["##simpath##"] = abspath(joinpath("sims", "SF-$id"))
end

function cleanup()
    rm("SF-array.slurm", force=true)
    rm("sims", recursive=true, force=true)
    mkdir("sims")
    rm("shared", recursive=true, force=true)
    mkdir("shared")
end

end
