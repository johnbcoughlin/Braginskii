struct LowRankVXVZSimulation{SIMMD, U}
    sim_metadata::SIMMD

    rank::Int

    # Organized as ((Ne, Ni), (Xe, Xi), (Se, Si), (Ze, Zi)
    u::U
end

struct LowRankSpecies{NArray, XArray, ZArray}
    N::NArray
    X::XArray
    S::Matrix{Float64}
    Z::ZArray
end

getproperty(sim::LowRankVXVZSimulation, sym::Symbol) = begin
    if sym ∈ (:rank, :sim_metadata)
        getfield(sim, sym)
    else
        getproperty(sim.sim_metadata, sym)
    end
end

# Performs the projection P_Φ
function P_Phi!(result, df, buffer)
    tmp = alloc_array(buffer, Float64, size(result, 1))
    tmp .= 0.5 .* (@view df[:, 1, 3]) .+ 0.5 .* (@view df[:, 3, 1])
    result[:, 1, 3] .= tmp
    result[:, 3, 1] .= tmp
    # Copy 1 and vx
    result[:, 1:2, 1] .= df[:, 1:2, 1]
    # Copy vz
    result[:, 1, 2] .= df[:, 1, 2]
end

# Performs the projection P_Φ^perp
function P_Phi_perp!(result, df, buffer)
    tmp = alloc_array(buffer, Float64, size(result, 1))
    tmp .= 0.5 .* (@view df[:, 1, 3]) .+ 0.5 .* (@view df[:, 3, 1])
    result .= df
    result[:, 1, 3] .-= tmp
    result[:, 3, 1] .-= tmp
    # Zero out 1 and vx
    result[:, 1:2, 1] .= 0.0
    # Copy vz
    result[:, 1, 2] .= 0.0
end

function form_fs!(sim)
    buffer = sim.buffer

    for i in 1:length(sim.species)
        
    end
end

function form_f(sim, N, X, S, Z, species, buffer)
    f = alloc_zeros(buffer, Float64, size(species.discretization)...)
    for i in 1:sim.rank
        for j in 1:sim.rank
            @. f += (@view X[:, i]) * S[i, j] * (@view Z[:, j])
        end
    end
    return f
end

function strang_splitting_step!(sim::LowRankVXVZSimulation)
    N = sim.u.x[1]

    # Step 1
    E = poisson(sim, N, buffer)

    # Step 2

end

