struct LowRankVXVZSimulation{SIMMD, U}
    sim_metadata::SIMMD

    rank::Int

    # Organized as (fe, (Ni), (Xi), (Si), (Zi)
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

P_Phi(df::ArrayPartition, buffer) = begin
    ArrayPartition(map(df.x) do df_i
        P_Phi(df_i, buffer)
    end)
end

# Performs the projection P_Φ
P_Phi(df, buffer) = begin
    result = alloc_zeros(Float64, buffer, 3, 3, size(df, 3))
    P_Phi!(result, df, buffer)
    result
end

P_Phi!(P_df::ArrayPartition, df::ArrayPartition, buffer) = begin
    for i in eachindex(P_df.x)
        P_Phi!(P_df.x[i], df.x[i], buffer)
    end
end

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
P_Phi_perp(df::ArrayPartition, buffer) = begin
    ArrayPartition(map(df.x) do df_i
        P_Phi_perp(df_i, buffer)
    end)
end

P_Phi_perp(df, buffer) = begin
    result = alloc_zeros(Float64, buffer, size(df)...)
    P_Phi_perp!(result, df, buffer)
    result
end

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

function form_K(X, S, buffer)
    K = alloc_array(Float64, buffer, size(X)...)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, X, S, 0.0, K)
    K
end
function form_L(S, Z, buffer)
    L = alloc_array(Float64, buffer, size(Z)...)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, Z, S, 0.0, L)
    L
end

function form_g(sim, X, S, Z, buffer)
    Nvx = size(X, 1)
    Nvz = size(Z, 1)

    G = alloc_zeros(Float64, buffer, Nvx, Nvz, sim.NX)
    K = alloc_zeros(Float64, buffer, Nvx, sim.rank, sim.NX)

    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, X, S, 0.0, K)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, K, Z, 0.0, G)

    return G
end

function form_g_K(sim, K, Z, species, buffer)
    Nvx = size(K, 1)
    Nvz = size(Z, 1)
    G = alloc_zeros(Float64, buffer, Nvx, Nvz, sim.NX)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, K, Z, 0.0, G)
    return G
end

function form_g_L(sim, X, L, species, buffer)
    Nvx = size(X, 1)
    Nvz = size(L, 1)
    G = alloc_zeros(Float64, buffer, Nvx, Nvz, sim.NX)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', 1.0, X, L, 0.0, G)
    return G
end

function form_f(sim, N, G, species, buffer)
    G[1:3, 1:3, :] .+= N
    f = alloc_array(Float64, buffer, sim.NX, Nvx, Nvz)
    permutedims!(f, F, (3, 1, 2))
    return f
end

function form_fs(sim, N, buffer, g_op::Function)
    fs = []
    for i in eachindex(sim.species)
        G = g_op(i)
        f = form_f(sim, N.x[i], G, buffer)
        push!(fs, f)
    end
    return ArrayPartition(tuple(fs...))
end

function form_fs(sim, N, X, S, Z, buffer)
    form_fs(sim, N, buffer) do i
        form_g(sim, X.x[i], S.x[i], Z.x[i], buffer)
    end
end

function form_fs_K(sim, N, K, Z, buffer)
    form_fs(sim, N, buffer) do i
        form_g_K(sim, K.x[i], Z.x[i], buffer)
    end
end

function form_fs_L(sim, N, X, L, buffer)
    form_fs(sim, N, buffer) do i
        form_g_L(sim, X.x[i], L.x[i], buffer)
    end
end

# Permutes dfs and fs 
function lowrank_vlasov_fokker_planck!(fi, E, sim, buffer)
    NX = sim.NX

    # fi comes in as (Nvx, Nvz, NX)
    Nvx, Nvz, _ = size(fi)
    sz = size(sim.x_grid..., Nvx, 1, Nvz)
    f_permuted = alloc_array(Float64, buffer, NX, Nvx, Nvz)
    permutedims!(f_permuted, fi, (3, 1, 2))
    fi = reshape(f_permuted, (size(sim.x_grid)..., Nvx, 1, Nvz))
    dfs = alloc_vec(buffer, fs)

    vlasov_fokker_planck!(dfs, fs, E, sim.sim_metadata, Ref(0.0), buffer)

    dfs = ArrayPartition((map(dfs.x) do df
        Nx, Ny, Nz, Nvx, _, Nvz = size(df)
        df_reshaped = reshape(df, (NX, Nvx, 1, Nvz))
        df_permuted = alloc_array(Float64, buffer, Nvx, Nvz, NX)
        permutedims!(df_permuted, df_reshaped, (2, 3, 1))
    end)...)
    return dfs
end

function lie_splitting_step!(sim::LowRankVXVZSimulation, dt)
    buffer = sim.buffer
    Nⁿ, Xⁿ, Sⁿ, Zⁿ = sim.u.x

    Eⁿ = poisson(sim, Nⁿ, buffer)

    Nⁿ⁺¹ = forward_euler(Nⁿ, dt, buffer) do dN, N
        fs = form_fs(sim, N, Xⁿ, Sⁿ, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, Eⁿ, sim, buffer)
        P_Phi!(dN, dfs, buffer)
    end

    Kⁿ = form_K(Xⁿ, Sⁿ, buffer)
    Kⁿ⁺¹ = forward_euler(Kⁿ, dt, buffer) do dKs, K
        fs = form_fs_K(sim, Nⁿ, K, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, Eⁿ, sim, buffer)
        dgs = P_Phi_perp!(dfs, buffer)
        Z_projections!(dKs, dgs, Zⁿ)
    end
    Xⁿ⁺¹, S1 = cuda_batch_qr(Kⁿ⁺¹, buffer)

    S2 = forward_euler(S^n, dt, buffer) do dSs, S
        fs = form_fs(sim, Nⁿ, Xⁿ, S, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, Eⁿ, sim, buffer)
        dgs = P_Phi_perp(dfs, buffer)
        XZ_projections!(dSs, dgs, Xⁿ⁺¹, Zⁿ, buffer)
    end

    Lⁿ = form_L(S2, Zⁿ, buffer)
    Lⁿ⁺¹ = forward_euler(Lⁿ, dt, buffer) do dLs, L
        fs = form_fs_L(sim, Nⁿ, Xⁿ⁺¹, L, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, Eⁿ, sim, buffer)
        dgs = P_Phi_perp(dfs, buffer)
        X_projections!(dLs, dgs, Xⁿ⁺¹)
    end
    Zⁿ⁺¹, Sⁿ⁺¹ = cuda_batch_qr(Lⁿ⁺¹, buffer)

    sim.u.x[1] .= Nⁿ⁺¹
    sim.u.x[2] .= Xⁿ⁺¹
    sim.u.x[3] .= Sⁿ⁺¹
    sim.u.x[4] .= Zⁿ⁺¹
end

function strang_splitting_step!(sim::LowRankVXVZSimulation, dt)
    buffer = sim.buffer

    Nⁿ, Xⁿ, Sⁿ, Zⁿ = sim.u.x
    dfs = form_fs(sim, sim.u.x..., buffer)

    # Step 1
    Eⁿ = poisson(sim, Nⁿ, buffer)

    # Step 2
    N_½ = forward_euler(Nⁿ, dt/2, buffer) do dN, N
        fs = form_fs(sim, N, Xⁿ, Sⁿ, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, Eⁿ, sim, buffer)
        P_Phi!(dN, dfs, buffer)
    end

    # Step 3
    E_½ = poisson(sim, N_½, buffer)

    # Step 4
    Kⁿ = form_K(Xⁿ, Sⁿ, buffer)
    K_½ = ssprk2(Kⁿ, dt/2, buffer) do dKs, K
        fs = form_fs_K(sim, N_½, K, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, E_½, sim, buffer)
        dgs = P_Phi_perp(dfs, buffer)
        Z_projections!(dKs, dgs, Zⁿ)
    end
    X_½, S1 = cuda_batch_qr(K_½, buffer)

    S_2 = ssprk2(S1, dt/2, buffer) do dSs, S
        fs = form_fs(sim, N_½, X_½, S, Zⁿ, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, E_½, sim, buffer)
        dgs = P_Phi_perp(dfs, buffer)
        XZ_projections!(dSs, dgs, X_½, Zⁿ, buffer)
    end

    Lⁿ = form_L(S_2, Zⁿ, buffer)
    L_½ = ssprk2(Lⁿ, dt/2, buffer) do dLs, L
        fs = form_fs_L(sim, N_½, X_½, L, buffer)
        dfs = lowrank_vlasov_fokker_planck!(fs, E_½, sim, buffer)
        dgs = P_Phi_perp(dfs, buffer)
        X_projections!(dLs, dgs, X_½)
    end
    error("not implemented")

    sim.u.x[1] .= Nⁿ⁺¹
    sim.u.x[2] .= Xⁿ⁺¹
    sim.u.x[3] .= Sⁿ⁺¹
    sim.u.x[4] .= Zⁿ⁺¹
end

