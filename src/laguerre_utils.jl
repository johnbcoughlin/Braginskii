export bigfloat_weighted_laguerre_expansion, expand_bigfloat_laguerre_f

# All Laguerre polynomials here are unnormalized, from the family α = 0.

function L_up_to_n(Mμ, μ::AbstractVector) 
    result = zeros(BigFloat, Mμ+1, length(μ))
    L = zeros(BigFloat, Mμ+1)
    for i in 1:length(μ)
        result[:, i] .= L_up_to_n!(L, Mμ, big(μ[i]))
    end
    result
end

function L_up_to_n!(L, n, μ::BigFloat)
    @assert n >= 2
    @assert length(L) == n+1
    α = 0.0
    # k = 0
    L[1] = 1.0
    # k = 1
    L[2] = 1 + α - μ
    for k in 1:n-1
        L[k+1+1] = ((2k + 1 + α - μ) * L[k+1] - (k + α) * L[k-1+1]) / (k+1)
    end
    L
end

function bigfloat_weighted_laguerre_expansion(f::Function, Mμ::Int, Mvy::Int, X, Y, Z, μ0, vth)
    μ_nodes, μ_w = FastGaussQuadrature.gausslegendre(Mμ == 0 ? 1 : 250)
    vy_nodes, vy_w = FastGaussQuadrature.gausslegendre(Mvy == 0 ? 1 : 250)

    # Shift to [0, 2]
    μ_nodes .+= 1.0
    # Expand to [0, 50]
    μ_nodes .*= 25.0
    μ_w .*= 25.0

    μ_w = Mμ == 0 ? [1.0] : μ_w 
    vy_w = Mvy == 0 ? [1.0] : vy_w 

    unnormalized_L_n_μ_vand = L_up_to_n(Mμ, μ_nodes)
    normalized_He_n_vy_vand = He_up_to_n(Mvy, vy_nodes; normalized=true)

    #mass = unnormalized_L_n_μ_vand * Diagonal(μ_w) * unnormalized_L_n_μ_vand'
    #@assert isapprox(mass, diagm(diag(mass)), rtol=1e-10)

    μ_nodes = reshape(μ_nodes, (1, 1, 1, :, 1))
    vy_nodes = reshape(vy_nodes, (1, 1, 1, 1, :))

    X_array = Array(X)
    Y_array = Array(Y)
    Z_array = Array(Z)
    fxv = @. f(X_array, Y_array, Z_array, μ_nodes * μ0, vy_nodes * vth)

    result = zeros(size(fxv)[1:3]..., Mμ+1, Mvy+1)

    w_vand_μ = Float64.(unnormalized_L_n_μ_vand .* μ_w')
    w_vand_vy = Float64.(normalized_He_n_vy_vand .* vy_w')

    for λxyz in CartesianIndices((length(X), length(Y), length(Z)))
        for αμ in 1:length(μ_w), βμ in 1:Mμ+1
            for αvy in 1:length(vy_w), βvy in 1:Mvy+1
                w_vand = w_vand_μ[βμ, αμ] * w_vand_vy[βvy, αvy]
                result[λxyz, βμ, βvy] += fxv[λxyz, αμ, αvy] * w_vand
            end
        end
    end
    arraytype(X)(result)
end

function expand_bigfloat_laguerre_f(coefs::AbstractArray{Float64, 5}, vgrid::GyroVGrid, μ0, vth)
    α = 0.0
    Nx, Ny, Nz, Nμ, Nvy = size(coefs)
    Mμ = Nμ - 1
    Mvy = Nvy - 1

    μ_nodes = vgrid.μ.nodes
    vy_nodes = vgrid.y.nodes

    Kμ = length(μ_nodes)
    Kvy = length(vy_nodes)

    ndims = sum([Mμ > 0, Mvy > 0])

    (; Vμ, VY) = vgrid
    exp_weight = @. exp(-(VY^2) / 2vth^2 - Vμ / μ0) # Figure out normalization
    exp_weight = reshape(exp_weight, (Kμ, Kvy))

    μ_vand = L_up_to_n(Mμ, μ_nodes / μ0)
    norm = @. gamma((0:Mμ) + α + 1) / factorial(big(0:Mμ))
    μ_vand ./= norm
    vy_vand = He_up_to_n(Mvy, vy_nodes/vth; normalized=true) .|> Float64

    result = zeros(Nx, Ny, Nz, length(μ_nodes), length(vy_nodes))

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for αμ in eachindex(μ_nodes), βμ in 1:Mμ+1
            for αvy in eachindex(vy_nodes), βvy in 1:Mvy+1
                vand = μ_vand[βμ, αμ] * vy_vand[βvy, αvy]
                result[λxyz, αμ, αvy] += coefs[λxyz, βμ, βvy] * vand * exp_weight[αμ, αvy]
            end
        end
    end
    result
end
