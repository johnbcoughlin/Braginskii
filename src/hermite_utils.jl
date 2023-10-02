export bigfloat_weighted_hermite_expansion, expand_bigfloat_hermite_f

"""
Compute the nth-order normalized probabilist's Hermite polynomial.
"""
function He_n(n, x; normalized=true)
    if normalized
        basis(SpecialPolynomials.ChebyshevHermite, n)(big(x)) / sqrt(factorial(big(n)))
    else
        basis(SpecialPolynomials.ChebyshevHermite, n)(big(x))
    end
end

function He_up_to_n(n::Int, x::AbstractVector{Float64}; normalized=true)
    N = 0:n
    inv_sq_facs = 1 ./ sqrt.(factorial.(big.(N)))
    result = zeros(BigFloat, n+1, length(x))
    He = zeros(BigFloat, n+1)
    for i in 1:length(x)
        if normalized
            result[:, i] .= unnormalized_He_n_up_to!(He, n, big(x[i])) .* inv_sq_facs
        else
            result[:, i] .= unnormalized_He_n_up_to!(He, n, big(x[i]))
        end
    end
    result
end

function unnormalized_He_n_up_to!(He, n::Int, x::BigFloat)
    @assert n >= 0
    @assert length(He) == n+1
    He[1] = 1.0 
    n == 0 && return He
    He[2] = x
    for i in 2:n
        He[i+1] = x * He[i] - (i-1) * He[i-1]
    end
    He
end

"""
Computes the weighted Hermite expansion of the function f(v)
M = maximum order of approximation
"""
function bigfloat_weighted_hermite_expansion(f::Function, M::Int, v₀; normalized=true)
    nodes, w = FastGaussQuadrature.gausshermite(M+1)
    result = sqrt(2) * big.(w)' * (He_up_to_n(M, nodes*sqrt(2)/v₀; normalized=true)' .* f.(nodes*sqrt(2)) .* exp.(big.(nodes.^2)))
    
    @timeit "unnormalizing" for i in 0:M
        fi = result[i+1]
        if !normalized
            result[i+1] /= sqrt(factorial(big(i)))
        end
    end
    vec(result)
end

"""
Same as the above, but uses matrix operations to reuse BigFloat calculations.
"""
function bigfloat_weighted_hermite_expansion(f::Function, M::Int, x::AbstractVector, v₀; normalized=true)
    nodes, w = FastGaussQuadrature.unweightedgausshermite(M+1)
    w = w .* exp.(-big.(nodes).^2)
    fxv = f.(x, sqrt(2)*nodes')
    @timeit "fxv_exp" fxv_exp = fxv .* exp.(big.(nodes.^2))'
    normalized_He_n_vand = He_up_to_n(M, sqrt(2)*nodes/v₀; normalized=true)
    @timeit "diagw" mat = (fxv_exp * diagm(big.(w)) * normalized_He_n_vand')
    return sqrt(2) * mat
end

function bigfloat_weighted_hermite_expansion(f::Function, Mvx::Int, Mvy::Int, Mvz::Int, X, Y, Z, v₀)
    vx_nodes, vx_w = FastGaussQuadrature.unweightedgausshermite(Mvx+1)
    vy_nodes, vy_w = FastGaussQuadrature.unweightedgausshermite(Mvy+1)
    vz_nodes, vz_w = FastGaussQuadrature.unweightedgausshermite(Mvz+1)

    vx_w = vx_w .* exp.(-big.(vx_nodes).^2)
    vy_w = vy_w .* exp.(-big.(vy_nodes).^2)
    vz_w = vz_w .* exp.(-big.(vz_nodes).^2)

    normalized_He_n_vx_vand = He_up_to_n(Mvx, sqrt(2) * vx_nodes/v₀; normalized=true)
    normalized_He_n_vy_vand = He_up_to_n(Mvy, sqrt(2) * vy_nodes/v₀; normalized=true)
    normalized_He_n_vz_vand = He_up_to_n(Mvz, sqrt(2) * vz_nodes/v₀; normalized=true)

    vx_nodes = reshape(vx_nodes, (1, 1, 1, :, 1, 1))
    vy_nodes = reshape(vy_nodes, (1, 1, 1, 1, :, 1))
    vz_nodes = reshape(vz_nodes, (1, 1, 1, 1, 1, :))

    fxv = @. f(X, Y, Z, sqrt(2) * vx_nodes, sqrt(2) * vy_nodes, sqrt(2) * vz_nodes)
    fxv_exp = @. fxv * exp(big(vx_nodes^2 + vy_nodes^2 + vz_nodes^2))
    fxv_exp = Float64.(fxv_exp)

    result = zeros(size(fxv))

    w_vand_vx = Float64.(normalized_He_n_vx_vand .* vx_w')
    w_vand_vy = Float64.(normalized_He_n_vy_vand .* vy_w')
    w_vand_vz = Float64.(normalized_He_n_vz_vand .* vz_w')

    @turbo for λxyz in CartesianIndices((length(X), length(Y), length(Z)))
        for αvx in 1:Mvx+1, βvx in 1:Mvx+1
            for αvy in 1:Mvy+1, βvy in 1:Mvy+1
                for αvz in 1:Mvz+1, βvz in 1:Mvz+1
                    w_vand = w_vand_vx[βvx, αvx] * w_vand_vy[βvy, αvy] * w_vand_vz[βvz, αvz]
                    result[λxyz, βvx, βvy, βvz] += fxv_exp[λxyz, αvx, αvy, αvz] * w_vand
                end
            end
        end
    end
    sqrt(2^3) * result
end

"""
Expand the given vector of coefficients in the weighted Hermite basis.
"""
function expand_bigfloat_hermite_f(coefs::AbstractVector, v::AbstractVector, v₀; normalized=true)
    M = length(coefs)-1
    result = vec(coefs' * (He_up_to_n(M, v/v₀; normalized) .* exp.(-(v/v₀).^2/2)')) / sqrt(2π) / v₀
    result
end

"""
Expand the given matrix of coefficients.
The first dimension of `coefs` should be x, the second dimension is Hermite modes.
"""
function expand_bigfloat_hermite_f(coefs::AbstractMatrix, v::AbstractVector, v₀, normalized=true)
    M = size(coefs, 2) - 1
    normalized_He_n_vand = He_up_to_n(M, v/v₀; normalized)
    coefs * normalized_He_n_vand .* exp.(-(v/v₀).^2/2)' / sqrt(2π) / v₀
end

function expand_bigfloat_hermite_f(coefs::AbstractArray{Float64, 6}, vgrid::VGrid, v₀, normalized=true)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(coefs)
    Mvx = Nvx - 1
    Mvy = Nvy - 1
    Mvz = Nvz - 1

    vx_nodes = vgrid.x.nodes
    vy_nodes = vgrid.y.nodes
    vz_nodes = vgrid.z.nodes
    Kvx = length(vx_nodes)
    Kvy = length(vy_nodes)
    Kvz = length(vz_nodes)

    ndims = sum([Mvx > 0, Mvy > 0, Mvz > 0])

    (; VX, VY, VZ) = vgrid
    exp_weight = @. exp(-(VX^2 + VY^2 + VZ^2) / (2v₀)) / sqrt(2π)^ndims
    exp_weight = reshape(exp_weight, (Kvx, Kvy, Kvz))

    vx_vand = He_up_to_n(Mvx, vx_nodes/v₀; normalized=true) .|> Float64
    vy_vand = He_up_to_n(Mvy, vy_nodes/v₀; normalized=true) .|> Float64
    vz_vand = He_up_to_n(Mvz, vz_nodes/v₀; normalized=true) .|> Float64

    result = zeros(Nx, Ny, Nz, length(vx_nodes), length(vy_nodes), length(vz_nodes))

    @turbo for λxyz in CartesianIndices((Nx, Ny, Nz))
        for αvx in eachindex(vx_nodes), βvx in 1:Mvx+1
            for αvy in eachindex(vy_nodes), βvy in 1:Mvy+1
                for αvz in eachindex(vz_nodes), βvz in 1:Mvz+1
                    vand = vx_vand[βvx, αvx] * vy_vand[βvy, αvy] * vz_vand[βvz, αvz]
                    result[λxyz, αvx, αvy, αvz] += coefs[λxyz, βvx, βvy, βvz] * vand * exp_weight[αvx, αvy, αvz]
                end
            end
        end
    end
    result
end
