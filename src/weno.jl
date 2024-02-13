struct WENO5Stencils{StackedStencil, BETA_COMBINER, GAMMA}
    # The stencils S1, S2, S3, stacked in such a way that they can be independently
    # convolved with the input to obtain 3 channels
    S1S2S3_stencils::StackedStencil

    # A set of six stencils, each three wide, used in the construction of the smoothness
    # indicators.
    β_stencils::StackedStencil

    # A matrix with entries
    # [13/12     0   0;
    #    1/4     0   0;
    #        13/12   0;
    #          1/4   0;
    #            13/12;
    #              1/4]
    β_combiner::BETA_COMBINER

    # 1x3 matrix of the linear weights
    γ::GAMMA
end

# The flux arrays F⁺ and F⁻ are directly upwinded based on characteristic splitting.
# The tuple γ contains the WENO linear weights.
function perform_weno_differencing_z!(df, F⁺, F⁻,
    left_biased_stencils, right_biased_stencils, 
    x_grid, α, buffer; ϵ=1e-6)

    (Nx, Ny, Nz, Nvx, Nvy, Nvz) = size(α.discretization)

    F̂⁺ = weno_reconstruction_z!(reshape(F⁺, (Nx*Ny, :, Nvx*Nvy*Nvz)), left_biased_stencils, buffer)
    F̂⁻ = weno_reconstruction_z!(reshape(F⁻, (Nx*Ny, :, Nvx*Nvy*Nvz)), right_biased_stencils, buffer)

    dz = x_grid.z.dx

    @. df += ((@view F̂⁻[:, 2:end-1, :]) - (@view F̂⁺[:, 2:end-1, :])) / dz
end

function weno_reconstruction_z!(F::AbstractArray{Float64, 3}, stencils, buffer; ϵ=1e-6)
    Nxy, Nz6, Nvxyz = size(F)
    Nz = Nz6-6
    Nz2 = Nz+2

    S1S2S3 = alloc_array(Float64, buffer, Nxy, Nz2, Nvxyz, 3)
    convolve_z!(S1S2S3, F, stencils.S1S2S3.stencils, true, buffer, 3)

    β_terms = alloc_array(Float64, buffer, Nxy, Nz2, Nvxyz, 6)
    convolve_z!(β_terms, F, stencils.β_stencils, true, buffer, 6)

    # Square terms as per eqn (17) of Zhang and Shu (2016)
    @. β_terms *= β_terms

    β_terms = reshape(β_terms, (:, 6))
    β = alloc_array(Float64, buffer, Nxy*Nz2*Nvxyz, 3)
    mul!(β, β_terms, β_combiner)

    γ = stencils.γ
    α = alloc_array(Float64, buffer, Nxy*Nz2*Nvxyz, 3)
    @. α = γ / (ϵ + β)

    α_sum = sum(α, dims=2)

    w = alloc_array(Float64, buffer, Nxy*Nz2*Nvxyz, 3)
    @. w = α / α_sum

    S = reshape(S1S2S3, (:, 3))
    wS = alloc_array(Float64, buffer, Nxy*Nz2*Nvxyz, 3)
    @. wS = w * S

    return reshape(sum(wS, dims=2), (Nxy, Nz2, Nvxyz))
end

# Computes the reconstruction at i+1/2 using a left-biased stencil
function left_biased_weno5_stencils(T=Array)
    S1S2S3_stencils = [
    1/3   -7/6   11/6  0    0;
    0     -1/6   5/6  1/3   0;
    0      0     1/3  5/6  -1/6]' |> T

    β_stencils = [
    1  -2   1   0  0;
    1  -4   3   0  0;
    0   1  -2   1  0;
    0   1   0  -1  0;
    0   0   1  -2  1;
    0   0   3  -4  1]' |> T

    β_combiner = [
    13/12    0     0
    1/4      0     0;
     0   13/12     0;
     0     1/4     0;
     0      0   13/12;
     0      0     1/4] |> T

    γ = [1/10   3/5    3/10] |> T

    return WENO5Stencils(S1S2S3_stencils, β_stencils, β_combiner, γ)
end

# Computes the reconstruction at i-1/2 using a right-biased stencil
function right_biased_weno5_stencils(T=Array)
    S1S2S3_stencils = [
   -1/6  5/6   1/3  0   0;
    0   1/3   5/6   -1/6  0;
    0   0   11/6   -7/6   1/3]' |> T

    β_stencils = [
    1  -2   1  0  0;
    1  -4   3  0  0;
    0  1  -2   1   0;
    0 1  0  -1  0;
    0  0  1  -2  1;
    0  0  3  -4  1]' |> T

    β_combiner = [
    13/12    0     0;
    1/4      0     0;
     0   13/12     0;
     0     1/4     0;
     0      0   13/12;
     0      0     1/4] |> T

    γ = [3/10  3/5  1/10] |> T

    return WENO5Stencils(S1S2S3_stencils, β_stencils, β_combiner, γ)
end
