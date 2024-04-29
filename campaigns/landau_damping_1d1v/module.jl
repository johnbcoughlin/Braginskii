module LandauDamping1D1V

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using PDEHarness
using Interpolations

function params(; α, k)
    Nx = 64
    Nvx = 256

    Lx = 2pi / k

    fe_eq(x, vx) = (1 + α * cos(k * x)) / sqrt(2pi) * exp(-vx^2/2)

    return (; 
        Nx, Nvx, α, k,
        Lx,
        fe_eq)
end

function make_sim_vlasov(::Val{device}; k=1.0, α=1e-3) where {device}
    d = Dict{String, Any}()

    problem = "landau_damping_1d1v"

    (; Nx, Nvx,
        Lx,
        fe_eq) = params(; α, k)
    
    merge!(d, @strdict problem Nx Nvx Lx α k)

    # Now calculate the zeroth moment of fi_eq to obtain ne_eq
    @info "Setting up sim"
    sim = Helpers.single_species_1d1v_x(fe_eq;
        Nx, Nvx, Lx,
        vdisc=:hermite,
        device,
        νpτ=0.0, ωpτ=1.0, ωcτ=0.0, q=1.0, 
        z_bcs=:reservoir);
    @info "Done setting up sim"

    return (; d, sim)
end

end

