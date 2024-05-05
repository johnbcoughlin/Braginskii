using LinearAlgebra
using SpecialFunctions

function find_peaks(E)
    vcat(false, false, (diff(sign.(diff(log.(E)))) .== -2))
end

"""
find_damping_fit(t, E)

Find the best fit to an exponential damping model for the given energy trace.
That is, finds a fit of the form E ≈ E₀*exp(γ*t).
Returns the fitted exponential curve for plotting as well as the slope γ.
"""
function find_damping_fit(t, E, use_peaks=nothing)
    peaks = (diff(sign.(diff(log.(E)))) .== -2)

    j = sum(peaks)

    fit_slope(range) = begin
        t̂ = t[3:end][peaks][range]
        X = hcat(t̂, ones(length(range)))
        Ê = E[3:end][peaks][range] .|> log
        y = X \ Ê
    end

    fit_freq(range) = begin
        t̂ = t[3:end][peaks][range]
        X = hcat(1:length(range), ones(length(range)))
        y = X \ t̂
    end

    find_best_fit(fitfunc) = begin
        changes = fill(Inf, j, j)
        fits = Array{Vector{Float64}}(undef, j, j)

        for first in 1:j-1
            for last in (first+2):j
                y = fitfunc(first:last)
                fits[first, last] = y

                y_prev = fitfunc((first+1):last)
                y_next = fitfunc(first:(last-1))
                rel_prev = norm(y - y_prev) / max(norm(y), norm(y_prev))
                rel_next = norm(y - y_next) / max(norm(y), norm(y_next))

                changes[first, last] = max(rel_prev, rel_next)
            end
        end
        bestJ = argmin(changes)
        fits[bestJ]
    end

    γ, intercept = if isnothing(use_peaks) 
        find_best_fit(fit_slope)
    else
        fit_slope(use_peaks)
    end

    line = exp.(γ * t .+ intercept)

    invfreq, intercept = if isnothing(use_peaks)
        find_best_fit(fit_freq)
    else
        fit_freq(use_peaks)
    end
    freq = 2π/invfreq

    line, γ, freq
end

function find_pure_growth_slope(t, E, time_range=nothing)
    if !isnothing(time_range)
        range = searchsortedfirst(t, first(time_range)):searchsortedlast(t, last(time_range))
    else
        range = Colon()
    end
    t̂ = t[range]
    Ê = E[range]
    X = hcat(t̂, ones(length(t̂)))
    y = X \ (log.(Ê))
    γ, intercept = y
    line = exp.(γ * t .+ intercept)
    return line, γ
end

"""
Plasma dispersion functions

Growth rates occur at zeros of ϵ
"""

Z(ζ) = im * sqrt(π) * exp(-ζ^2) * (1 + erf(im * ζ))

ϵ(ω, k; vₜₕ = 1, λD = 1) = begin
    z = ω / k
    ζ = z / (√(2) * vₜₕ)
    1 + (1 / (k * λD)^2) * (1 + ζ * Z(ζ))
end


