function estimate_log_slope(Ns, errors)
    K = length(errors)
    X = hcat(ones(K), log.(Ns))
    x = X \ log.(errors)
    return x[2]
end
