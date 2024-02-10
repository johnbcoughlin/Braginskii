function make_sparse_array_from_stencil(stencil, N; periodic=false)
    K = length(stencil)
    if K % 2 == 0
        error("Stencils must be odd-length and centered around 0")
    end

    c = K÷2
    ks = -c:c
    pairs = [k => s*ones(N-abs(k)) for (k, s) in zip(ks, stencil)]
    A = spdiagm(pairs...)

    if periodic
        A[1:c, end-c+1:end] .= A[c+1:2c, 1:c]
        A[end-c+1:end, 1:c] .= A[end-2c+1:end-c, end-c+1:end]
    end

    return A
end
