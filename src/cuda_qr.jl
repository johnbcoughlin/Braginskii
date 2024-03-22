function cuda_batch_qr(A, buffer)
    n, r, NX = size(A)
    R = alloc_zeros(Float64, buffer, r, r, NX)
    Rinvs = alloc_array(Float64, buffer, r, r, NX)
    Taus = alloc_array(Float64, buffer, r, NX)
    Q = alloc_array(Float64, buffer, size(A)...)
    # Scratch space so that we don't overwrite A
    B = alloc_array(Float64, buffer, size(A)...)
    B .= A
    cuda_batch_qr!(Q, R, B, Rinvs, Taus)
end

function cuda_batch_qr!(Q, R, A, Rinvs, Taus, buffer)
    A_ptrs = CUBLAS.unsafe_strided_batch(A)
    Tau_ptrs = CUBLAS.unsafe_strided_batch(Taus)
    lda = CUDA.stride(A, 2)
    m, n, NX = size(A)
    info = Ref{Cint}()
    CUBLAS.cublasDgeqrfBatched(CUBLAS.handle(), m, n, A_ptrs, lda, Tau_ptrs, info, NX)
    if info[] != 0
        throw(ArgumentError,string("Invalid value at ",-info[]))
    end
    for i in 1:size(R, 1)
        R[1:i, i, :] .= A[1:i, i, :]
        A[1:i-1, i, :] .= 0.0
        A[i, i, :] .= 1.0
    end
    Q = cuda_form_Q_from_Hs(A, Taus, buffer)
    return Q, R
end

function cuda_form_Q_from_Hs(A, Taus, buffer)
    n, r, NX = size(A)
    Q = alloc_zeros(Float64, buffer, n, r, NX)
    # Set up scratch space for the vector
    b = alloc_zeros(Float64, buffer, 1, 1, NX)

    b_ptrs = CUBLAS.unsafe_strided_batch(b)

    cuda_A_pointers_vec = [CUDA.pointer(A, 1) for k in 1:NX]
    cuda_Q_pointers_vec = [CUDA.pointer(Q, 1) for k in 1:NX]
    stride = n * r

    # Computing columns from right to left
    for i in r:-1:1
        Q[i, i, :] .= 1.0
        Qi_col_start = (i-1) * n + 1
        for k in 1:NX
            cuda_Q_pointers_vec[k] = CUDA.pointer(Q, (k-1)*stride + Qi_col_start)
        end
        Qi_ptrs = CuArray(cuda_Q_pointers_vec)

        for j in i:-1:1
            vj_col_start = (j-1) * n + 1
            for k in 1:NX
                cuda_A_pointers_vec[k] = CUDA.pointer(A, (k-1)*stride + vj_col_start)
            end
            vj_ptrs = CuArray(cuda_A_pointers_vec)

            # Compute b = vj' * Q[i]
            # Consider vj as an n x 1 matrix
            CUBLAS.cublasDgemvBatched(CUBLAS.handle(), 'T', n, 1, 1.0, 
                vj_ptrs, n, 
                Qi_ptrs, 1,
                0.0, b_ptrs, 1,
                NX)
            Q[:, i, :] .-= A[:, j, :] .* Taus[j, :]' .* b[1, 1, :]'
        end
    end
    return Q
end

function Z_proj!(dest, f, Z)
    CUBLAS.gemm_strided_batched!('N', 'N', 1.0, f, Z, 0.0, dest)
end

function Z_projections!(dests::ArrayPartition, fs::ArrayPartition, Zs::ArrayPartition)
    for i in eachindex(dests.x)
        Z_proj!(dests.x[i], fs.x[i], Zs.x[i])
    end
end

function X_proj!(dest, f, X)
    CUBLAS.gemm_strided_batched!('T', 'N', 1.0, f, X, 0.0, dest)
end

function X_projections!(dests::ArrayPartition, fs::ArrayPartition, Xs::ArrayPartition)
    for i in eachindex(dests.x)
        X_proj!(dests.x[i], fs.x[i], Xs.x[i])
    end
end

function XZ_proj!(dest, f, X, Z, buffer)
    Nvx, Nvz, NX = size(f)
    _, r, _ = size(X)
    tmp = alloc_array(Float64, buffer, Nvx, r, NX)
    Z_proj!(tmp, f, Z)
    X_proj!(dest, tmp, X)
end

function XZ_projections!(dests::ArrayPartition, fs::ArrayPartition, Xs::ArrayPartition, Zs::ArrayPartition, buffer)
    for i in eachindex(dests.x)
        XZ_proj!(dests.x[i], fs.x[i], Xs.x[i], Zs.x[i], buffer)
    end
end
