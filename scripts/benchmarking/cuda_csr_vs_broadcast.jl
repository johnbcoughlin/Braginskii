using CUDA
using SparseArrays
using LinearAlgebra
using CUDA.CUSPARSE
using CUDA: i32
using BenchmarkTools

function do_cusparse_mul!(g, f, V)
    N = size(g, 1)
    g = reshape(g, (N*N, N*N))
    f = reshape(f, (N*N, N*N))

    CUDA.@sync mul!(g, f, V')
    #=
    @btime begin 
        CUDA.@sync mul!($g, $f, $V')
    end samples = 1000
    =#
end

function do_bcast!(g, f, X_diag)
    X = reshape(cu(X_diag), (1, 1, :, 1))

    for i in 1:3000
        g .= 0
        #g[:, :, 1, :] .= (@view f[:, :, 3, :]) * 2
        @. g[:, :, 1:end-1, :] += (@view f[:, :, 2:end, :]) * X
        @. g[:, :, 2:end, :] += (@view f[:, :, 1:end-1, :]) * X
    end
end

function kernel_naive!(g, f, X_diag, NX, Nvx, Nvy)
    # Compute position in output that this thread is responsible for
    iX = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    ivx = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    ivy = (blockIdx().z-1i32) * blockDim().z + threadIdx().z

    tmp = 0.0
    if ivx > 1 && ivx <= Nvx && iX <= NX && ivy <= Nvy
        @inbounds tmp += X_diag[ivx-1] * f[iX, ivx-1, ivy]
    end
    if ivx < Nvx && iX <= NX && ivy <= Nvy
        @inbounds tmp += X_diag[ivx] * f[iX, ivx+1, ivy]
    end
    if ivx <= Nvx && iX <= NX && ivy <= Nvy
        g[iX, ivx, ivy] = tmp
    end

    return
end

function naive_driver!(g, f, X_diag)
    N = size(g, 1)
    g = reshape(g, (N*N, N, N))
    f = reshape(f, (N*N, N, N))
    X_diag = cu(X_diag)

    threads = (32, 4, 4)
    blocks = Int.(ceil.((N*N, N, N) ./ threads))
    @cuda threads=threads blocks=blocks kernel_naive!(g, f, X_diag, N*N, N, N)
    #=
    @btime begin
        CUDA.@sync begin
            @cuda threads=$threads blocks=$blocks kernel_naive!($g, $f, $X_diag, $N*$N, $N, $N)
        end
    end samples=1000
    =#
end

function kernel_smem_chunked!(g, f, X_diag, NX, Nvx, Nvy)
    # Compute position in output that this thread is responsible for
    iX = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ivx = (blockIdx().y-1) * blockDim().y + threadIdx().y
    ivy = (blockIdx().z-1) * blockDim().z + threadIdx().z

    chunksize_X = blockDim().x
    chunksize_vx = blockDim().y + 2i32
    chunksize_vy = blockDim().z

    iX_local = threadIdx().x
    ivx_local = threadIdx().y + 1i32
    ivy_local = threadIdx().z

    f_chunk = CuDynamicSharedArray(Float64, (chunksize_X, chunksize_vx, chunksize_vy))
    X_diag_chunk = CuDynamicSharedArray(Float64, (chunksize_vx,))

    # Each thread reads its own element of f into shared mem
    if iX <= NX && ivy <= Nvy
        if ivx <= Nvx
            f_chunk[iX_local, ivx_local, ivy_local] = f[iX, ivx, ivy]
            X_diag_chunk[ivx_local] = X_diag[ivx_local]
        end
        if ivx > 1 && ivx_local == 2
            f_chunk[iX_local, ivx_local-1, ivy_local] = f[iX, ivx-1, ivy]
        end
        if ivx < Nvx && ivx_local == chunksize_vx-1
            f_chunk[iX_local, chunksize_vx, ivy_local] = f[iX, ivx+1, ivy]
        end
    end

    sync_threads()

    tmp = 0.0
    if ivx > 1 && ivx <= Nvx && iX <= NX && ivy <= Nvy
        tmp += X_diag[ivx-1] * f_chunk[iX_local, ivx_local-1, ivy_local]
    end
    if ivx < Nvx && iX <= NX && ivy <= Nvy
        tmp += X_diag[ivx] * f_chunk[iX_local, ivx_local+1, ivy_local]
    end
    if ivx <= Nvx && iX <= NX && ivy <= Nvy
        g[iX, ivx, ivy] = tmp
    end

    sync_threads()

    return
end

function smem_chunked_driver!(g, f, X_diag)
    N = size(g, 1)
    g = reshape(g, (N*N, N, N))
    f = reshape(f, (N*N, N, N))
    X_diag = cu(X_diag)

    threads = (32, 4, 4)
    blocks = Int.(ceil.((N*N, N, N) ./ threads))
    @cuda threads=threads blocks=blocks shmem=32000 kernel_smem_chunked!(g, f, X_diag, N*N, N, N)
    #=
    @btime begin
        CUDA.@sync begin
            @cuda threads=$threads blocks=$blocks shmem=16000 kernel_smem_chunked!($g, $f, $X_diag, $N*$N, $N, $N)
        end
    end samples=1000
    =#
end
function do_bench()
    N = 19
    X_diag = sqrt.(1:N)
    X = spdiagm(-1 => X_diag, 1 => X_diag)
    VX_csr = CuSparseMatrixCSR(X)
    VX_csc = CuSparseMatrixCSC(X)
    VX_coo = CuSparseMatrixCOO(X)
    @show typeof(VX_csr)
    @show typeof(VX_csc)
    VY = cu(I(N+1))

    V_csr = kron(VY, VX_csr)
    V_csc = kron(VY, VX_csc)
    V_coo = kron(VY, VX_coo)

    f = CUDA.rand(Float64, N+1, N+1, N+1, N+1)
    g = similar(f)

    #CUDA.@time do_cusparse_mul!(g, f, V_csc);
    g2 = copy(g)

    #CUDA.@time do_cusparse_mul!(g, f, V_coo);
    g3 = copy(g)

    #CUDA.@time do_bcast!(g, f, X_diag)
    #g4 = copy(g)
    #@assert isapprox(g1, g4, rtol=1e-7)
    
    smem_chunked_driver!(g, f, X_diag)
    g2 = copy(g)

    naive_driver!(g, f, X_diag)
    g5 = copy(g)

    do_cusparse_mul!(g, f, V_csr);
    g1 = copy(g)

    display((g1 - g2)[end, end, :, :])
    @show sum(abs.(g1 - g2) .> 1e-7)
    @show length(g1)

    @assert isapprox(g1, g5, rtol=1e-7)
    @assert isapprox(g2, g1, rtol=1e-7)
end

do_bench()

0.0

