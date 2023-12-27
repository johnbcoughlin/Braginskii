using CUDA
using SparseArrays
using LinearAlgebra
using CUDA.CUSPARSE
using CUDA: i32
using BenchmarkTools

function do_cusparse_mul!(g, f, V, α)
    N = size(g, 1)
    g = reshape(g, (N*N, N*N))
    f = reshape(f, (N*N, N*N))

    for i in 1:1000
        CUDA.@sync mul!(g, f, V', α, 0.0)
    end
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
    for i in 1:1000
        @cuda threads=threads blocks=blocks kernel_naive!(g, f, X_diag, N*N, N, N)
    end
    #=
    @btime begin
        CUDA.@sync begin
            @cuda threads=$threads blocks=$blocks kernel_naive!($g, $f, $X_diag, $N*$N, $N, $N)
        end
    end samples=1000
    =#
end

function kernel_striped!(g, f, X_diag, NX, Nvx, Nvy)
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

function striped_driver!(g, f, X_diag)
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
    iX = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    ivx = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    ivy = (blockIdx().z-1i32) * blockDim().z + threadIdx().z

    chunksize_X = blockDim().x
    chunksize_vx = blockDim().y + 2i32
    chunksize_vy = blockDim().z

    iX_local = threadIdx().x
    ivx_local = threadIdx().y + 1i32
    ivy_local = threadIdx().z

    f_chunk = CuDynamicSharedArray(Float64, (chunksize_X, chunksize_vx, chunksize_vy))
    offset1 = 8 * chunksize_X * chunksize_vx * chunksize_vy
    X_diag_chunk = CuDynamicSharedArray(Float64, (chunksize_vx,), offset1)

    # Each thread reads its own element of f into shared mem
    if iX <= NX && ivy <= Nvy
        if ivx <= Nvx
            @inbounds f_chunk[iX_local, ivx_local, ivy_local] = f[iX, ivx, ivy]
        end
        if ivx < Nvx
            @inbounds X_diag_chunk[ivx_local] = X_diag[ivx]
        end
        if ivx > 1 && ivx_local == 2
            @inbounds f_chunk[iX_local, ivx_local-1, ivy_local] = f[iX, ivx-1, ivy]
            @inbounds X_diag_chunk[ivx_local-1] = X_diag[ivx-1]
        end
        if ivx < Nvx && ivx_local == chunksize_vx-1
            @inbounds f_chunk[iX_local, chunksize_vx, ivy_local] = f[iX, ivx+1, ivy]
        end
    end

    sync_threads()

    tmp = 0.0
    if iX <= NX && ivy <= Nvy
        if ivx > 1 && ivx <= Nvx
            @inbounds tmp += X_diag_chunk[ivx_local-1] * f_chunk[iX_local, ivx_local-1, ivy_local]
        else
            tmp += 0.0
        end

        if ivx < Nvx
            @inbounds tmp += X_diag_chunk[ivx_local] * f_chunk[iX_local, ivx_local+1, ivy_local]
        else
            tmp += 0.0
        end

        if ivx <= Nvx
            @inbounds g[iX, ivx, ivy] = tmp
        end
    end

    return
end

function smem_chunked_driver!(g, f, X_diag)
    N = size(g, 1)
    g = reshape(g, (N*N, N, N))
    f = reshape(f, (N*N, N, N))
    X_diag = cu(X_diag)

    threads = (48, 10, 1)
    shmem = 8 * (threads[1] * (threads[2]+2) * threads[3] + threads[2]+2)
    blocks = Int.(ceil.((N*N, N, N) ./ threads))
    @cuda threads=threads blocks=blocks shmem=shmem kernel_smem_chunked!(g, f, X_diag, N*N, N, N)
    #=
    @btime begin
        CUDA.@sync begin
            @cuda threads=$threads blocks=$blocks shmem=$shmem kernel_smem_chunked!($g, $f, $X_diag, $N*$N, $N, $N)
        end
    end samples=1000
    =#
end
function do_bench()
    N = 19
    X_diag = sqrt.(1:N) .|> Float64
    X = spdiagm(-1 => X_diag, 1 => X_diag) .|> Float64
    VX_csr = CuSparseMatrixCSR(X)
    VX_csc = CuSparseMatrixCSC(X)
    VX_coo = CuSparseMatrixCOO(X)
    VY = cu(I(N+1))


    V_csr = kron(VY, VX_csr)
    V_csc = kron(VY, VX_csc)
    V_coo = kron(VY, VX_coo)

    f = CUDA.rand(Float64, N+1, N+1, N+1, N+1)
    g = similar(f)
    T = CUDA.rand(Float64, N+1, N+1, 1, 1)
    for i in 1:1000
        @. g = f * T
    end
    f = CUDA.rand(Float64, N+1, N+1, N+1, N+1)
    g = similar(f)

    #CUDA.@time do_cusparse_mul!(g, f, V_csc);
    g2 = copy(g)

    #CUDA.@time do_cusparse_mul!(g, f, V_coo);
    g3 = copy(g)

    #CUDA.@time do_bcast!(g, f, X_diag)
    #g4 = copy(g)
    #@assert isapprox(g1, g4, rtol=1e-7)
    
    #smem_chunked_driver!(g, f, X_diag)

    striped_driver!(g, f, X_diag*4.5)
    g2 = copy(g)

    naive_driver!(g, f, X_diag*4.5)
    g2 = copy(g)

    do_cusparse_mul!(g, f, V_csr, 4.5);
    g1 = copy(g)

    #=
    display((g1 - g2)[end, end, :, :])
    @show sum(abs.(g1 - g2) .> 1e-6)
    indices = findall(>(1e-6), Array(abs.(g1 - g2)))
    @show indices
    display(Array(abs.(g1 - g2))[indices])
    @show length(g1)
    =#

    #@assert isapprox(g1, g2, rtol=1e-7)
    @assert isapprox(g2, g1, rtol=1e-7)
end

do_bench()

0.0

