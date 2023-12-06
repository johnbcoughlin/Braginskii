using CUDA
using SparseArrays
using LinearAlgebra
using CUDA.CUSPARSE

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

    CUDA.@time do_cusparse_mul!(g, f, V_csr);
    g1 = copy(g)

    CUDA.@time do_cusparse_mul!(g, f, V_csc);
    g2 = copy(g)

    CUDA.@time do_cusparse_mul!(g, f, V_coo);
    g3 = copy(g)

    CUDA.@time do_bcast!(g, f, X_diag)
    g4 = copy(g)

    @assert isapprox(g1, g2, rtol=1e-7)
    @assert isapprox(g1, g3, rtol=1e-7)
end

0.0

do_bench()

