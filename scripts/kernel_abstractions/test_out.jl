module KernelAbstractionsTest

using KernelAbstractions
using Metal
using BenchmarkTools

@kernel function mul2_kernel(A)
  I = @index(Global)
  A[I] = 2 * A[I]
end

@kernel function mul_hermite_vx_kernel(A)
    NX, Nvx, Nvy, Nvz = @ndrange()

    iX, ivx, ivy, ivz = @index(Global, NTuple)

    if ivx > 1
        A[iX, ivx, ivy, ivz] += sqrt(Float32(ivx-1)) * A[iX, ivx-1, ivy, ivz]
    end
    if ivx < Nvx
        A[iX, ivx, ivy, ivz] += sqrt(Float32(ivx)) * A[iX, ivx+1, ivy, ivz]
    end
end

function call_kernel()
    dev = CPU()
    kernel = mul_hermite_vx_kernel(dev, 256)

    A=ones(Float32, 1024, 32, 32, 1)

    @btime begin 
        ev = $kernel(A, ndrange=size(A))
        KernelAbstractions.synchronize($dev)
    end setup=(A=ones(Float32, 1024, 32, 32, 1))

    A = ones(Float32, 1024, 32, 32, 1)
    ev = kernel(A, ndrange=size(A))
    KernelAbstractions.synchronize(dev)
    display(A[1, :, :, 1])
end

end
