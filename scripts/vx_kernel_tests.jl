module CUDAVXKernel

using DrWatson
@quickactivate :Braginskii
using Braginskii.Helpers
using CairoMakie

using CUDA
using BenchmarkTools
using SparseArrays
using LinearAlgebra

Nx = 16
Nz = 72
Nvx = 30
Nvz = 30
δ = 0.001
q = 0.0
k = 0.5
By0 = 0.0
gz = -1.0

Lx = 0.75
Ly = 1.5

n_ref = 0.5
T_ref = 30.0

Kn = 0.01
λ_mfp = Kn * Lx
vth = sqrt(T_ref)
@show ν_p = vth / λ_mfp

alpha = 25.0



n0(x, z) = begin
    return n_ref/2 * tanh(alpha*z / Ly) + 1.5*n_ref
end
p0(x, z) = begin
    return -gz * n_ref/2 * (-log(cosh(alpha*z / Ly))/alpha*Ly - 3*z) + 1.5*n_ref * T_ref
end
T0(x, z) = p0(x, z) / n0(x, z)

k = π / Lx
yr = Ly / 10
uz0(x, z) = -0.1*cos(k*x)*exp(-z^2/(2*yr^2))


f_0(x, z, vx, vz) = begin
    return n0(x, z) / (2π*T0(x, z)) * exp(-(vx^2 + (vz-uz0(x, z))^2) / (2T0(x, z)))
end

sim = Helpers.single_species_xz_2d2v((; f_0, By0);
    Nx, Nz, Nvx, Nvz,
    zmin=-Ly, zmax=Ly, Lx=2*Lx,
    vdisc=:hermite, ϕ_left=1.0, ϕ_right=1.0, vth=sqrt(6.0),
    ν_p, q, gz, device=:gpu, z_bcs=:reservoir);


function hermite_vx_kernel!(dest, f, vth::Float32)
    thread_index_X = threadIdx().x
    stride_X = blockDim().x
    ivyvz = blockIdx().x
    
#     block_idx_vyvz = blockIdx().x
#     stride_vyvz = blockDim().x

    Nvx = size(dest, 2)
    
#     for ivyvz in block_idx_vyvz:stride_vyvz:size(dest, 3)
        @inbounds for iX in thread_index_X:stride_X:size(dest, 1)
            dest[iX, 1, ivyvz] = f[iX, 2, ivyvz] * vth
            for ivx in 2:size(dest, 2)-1
                dest[iX, ivx, ivyvz] = vth * (sqrt(Float32(ivx-1)) * f[iX, ivx-1, ivyvz] + sqrt(Float32(ivx)) * f[iX, ivx+1, ivyvz])
            end
            dest[iX, Nvx, ivyvz] = vth * (sqrt(Float32(Nvx-1)) * f[iX, Nvx-1, ivyvz])
        end
#     end
end

function vx_mul_by_hand!(dest, f, discretization)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    shape3d = (Nx*Ny*Nz, Nvx, Nvy*Nvz)
    f = reshape(f, shape3d)
    dest = reshape(dest, shape3d)
    
    kernel = @cuda launch=false hermite_vx_kernel!(dest, f, Float32(discretization.vdisc.vth))
    config = launch_configuration(kernel.fun)
    @show CUDA.registers(kernel)
#     @info "" config
    
#     blocks = min(config.blocks, Nvy*Nvz)
#     threads = config.threads
#     kernel(dest, f, discretization.vdisc.vth; threads, blocks)
    
    @cuda threads=512 blocks=Nvy*Nvz hermite_vx_kernel!(dest, f, Float32(discretization.vdisc.vth))
end

function test_out_vx_kernel()
    Nvx = 300
    f = cu(rand(320, Nvx, 3))
    dest = similar(cu(f))
    dest .= 0
    @cuda threads=32 blocks=3 hermite_vx_kernel!(dest, f, 1.0)
    Ξx = spdiagm(-1 => sqrt.(1:Nvx-1), 1 => sqrt.(1:Nvx-1))
    
    expected = f[:, :, 2] * cu(Ξx)'
    actual = dest[:, :, 2]
#     @info "" Array(expected - actual)
    @show expected ≈ actual
end

test_out_vx_kernel()


function profile_my_kernel()
    f = sim.u.x[1]

    dest1 = similar(f);
    dest1 .= 0.0
    dest2 = similar(f);
    dest2 .= 0.0

    Braginskii.mul_by_vx!(dest1, f, sim.species[1].discretization);
    vx_mul_by_hand!(dest2, f, sim.species[1].discretization);

    @show norm(dest1 - dest2) / norm(dest1)

    iters = 1

    CUDA.@profile external=true begin
        for i in 1:iters
            vx_mul_by_hand!(dest2, f, sim.species[1].discretization);
        end
        CUDA.@sync dest2;
    end
    nothing
end

function run_my_kernel()
    f = sim.u.x[1]

    dest1 = similar(f);
    dest1 .= 0.0
    dest2 = similar(f);
    dest2 .= 0.0

    Braginskii.mul_by_vx!(dest1, f, sim.species[1].discretization);
    vx_mul_by_hand!(dest2, f, sim.species[1].discretization);

    @show norm(dest1 - dest2) / norm(dest1);
end

end
