# Computes and returns ∂xu if u is a size (Nx, Ny, Nz)
function ∂x(u, sim, buffer)
    Nx, Ny, Nz = size(sim.x_grid)
    Kx = (Nx ÷ 2) + 1

    result = alloc_array(Float64, buffer, Nx, Ny, Nz)
    xy_modes = alloc_zeros(Complex{Float64}, buffer, Kx, Ny, Nz)
    mul!(xy_modes, sim.fft_plans.kxy_rfft, u)
    kxs = alloc_array(Complex{Float64}, buffer, Kx, 1, 1)
    kxs .= (im * 2π / sim.x_grid.x.L) * ((0:Kx-1))
    xy_modes .*= kxs

    mul!(result, sim.fft_plans.kxy_irfft, xy_modes)
    return result
end

# Computes y s.t. ∂z_y = u
function ∂z_inv(u, sim, buffer)
    Nx, Ny, Nz = size(sim.x_grid)
    u = reshape(u, (:, Nz))
    result = alloc_array(Float64, buffer, Nx, Ny, Nz)
    Dz_inv = sim.x_grid.Dz_inv
    mul!(reshape(result, (:, Nz)), u, Dz_inv')
    return result
end
