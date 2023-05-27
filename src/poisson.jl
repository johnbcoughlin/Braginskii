function poisson(u, sim, buffer)
    (; x_grid) = sim

    Nx, Ny, Nz = size(x_grid)

    Ex = alloc(Float64, buffer, Nx, Ny, Nz)
    Ex .= 0

    Ey = alloc(Float64, buffer, Nx, Ny, Nz)
    Ey .= 0

    return Ex, Ey
end

function apply_laplacian!(dest, ϕ, ϕ_left, ϕ_right, grid, buffer)
    Nx, Ny, Nz = size(ϕ)

    dx = grid.x.dx

    @no_escape buffer begin
        ϕ_with_x_bdy = alloc(Float64, buffer, Nx+6, Ny, Nz) |> Origin(-2, 1, 1)
        ϕ_with_x_bdy[1:Nx, :, :] .= ϕ
        apply_poisson_bcs!(ϕ_with_x_bdy, ϕ_left, ϕ_right)

        stencil = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / dx^2;
        convolve_x!(dest, ϕ_with_x_bdy, stencil, true, buffer)

        ϕ_yy = alloc(Float64, buffer, Nx, Ny, Nz)
        ϕ_yy .= ϕ
        in_ky_domain!(ϕ_yy, buffer) do ϕ̂
            apply_k²!(ϕ̂)
        end

        ϕ_zz = alloc(Float64, buffer, Nx, Ny, Nz)
        ϕ_zz .= ϕ
        in_kz_domain!(ϕ_zz, buffer) do ϕ̂
            apply_k²!(ϕ̂)
        end

        dest .+= ϕ_yy .+ ϕ_zz
        nothing
    end
end

function apply_k²!(ϕ̂)
    N1, N2 = size(ϕ̂)
    ϕ̂ = reshape(reinterpret(reshape, Float64, ϕ̂), (2N1, N2, :))
    for i in axes(ϕ̂, 1), k in axes(ϕ̂, 2), j in axes(ϕ̂, 3)
        ϕ̂[i, k, j] *= -(k-1)^2
    end
end

function apply_poisson_bcs!(ϕ, ϕ_left, ϕ_right)
    Nx6, Ny, Nz = size(ϕ)
    Nx = Nx6-6
    
    # Do left side first
    M = @SMatrix [3  -20  90;
                  0   -5  60;
                  0    0  35];
    Q = @SMatrix [60   -5  0  0;
                  90  -20  3  0;
                  140 -70 28 -5];
    rhs = -Q * reshape(ϕ[1:4, :, :], (4, :)) 
    rhs .+= 128 * [1, 1, 1] .* vec(ϕ_left)'

    ϕ_ghosts = reshape(M \ rhs, (3, Ny, Nz))
    ϕ[-2:0, :, :] .= ϕ_ghosts

    # Right side
    M = @SMatrix [90 -20   3;
                  60  -5   0;
                  35   0   0];
    Q = @SMatrix [ 0  0  -5  60;
                   0  3 -20  90;
                  -5 28 -70 140];
    rhs = -Q * reshape(ϕ[Nx-3:Nx, :, :], (4, :)) 
    rhs .+= 128 * [1, 1, 1] * vec(ϕ_right)'

    ϕ_ghosts = reshape(M \ rhs, (3, Ny, Nz))
    ϕ[Nx+1:Nx+3, :, :] .= ϕ_ghosts

end
