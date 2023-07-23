export as_xvx, as_yvy, as_zvz, as_vxvy, as_xyvxvy, as_vxvz

as_xvx(f) = f[:, 1, 1, :, 1, 1] |> Array |> copy
as_yvy(f) = f[1, :, 1, 1, :, 1] |> Array |> copy
as_zvz(f) = f[1, 1, :, 1, 1, :] |> Array |> copy
as_vxvy(f) = f[1, 1, 1, :, :, 1] |> Array |> copy
as_vxvz(f) = f[1, 1, 1, :, 1, :] |> Array |> copy
as_xyvxvy(f) = f[:, :, 1, :, :, 1] |> Array |> copy

# Scheme may be :WENO or :fourier.
# points gives the number of points to interpolate to per cell.
function interpolate(f, (scheme1, scheme2), (points1, points2))
    Nx, Ny = size(f)
    k1 = length(points1)
    k2 = length(points2)

    g1 = zeros(Nx*k1, Ny*k2)
    for λy in 1:Ny
        if scheme1 == :weno
            weno_interpolate!(reshape(g1[:, λy], k1, Nx), f[:, λy], points1)
        end
    end

    return g1

    g2 = zeros(Nx*k1, Ny*k2)
    for λx in 1:Nx
        if scheme2 == :weno
            weno_interpolate!(reshape(g2[λx, :], k2, Ny), g1[λx, :], points2)
        end
    end

    g2
end

# Points should be a vector of points in the interval [-1/2, 1/2].
function weno_interpolate!(g, f, points)
    N = length(f)

    # Based on u(-2, -1, 0) = [u_i-2, u_i-1, u_i]
    p1(u) = x -> u[-2] + 0.5*(2+x)*((-1+x)*u[-2]-2*x*u[-1]+(1+x)*u[0])
    # Based on [u_i-1, u_i, u_i+1]
    p2(u) = x -> u[-1] + 0.5*(1+x)*((-2+x)*u[-1]-2*(-1+x)*u[0] + x*u[1])
    # Based on [u_i, u_i+1, u_i+2]
    p3(u) = x -> u[0] + 0.5*x*((-3+x)*u[0] - 2*(-2+x)*u[1] + (-1+x)*u[2])

    β1(u) = 1/3 * (4u[-2]^2 - 19u[-2]*u[-1] + 25u[-1]^2 + 11u[-2]*u[0] - 31u[-1]*u[0] + 10u[0]^2)
    β2(u) = 1/3 * (4u[-1]^2 - 13u[-1]*u[0] + 13u[0]^2 + 5u[-1]*u[1] - 13u[0]*u[1] + 4u[1]^2)
    β3(u) = 1/3 * (10u[0]^2 - 31u[0]*u[1] + 25u[1]^2 + 11u[0]*u[2] - 19u[1]*u[2] + 4u[2]^2)

    g[:, 1] .= p3(Origin(0)(f[1:3])).(points)
    g[:, 2] .= p2(Origin(-1)(f[1:3])).(points)
    for i in 3:N-2
        u1 = Origin(-2)(f[i-2:i])
        u2 = Origin(-1)(f[i-1:i+1])
        u3 = Origin(0)(f[i:i+2])

        β1i = β1(u1)
        β2i = β2(u2)
        β3i = β3(u3)

        q(x) = (1/4 - x^2)
        γ1(x) = 3/16 - x/4 - 1/2 * q(x)
        γ2(x) = 5/8 + (1/4 - x^2)
        γ3(x) = 3/16 + x/4 - 1/2 * q(x)


        w1(x) = γ1(x) / (1e-6 + β1i)^2
        w2(x) = γ2(x) / (1e-6 + β2i)^2
        w3(x) = γ3(x) / (1e-6 + β3i)^2

        w(x) = begin
            w̃ = [w1(x), w2(x), w3(x)]
            w̃ ./ sum(w̃)
        end

        wi = w.(points)

        for k in eachindex(points)
            pk = points[k]
            wik = w(pk)
            g[k, i] = wik[1] * p1(u1)(pk) + wik[2] * p2(u2)(pk) + wik[3] * p3(u3)(pk)
        end
    end
    g[:, N-1] .= p2(Origin(-1)(f[N-2:N])).(points)
    g[:, N] .= p1(Origin(-2)(f[N-2:N])).(points)

    g
end
