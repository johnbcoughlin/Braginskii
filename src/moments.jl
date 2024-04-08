# Moments are computed using a "periodic" Trapezoid rule

function moments(f, disc::XVDiscretization{WENO5}, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1y = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M2 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    grid = disc.vdisc.grid

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.z.dx
    end

    fvx = f .* grid.VX
    fvy = f .* grid.VY
    fvz = f .* grid.VZ
    fv2 = @. f * (grid.VX^2 + grid.VY^2 + grid.VZ^2)
    sum!(M0, f)
    sum!(M1x, fvx)
    sum!(M1y, fvy)
    sum!(M1z, fvz)
    sum!(M2, fv2)

    M0 .*= dv
    M1x .*= dv
    M1y .*= dv
    M1z .*= dv
    M2 .*= dv

    M0, (M1x, M1y, M1z), M2
end

function density(f, disc::XVDiscretization{WENO5}, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    grid = disc.vdisc.grid

    dv = 1.0
    if :vx ∈ v_dims
        dv *= grid.x.dx
    end
    if :vy ∈ v_dims
        dv *= grid.y.dx
    end
    if :vz ∈ v_dims
        dv *= grid.z.dx
    end

    sum!(M0, f)
    M0 .*= dv

    return M0
end

function collisional_moments!(sim, f, buffer)
    if length(sim.species) == 1
        α = sim.species[1]

        f = f.x[1]

        (ux, uy, uz), T, ν = collisional_moments_single_species(α, f, sim.x_grid, sim.νpτ, buffer)

        cm = sim.collisional_moments[(α.name, α.name)]
        cm.ux .= ux
        cm.uy .= uy
        cm.uz .= uz
        cm.T .= T
        cm.ν .= ν
        return
    elseif length(sim.species) == 2
        sp_moments_1 = moments(f.x[1], sim.species[1].discretization, sim.species[1].v_dims, buffer)
        sp_moments_2 = moments(f.x[2], sim.species[2].discretization, sim.species[2].v_dims, buffer)
        for i in 1:2
            for j in 1:2
                α = sim.species[i]
                β = sim.species[j]
                α_moments = i == 1 ? sp_moments_1 : sp_moments_2
                β_moments = j == 1 ? sp_moments_1 : sp_moments_2

                (ux, uy, uz), T, ν = collisional_moments_two_species(α, β, α_moments, β_moments, sim.x_grid, sim.νpτ, buffer)

                cm = sim.collisional_moments[(α.name, β.name)]
                cm.ux .= ux
                cm.uy .= uy
                cm.uz .= uz
                cm.T .= T
                cm.ν .= ν

            end
        end
    else
        error("more than 2 species unsupported")
    end
end

function collisional_moments_single_species(α, f, x_grid, ν_p, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    M0, M1, M2 = moments(f, α.discretization, α.v_dims, buffer)
    M1x, M1y, M1z = M1

    ux = M1x ./ M0
    uy = M1y ./ M0
    uz = M1z ./ M0
    d = length(α.v_dims)

    T = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @. T = α.m * (M2 / M0 - (ux^2 + uy^2 + uz^2)) / d
    ν = alloc_zeros(Float64, buffer, size(x_grid)...)
    ν .= ν_p

    return (ux, uy, uz), T, ν
end

function collisional_moments_two_species(α, β, α_moments, β_moments, x_grid, νpτ, buffer)
    Nx, Ny, Nz = size(x_grid)

    ma = α.m
    mb = β.m

    M0α, M1α, M2α = α_moments
    M0β, M1β, M2β = β_moments

    na = M0α
    nb = M0β

    Ta = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    uxα = M1α[1] ./ M0α
    uyα = M1α[2] ./ M0α
    uzα = M1α[3] ./ M0α
    d = length(α.v_dims)
    @. Ta = ma * (M2α / M0α - (uxα^2 + uyα^2 + uzα^2)) / d

    Tb = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    uxβ = M1β[1] ./ M0β
    uyβ = M1β[2] ./ M0β
    uzβ = M1β[3] ./ M0β
    d = length(β.v_dims)
    @. Tb = mb * (M2β / M0β - (uxβ^2 + uyβ^2 + uzβ^2)) / d

    νab = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    νab_fac = sqrt(α.m * β.m) * (α.m + β.m) * (α.q * β.q)^2 / α.m
    @. νab = νpτ * νab_fac * nb / (α.m * Tb + β.m * Ta)^(3/2)

    uab(ua, ub) = begin
        res = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
        @. res = (νab * α.m * na * ua + νab * β.m * nb * ub) / (νab * α.m * na + νab * β.m * nb)
        res
    end
    uab_x = uab(uxα, uxβ)
    uab_y = uab(uyα, uyβ)
    uab_z = uab(uzα, uzβ)


    u2(ux, uy, uz) = begin
        res = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
        @. res = ux^2 + uy^2 + uz^2
        res
    end
    uab2 = u2(uab_x, uab_y, uab_z)
    ua2 = u2(uxα, uyα, uzα)
    ub2 = u2(uxβ, uyβ, uzβ)

    Tab = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @. Tab = (Ta * νab * na + Tb * νab * nb) / (νab * na + νab * nb)
    @. Tab -= (νab * na * ma * (uab2 - ua2) + νab * nb * mb * (uab2 - ub2)) / (3*νab * (na + nb))

    #=
    @show (α.name, β.name)
    @show Tab
    @show uab_x
    @show uab_y
    @show uab_z
    @show extrema(Ta)
    @show extrema(Tb)
    @show extrema(Tab)
    @show extrema(νab)
    =#

    return (uab_x, uab_y, uab_z), Tab, νab
end

function density(f, vdisc::Hermite, _, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)
    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M0 .= (@view f[:, :, :, 1, 1, 1])
    return M0
end

function density(f, α::Species{<:Hermite}, args...)
    density(f, α.discretization.vdisc, args...)
end

function density(f, α::Species{<:HermiteLaguerre}, B, buffer)
    Nx, Ny, Nz, Nμ, Nvy = size(f)
    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    @. M0 = (@view f[:, :, :, 1, 1])
    return M0
end

function moments(f, disc::XVDiscretization{<:Hermite}, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    vth = disc.vdisc.vth

    M0 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1y = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M1z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    M2 = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    H000 = @view f[:, :, :, 1, 1, 1]
    M0 .= H000

    if :vx in v_dims
        H100 = @view f[:, :, :, 2, 1, 1]
        @. M1x = H100 * vth
        H200 = @view f[:, :, :, 3, 1, 1]
        @. M2 += vth^2 * (sqrt(2) * H200 + H000)
    end
    if :vy in v_dims
        H010 = @view f[:, :, :, 1, 2, 1]
        @. M1y = H010 * vth
        H020 = @view f[:, :, :, 1, 3, 1]
        @. M2 += vth^2 * (sqrt(2) * H020 + H000)
    end
    if :vz in v_dims
        H001 = @view f[:, :, :, 1, 1, 2]
        @. M1z = H001 * vth
        H002 = @view f[:, :, :, 1, 1, 3]
        @. M2 += vth^2 * (sqrt(2) * H002 + H000)
    end

    M0, (M1x, M1y, M1z), M2
end

function heat_flux(f, disc::XVDiscretization{<:Hermite}, v_dims, buffer, (M0, M1, M2))
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    vth = disc.vdisc.vth


    # The moment of vx*|v|^2
    M3x = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    # The moment of vy*|v|^2
    M3y = alloc_zeros(Float64, buffer, Nx, Ny, Nz)
    # The moment of vz*|v|^2
    M3z = alloc_zeros(Float64, buffer, Nx, Ny, Nz)

    if :vx in v_dims
        H100 = @view f[:, :, :, 2, 1, 1]
        H200 = @view f[:, :, :, 3, 1, 1]
        H300 = @view f[:, :, :, 4, 1, 1]

        # vx^3 term
        @. M3x += vth^3 * (sqrt(6) * H300 + H100)
        # vxvy^2 term

        @. M2 += vth^2 * (sqrt(2) * H200 + H000)
    end
end

function dyadic_moments(f, disc::XVDiscretization{<:Hermite}, v_dims, buffer)
    dyadic_moments_hermite(f, disc.vdisc.vth, v_dims, buffer)
end

# Computes the sequence of dyadic moment tensors up to order 3.
function dyadic_moments_hermite(f, vth, v_dims, buffer)
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    f = reshape(f, (:, Nvx, Nvy, Nvz))

    M0 = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    M1 = alloc_zeros(Float64, buffer, 3, Nx*Ny*Nz)
    M2 = alloc_zeros(Float64, buffer, 3, 3, Nx*Ny*Nz)
    M3 = alloc_zeros(Float64, buffer, 3, 3, 3, Nx*Ny*Nz)
    
    H000 = @view f[:, 1, 1, 1]
    M0 .= H000

    #display(f)

    if :vx in v_dims
        H100 = @view f[:, 2, 1, 1]
        #display(H100)
        @. M1[1, :] = H100 * vth

        H200 = @view f[:, 3, 1, 1]
        @. M2[1, 1, :] = vth^2 * (sqrt(2) * H200 + H000)

        H300 = @view f[:, 4, 1, 1]
        @. M3[1, 1, 1, :] = vth^3 * (sqrt(6) * H300 + 3*H100)
    end
    if :vy in v_dims
        H010 = @view f[:, 1, 2, 1]
        M1[2, :] .= H010 .* vth

        H020 = @view f[:, 1, 3, 1]
        @. M2[2, 2, :] = vth^2 * (sqrt(2) * H020 + H000)

        H030 = @view f[:, 1, 4, 1]
        @. M3[2, 2, 2, :] = vth^3 * (sqrt(6) * H030 + 3*H010)
    end
    if :vz in v_dims
        H001 = @view f[:, 1, 1, 2]
        M1[3, :] .= H001 .* vth

        H002 = @view f[:, 1, 1, 3]
        @. M2[3, 3, :] = vth^2 * (sqrt(2) * H002 + H000)

        H003 = @view f[:, 1, 1, 4]
        @. M3[3, 3, 3, :] = vth^3 * (sqrt(6) * H003 + 3*H001)
    end

    if :vx in v_dims && :vy in v_dims
        H110 = @view f[:, 2, 2, 1]
        @. M2[1, 2, :] = vth^2 * H110
        @. M2[2, 1, :] = vth^2 * H110

        H210 = @view f[:, 3, 2, 1]
        @. M3[1, 1, 2, :] = M3[1, 2, 1, :] = M3[2, 1, 1, :] = vth^3 * (sqrt(2) * H210 + H010)
        H120 = @view f[:, 2, 3, 1]
        @. M3[1, 2, 2, :] = M3[2, 1, 2, :] = M3[2, 2, 1, :] = vth^3 * (sqrt(2) * H120 + H100)
    end
    if :vx in v_dims && :vz in v_dims
        H101 = @view f[:, 2, 1, 2]
        @. M2[1, 3, :] = vth^2 * H101
        @. M2[3, 1, :] = vth^2 * H101

        H201 = @view f[:, 3, 1, 2]
        @. M3[1, 1, 3, :] = M3[1, 3, 1, :] = M3[3, 1, 1, :] = vth^3 * (sqrt(2) * H201 + H001)
        H102 = @view f[:, 2, 1, 3]
        @. M3[1, 3, 3, :] = M3[3, 1, 3, :] = M3[3, 3, 1, :] = vth^3 * (sqrt(2) * H102 + H100)
    end
    if :vy in v_dims && :vz in v_dims
        H011 = @view f[:, 1, 2, 2]
        @. M2[2, 3, :] = vth^2 * H011
        @. M2[3, 2, :] = vth^2 * H011

        H012 = @view f[:, 1, 2, 3]
        @. M3[2, 2, 3, :] = M3[2, 3, 2, :] = M3[3, 2, 2, :] = vth^3 * (sqrt(2) * H012 + H010)
        H021 = @view f[:, 1, 3, 2]
        @. M3[2, 3, 3, :] = M3[3, 2, 3, :] = M3[3, 3, 2, :] = vth^3 * (sqrt(2) * H021 + H001)
    end

    if :vx in v_dims && :vy in v_dims && :vz in v_dims
        H111 = @view f[:, 2, 2, 2]
        @. M3[1, 2, 3, :] = M3[2, 3, 1, :] = M3[3, 1, 2, :] = vth^3 * H111
        @. M3[3, 2, 1, :] = M3[2, 1, 3, :] = M3[1, 3, 2, :] = vth^3 * H111
    end

    return M0, M1, M2, M3
end

function moments_for_wsindy(f, disc::XVDiscretization{<:Hermite}, v_dims, buffer)
    moments_for_wsindy(f, disc.vdisc.vth, v_dims, buffer)
end

function moments_for_wsindy(f, vth, v_dims, buffer)
    M0, M1, M2, M3 = dyadic_moments_hermite(f, vth, v_dims, buffer)

    d = length(v_dims)

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(f)

    n = alloc_zeros(Float64, buffer, Nx*Ny*Nz)

    u_x = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    u_y = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    u_z = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    T = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_xx = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_yy = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_zz = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_xy = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_xz = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    Pi_yz = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    q_x = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    q_y = alloc_zeros(Float64, buffer, Nx*Ny*Nz)
    q_z = alloc_zeros(Float64, buffer, Nx*Ny*Nz)

    @. n = M0

    @. u_x = (@view M1[1, :]) / M0
    @. u_y = (@view M1[2, :]) / M0
    @. u_z = (@view M1[3, :]) / M0

    Tr_M2 = @. @views (M2[1, 1, :] + M2[2, 2, :] + M2[3, 3, :])
    u2 = @. u_x^2 + u_y^2 + u_z^2

    @. T = (Tr_M2 - M0 * u2) / (d * M0)

    # Full pressure tensor
    #Pxx = @. (@view M2[1, 1, :]) - n * u_x^2
    #Pyy = @. (@view M2[2, 2, :]) - n * u_y^2
    #Pzz = @. (@view M2[3, 3, :]) - n * u_z^2
    #Pxy = @. (@view M2[1, 2, :]) - n * u_x * u_y
    #Pxz = @. (@view M2[1, 3, :]) - n * u_x * u_z
    #Pyz = @. (@view M2[2, 3, :]) - n * u_y * u_z

    # Deviatoric part of pressure tensor
    @. Pi_xx = (@view M2[1, 1, :]) - n * u_x^2 - n * T
    @. Pi_yy = (@view M2[2, 2, :]) - n * u_y^2 - n * T
    @. Pi_zz = (@view M2[3, 3, :]) - n * u_z^2 - n * T
    @. Pi_xy = (@view M2[1, 2, :]) - n * u_x * u_y
    @. Pi_xz = (@view M2[1, 3, :]) - n * u_x * u_z
    @. Pi_yz = (@view M2[2, 3, :]) - n * u_y * u_z

    Q3(comp) = @. @views (M3[comp, 1, 1, :] + M3[comp, 2, 2, :] + M3[comp, 3, 3, :])
    M2_dot_u(comp) = @. @views (M2[comp, 1, :] * u_x + M2[comp, 2, :] * u_y + M2[comp, 3, :] * u_z)

    @. q_x = $Q3(1) - 2 * $M2_dot_u(1) - u_x * Tr_M2 + 2 * M0 * u_x * u2
    @. q_y = $Q3(2) - 2 * $M2_dot_u(2) - u_y * Tr_M2 + 2 * M0 * u_y * u2
    #@. q_z = $Q3(3) - 2 * $M2_dot_u(3) - u_z * Tr_M2 + 2 * M0 * u_z * u2
    #
    @. q_z = @views M3[3, 3, 3, :] - 3*u_z * M2[3, 3, :] + 2*u_z^3 * M0

    #@info "" hcat(Q3(3), 2*M2_dot_u(3), u_z .* Tr_M2, 2*M0.*u_z .* u2)[100, :]
    #@info "" vec(q_z)[100]

    re(A) = reshape(A, (Nx, Ny, Nz))

    return (
        re(n), 
        re(u_x), re(u_y), re(u_z), 
        re(T), 
        re(Pi_xx), re(Pi_yy), re(Pi_zz),
        re(Pi_xy), re(Pi_xz), re(Pi_yz),
        re(q_x), re(q_y), re(q_z)
    )
end
