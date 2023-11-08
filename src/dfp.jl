function dfp!(df, f, α::Species, sim, buffer)
    for β in sim.species
        cm = sim.collisional_moments[(α.name, β.name)]
        dfp!(df, f, cm, α, buffer)
    end
end

function dfp!(df, f, cm::CollisionalMoments, species::Species, buffer)
    (; ux, uy, uz, T, ν) = cm
    no_escape(buffer) do
        df_dfp = alloc_zeros(Float64, buffer, size(df)...)
        if :vx ∈ species.v_dims
            dfp_vx!(df_dfp, f, ux, T, ν, species, buffer)
        end
        if :vy ∈ species.v_dims
            dfp_vy!(df_dfp, f, uy, T, ν, species, buffer)
        end
        if :vz ∈ species.v_dims
            dfp_vz!(df_dfp, f, uz, T, ν, species, buffer)
        end
        #display(as_xvx(df_dfp)')
        df .+= df_dfp
        nothing
    end
end

function dfp_vx!(df, f, ux, T, ν, species::Species{<:Hermite}, buffer)
    (; discretization) = species
    (; Dvx, Ξx) = discretization.vdisc
    dfp_vi!(df, f, ux, T, ν, Ξx, Dvx, species, buffer)
end

function dfp_vy!(df, f, uy, T, ν, species::Species{<:Hermite}, buffer)
    (; discretization) = species
    (; Dvy, Ξy) = discretization.vdisc
    dfp_vi!(df, f, uy, T, ν, Ξy, Dvy, species, buffer)
end

function dfp_vz!(df, f, uz, T, ν, species::Species{<:Hermite}, buffer)
    (; discretization) = species
    (; Dvz, Ξz) = discretization.vdisc
    dfp_vi!(df, f, uz, T, ν, Ξz, Dvz, species, buffer)
end

function dfp_vi!(df, f, u, T, ν, Ξ, Dv, species::Species{<:Hermite}, buffer)
    (; discretization) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    vth = discretization.vdisc.vth

    NX = Nx*Ny*Nz
    NV = Nvx*Nvy*Nvz

    f = reshape(f, (NX, NV))
    u = vec(u)
    T = vec(T)

    no_escape(buffer) do
        u_f = alloc_array(Float64, buffer, NX, NV)
        @. u_f = u * f
        v_f = alloc_array(Float64, buffer, NX, NV)
        mul!(v_f, f, Ξ')

        T_Dv_f = alloc_array(Float64, buffer, NX, NV)
        mul!(T_Dv_f, f, Dv', 1/vth, 0.0)
        @. T_Dv_f *= T

        L = alloc_array(Float64, buffer, NX, NV)
        @. L = T_Dv_f + v_f - u_f

        Dv_L = alloc_array(Float64, buffer, NX, NV)
        mul!(Dv_L, L, Dv', 1/vth, 0.0)

        Dv_L = reshape(Dv_L, size(discretization))
        @. df += ν * Dv_L
    end
end

function dfp_vx!(df, f, ux, T, ν, species::Species{WENO5}, buffer)
    (; discretization) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    vgrid = discretization.vdisc.grid
    dvx = vgrid.x.dx

    fM₋½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
    fM₊½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            vx = vgrid.VX[λvx]

            fM₋½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(ux[λxyz], T[λxyz], vx, -dvx/2)
            fM₊½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(ux[λxyz], T[λxyz], vx, dvx/2)
        end
    end

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvy in 1:Nvy, λvz in 1:Nvz
            for λvx in 2:Nvx-1
                df[λxyz, λvx, λvy, λvz] += (fM₋½[λxyz, λvx+1, λvy, λvz] - fM₋½[λxyz, λvx, λvy, λvz]) / dvx^2
                df[λxyz, λvx, λvy, λvz] += (fM₊½[λxyz, λvx-1, λvy, λvz] - fM₊½[λxyz, λvx, λvy, λvz]) / dvx^2
            end

            df[λxyz, 1, λvy, λvz] += (fM₋½[λxyz, 2, λvy, λvz] - fM₋½[λxyz, 1, λvy, λvz]) / dvx^2
            df[λxyz, Nvx, λvy, λvz] += (fM₊½[λxyz, Nvx-1, λvy, λvz] - fM₊½[λxyz, Nvx, λvy, λvz]) / dvx^2

            left = f[λxyz, 1, λvy, λvz] * M_ratio(ux[λxyz], T[λxyz], vgrid.VX[1], -dvx/2)
            df[λxyz, 1, λvy, λvz] += (left - fM₊½[λxyz, 1, λvy, λvz]) / dvx^2

            right = f[λxyz, Nvx, λvy, λvz] * M_ratio(ux[λxyz], T[λxyz], vgrid.VX[Nvx], dvx/2)
            df[λxyz, Nvx, λvy, λvz] += (right - fM₋½[λxyz, Nvx, λvy, λvz]) / dvx^2
        end
    end
    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvxyz in CartesianIndices((Nvx, Nvy, Nvz))
            df[λxyz, λvxyz] *= ν[λxyz] * T[λxyz]
        end
    end
    df
end

function dfp_vy!(df, f, uy, T, ν, species::Species{WENO5}, buffer)
    (; discretization) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    vgrid = discretization.vdisc.grid
    dvy = vgrid.y.dx

    fM₋½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
    fM₊½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            vy = vgrid.VY[λvy]

            fM₋½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(uy[λxyz], T[λxyz], vy, -dvy/2)
            fM₊½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(uy[λxyz], T[λxyz], vy, dvy/2)
        end
    end

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvz in 1:Nvz
            for λvy in 2:Nvy-1
                df[λxyz, λvx, λvy, λvz] += (fM₋½[λxyz, λvx, λvy+1, λvz] - fM₋½[λxyz, λvx, λvy, λvz]) / dvy^2
                df[λxyz, λvx, λvy, λvz] += (fM₊½[λxyz, λvx, λvy-1, λvz] - fM₊½[λxyz, λvx, λvy, λvz]) / dvy^2
            end

            df[λxyz, λvx, 1, λvz] += (fM₋½[λxyz, λvx, 2, λvz] - fM₋½[λxyz, λvx, 1, λvz]) / dvy^2
            df[λxyz, λvx, Nvy, λvz] += (fM₊½[λxyz, λvx, Nvy-1, λvz] - fM₊½[λxyz, λvx, Nvy, λvz]) / dvy^2

            left = f[λxyz, λvx, 1, λvz] * M_ratio(uy[λxyz], T[λxyz], vgrid.VY[1], -dvy/2)
            df[λxyz, λvx, 1, λvz] += (left - fM₊½[λxyz, λvx, 1, λvz]) / dvy^2

            right = f[λxyz, λvx, Nvy, λvz] * M_ratio(uy[λxyz], T[λxyz], vgrid.VY[Nvy], dvy/2)
            df[λxyz, λvx, Nvy, λvz] += (right - fM₋½[λxyz, λvx, Nvy, λvz]) / dvy^2
        end
    end
    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvxyz in CartesianIndices((Nvx, Nvy, Nvz))
            df[λxyz, λvxyz] *= ν[λxyz] * T[λxyz]
        end
    end
    df
end

function dfp_vz!(df, f, uz, T, ν, species::Species{WENO5}, buffer)
    (; discretization) = species
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)
    vgrid = discretization.vdisc.grid
    dvz = vgrid.z.dx

    fM₋½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)
    fM₊½ = alloc_zeros(Float64, buffer, Nx, Ny, Nz, Nvx, Nvy, Nvz)

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy, λvz in 1:Nvz
            vz = vgrid.VZ[λvz]

            fM₋½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(uz[λxyz], T[λxyz], vz, -dvz/2)
            fM₊½[λxyz, λvx, λvy, λvz] = f[λxyz, λvx, λvy, λvz] * M_ratio(uz[λxyz], T[λxyz], vz, dvz/2)
        end
    end

    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvx in 1:Nvx, λvy in 1:Nvy
            for λvz in 2:Nvz-1
                df[λxyz, λvx, λvy, λvz] += (fM₋½[λxyz, λvx, λvy, λvz+1] - fM₋½[λxyz, λvx, λvy, λvz]) / dvz^2
                df[λxyz, λvx, λvy, λvz] += (fM₊½[λxyz, λvx, λvy, λvz-1] - fM₊½[λxyz, λvx, λvy, λvz]) / dvz^2
            end

            df[λxyz, λvx, λvy, 1] += (fM₋½[λxyz, λvx, λvy, 2] - fM₋½[λxyz, λvx, λvy, 1]) / dvz^2
            df[λxyz, λvx, λvy, Nvz] += (fM₊½[λxyz, λvx, λvy, Nvz-1] - fM₊½[λxyz, λvx, λvy, Nvz]) / dvz^2

            left = f[λxyz, λvx, λvy, 1] * M_ratio(uz[λxyz], T[λxyz], vgrid.VZ[1], -dvz/2)
            df[λxyz, λvx, λvy, 1] += (left - fM₊½[λxyz, λvx, λvy, 1]) / dvz^2

            right = f[λxyz, λvx, λvy, Nvz] * M_ratio(uz[λxyz], T[λxyz], vgrid.VZ[Nvz], dvz/2)
            df[λxyz, λvx, λvy, Nvz] += (right - fM₋½[λxyz, λvx, λvy, Nvz]) / dvz^2
        end
    end
    for λxyz in CartesianIndices((Nx, Ny, Nz))
        for λvxyz in CartesianIndices((Nvx, Nvy, Nvz))
            df[λxyz, λvxyz] *= ν[λxyz] * T[λxyz]
        end
    end
    df
end

# Compute Mᵢ±½ / Mᵢ
function M_ratio(u, T, vᵢ, dv_inc)
    vᵢ_inc = vᵢ + dv_inc
    return exp((vᵢ - vᵢ_inc) * (vᵢ + vᵢ_inc - 2u) / 2T)
end
