function dfp!(df, f, α::Species, sim, buffer)
    for β in sim.species
        cm = sim.collisional_moments[(α.name, β.name)]
        dfp!(df, f, cm, α, buffer)
    end
end

function dfp!(df, f, cm::CollisionalMoments, species::Species, buffer)
    (; ux, uy, uz, T, ν) = cm
    @no_escape buffer begin
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
        df .+= df_dfp
        nothing
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

function dfp_vy!(df, f, uy, T, ν, species, buffer)
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

# Compute Mᵢ±½ / Mᵢ
function M_ratio(u, T, vᵢ, dv_inc)
    vᵢ_inc = vᵢ + dv_inc
    return exp((vᵢ - vᵢ_inc) * (vᵢ + vᵢ_inc - 2u) / 2T)
end
