function mul_by_vx!(dest, f, discretization::XVDiscretization{WENO5})
    @. dest = f * discretization.vdisc.grid.VX
end

function mul_by_vx!(dest, f, discretization::XVDiscretization{<:Hermite})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    f = reshape(f, (:, Nvx*Nvy*Nvz))
    dest = reshape(dest, (:, Nvx*Nvy*Nvz))

    mul!(dest, f, discretization.vdisc.Ξx', discretization.vdisc.vth, 0.0)
end

function mul_by_vy!(dest, f, discretization::XVDiscretization{WENO5})
    @. dest = f * discretization.vdisc.grid.VY
end

function mul_by_vy!(dest, f, discretization::XVDiscretization{<:Hermite})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    f = reshape(f, (:, Nvx*Nvy*Nvz))
    dest = reshape(dest, (:, Nvx*Nvy*Nvz))

    mul!(dest, f, discretization.vdisc.Ξy', discretization.vdisc.vth, 0.0)
end

function mul_by_vz!(dest, f, discretization::XVDiscretization{<:Hermite})
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(discretization)

    f = reshape(f, (:, Nvx*Nvy*Nvz))
    dest = reshape(dest, (:, Nvx*Nvy*Nvz))

    mul!(dest, f, discretization.vdisc.Ξz', discretization.vdisc.vth, 0.0)
end

