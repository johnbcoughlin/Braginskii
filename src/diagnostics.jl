function core_diagnostics(sim, t)
    return (; t, 
        electric_energy=electric_energy(sim), 
        kinetic_energy_z=kinetic_energy_z(sim))
end

function kinetic_energy_z(sim)
    α = sim.species[1]

    M0, M1, _ = moments(sim.u.x[1], α.discretization, α.v_dims, sim.buffer)
    M1x, M1y, M1z = M1
    return sum(M1z.^2 ./ M0) / 2
end

function electric_energy(sim)
    f = sim.u
    Ex, Ey, Ez = poisson(sim, f, allocator(sim.device))

    E2 = 0.0

    dxyz = 1.0
    if :x ∈ sim.x_dims
        dxyz *= sim.x_grid.x.dx
    end
    if :y ∈ sim.x_dims
        dxyz *= sim.x_grid.y.dx
    end
    if :z ∈ sim.x_dims
        dxyz *= sim.x_grid.z.dx
    end

    if :x ∈ sim.x_dims
        E2 += sum(Ex.^2) * dxyz
    end
    if :y ∈ sim.x_dims
        E2 += sum(Ey.^2) * dxyz
    end
    if :z ∈ sim.x_dims
        E2 += sum(Ez.^2) * dxyz
    end

    return E2 / 2
end
