function core_diagnostics(sim, t)
    return (; t, electric_energy=electric_energy(sim))
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
