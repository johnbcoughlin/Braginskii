@testset "Dougherty-Fokker-Planck" begin
    @testset "Maxwellian ratio" begin
        u = rand()
        T = 1 + rand()
        vi = 6.3
        dv = 0.2

        M1 = exp(-(vi - u)^2/2T)
        M2 = exp(-(vi+dv - u)^2/2T)

        actual = Braginskii.M_ratio(u, T, vi, dv)
        expected = M2 / M1
        @test actual ≈ expected
    end

    @testset "Holds Maxwellian equilibrium" begin
        for device in supported_devices(), vdisc in [:weno, :hermite]
        Nx = 40
        Nvx = 80
        temp(x) = 1.0 + 0.3*sin(x)
        u(x) = 0.3*cos(x)
        sim = single_species_1d1v_x(; Nx, Nvx, vxmax=10.0, free_streaming=false, ν_p=1.0, vdisc, vth=1.0) do x, vx
            1.0 / sqrt(2π*temp(x)) * exp(-(vx-u(x))^2/(2*temp(x)))
        end

        disc = sim.species[1].discretization

        vgrid = vgrid_of(disc.vdisc, 50)
        #display(sim.u.x[1])
        actual0 = expand_f(copy(sim.u.x[1]), disc, vgrid) |> as_xvx

        dt = 0.001
        T = dt*100
        df = runsim_lightweight!(sim, T, dt)
        actual = expand_f(copy(sim.u.x[1]), disc, vgrid) |> as_xvx

        @show (norm(actual0 - actual) / norm(actual))
        @test actual0 ≈ actual
        end
    end

    @testset "Conserves energy" begin
        for device in supported_devices(), vdisc in [:weno, :hermite]
        Nz = 40
        Nvz = 100
        sim = single_species_1d1v_z(; Nz, Nvz, zmin=-π, zmax=π, vzmax=10.0, free_streaming=false, ν_p=1.0, vdisc,
        vth=1.0) do z, vz
            0.5 / sqrt(2π) * (exp(-(vz-1-sin(z))^2/2) + 0*exp(-((vz+1)^2/2)))
        end

        disc = sim.species[1].discretization

        buffer = default_buffer()

        T = 1.0
        dt = 0.002
        M0_0, (_, _, M1z_0), M2_0 = Braginskii.moments(sim.u.x[1], disc, [:vz], buffer)
        df = runsim_lightweight!(sim, T, dt)
        M0, (_, _, M1z), M2 = Braginskii.moments(sim.u.x[1], disc, [:vz], buffer)

        M0_error = norm(M0_0 - M0) / norm(M0_0)
        M1_error = norm(M1z_0 - M1z) / norm(M1z_0)
        M2_error = norm(M2_0 - M2) / norm(M2_0)
        if vdisc == :weno
            @test M0_error < 1e-6
            @test M1_error < 1e-6
            @test M2_error < 1e-5
        end
        if vdisc == :hermite
            @test M0_error < eps()
            @test M1_error < eps()
            @test M2_error < eps()
        end
    end
    end
end
