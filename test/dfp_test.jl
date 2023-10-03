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
        Nvx = 60
        sim = single_species_1d1v_x(; Nx, Nvx, vxmax=10.0, free_streaming=false, ν_p=1.0, vdisc) do x, vx
            1.0 / sqrt(2π * (1.2 + 0.1sin(π*x))) * exp(-(vx-0.3cos(x))^2/(2*(1.2+0.1sin(π*x))))
        end

        dt = 0.001
        T = 0.1
        actual0 = as_xvx(sim.u.x[1])
        df = runsim_lightweight!(sim, T, dt)
        actual = as_xvx(sim.u.x[1])
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
