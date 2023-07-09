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
        Nx = 40
        Nvx = 60
        sim = single_species_1d1v_x(; Nx, Nvx, vxmax=10.0, free_streaming=false, ν_p=1.0, vdisc=:weno) do x, vx
            1.0 / sqrt(2π * (1.2 + 0.1sin(π*x))) * exp(-(vx-0.3cos(x))^2/(2*(1.2+0.1sin(π*x))))
        end

        dt = 0.001
        T = 0.1
        actual0 = as_xvx(sim.u.x[1])
        df = runsim_lightweight!(sim, T, dt)
        actual = as_xvx(sim.u.x[1])
        @test actual0 ≈ actual
    end

    @testset "Conserves energy" begin
        Nx = 40
        Nvx = 200
        sim = single_species_1d1v_x(; Nx, Nvx, xmin=-π, xmax=π, vxmax=10.0, free_streaming=false, ν_p=1.0, vdisc=:weno) do x, vx
            0.5 / sqrt(2π) * (exp(-(vx-1-sin(x))^2/2) + exp(-((vx+1)^2/2)))
        end

        v = sim.species[1].discretization.vdisc.grid.VX |> vec

        T = 1.0
        dt = 0.002
        actual0 = as_xvx(sim.u.x[1])
        M2_0 = sum(actual0 .* (v').^2, dims=2)
        df = runsim_lightweight!(sim, T, dt)
        actual = as_xvx(sim.u.x[1])
        M2 = sum(actual .* (v').^2, dims=2)

        @show error = norm(M2_0 - M2) / norm(M2_0)
    end
end
