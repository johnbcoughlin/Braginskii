@testset "Electrostatic" begin
    @testset "Cyclotron rotation" begin
        @no_escape begin
        for device in supported_devices()
        dt = 0.01

        T = π/4
        f0(vx, vy) = exp(-((vx - 1)^2 + (vy - 1)^2) / 2)
        Bz = 1.0

        errors = Float64[]
        Ns = [20, 40, 80]

        for N in Ns
            sim = single_species_0d2v((; f=f0, Bz), N, N; vxmax=6.5, vymax=6.5, vdisc=:weno)
            (; VX, VY) = sim.species[1].discretization.vdisc.grid

            runsim_lightweight!(sim, T, dt)
            actual = as_vxvy(sim.u.x[1])

            regression_test_value = actual[N÷2, N÷2]
            if N == 40
                @test regression_test_value ≈ 0.28479738517275455
            end

            expected_f(vx, vy) = exp(-((vx-√(2))^2 + vy^2) / 2)
            expected = expected_f.(vec(VX), vec(VY)')

            error = norm(actual - expected) / norm(expected)

            push!(errors, error)
        end

        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ >= -5
        end
        end
    end
end
