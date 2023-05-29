@testset "Electrostatic" begin

    @testset "Cyclotron rotation" begin
        dt = 0.01

        T = π/4
        f0(vx, vy) = exp(-((vx - 1)^2 + (vy - 1)^2) / 2)
        Bz = 1.0

        errors = Float64[]
        Ns = [20, 40, 80]
        for N in Ns
            sim = single_species_0d2v((; f=f0, Bz), N, N, 6.5, 6.5)
            (; VX, VY) = sim.species[1].grid

            runsim_lightweight!(sim, T, dt)
            actual = as_vxvy(sim.u.x[1])

            expected_f(vx, vy) = exp(-((vx-√(2))^2 + vy^2) / 2)
            expected = expected_f.(vec(VX), vec(VY)')

            error = norm(actual - expected) / norm(expected)

            push!(errors, error)
        end

        display(errors)
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ >= -5
    end

end
