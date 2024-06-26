@testset "Electrostatic" begin
    @testset "Cyclotron rotation" begin
        @no_escape begin
        for device in supported_devices(), vdisc in [:hermite]
        dt = 0.01

        T = π/4
        f0(vx, vz) = exp(-((vx - 1)^2 + (vz - 1)^2) / 2)
        By = 1.0

        errors = Float64[]
        Ns = [30, 40, 80]

        for N in Ns
            sim = single_species_0d2v((; f=f0, By), N, N; 
                vxmax=6.5, vzmax=6.5, vdisc)

            runsim_lightweight!(sim, T, dt)

            disc = sim.species[1].discretization

            vgrid = vgrid_of(disc.vdisc, 50, 8.0)
            (; VX, VZ) = vgrid

            actual = expand_f(sim.u.x[1], disc, vgrid) |> as_vxvz

            regression_test_value = actual[N÷2, N÷2]
            @test 40 ∈ Ns
            if N == 40 && vdisc == :weno
                @test regression_test_value ≈ 0.28479738517275455
            end

            expected_f(vx, vz) = exp(-((vz-√(2))^2 + vx^2) / 2)
            expected = expected_f.(vec(VX), vec(VZ)')

            error = norm(actual - expected) / norm(expected)

            push!(errors, error)
        end

        if vdisc == :weno
            γ = estimate_log_slope(Ns, errors)
            @test -4 >= γ >= -5
        end
        if vdisc == :hermite
            @test all(errors .< 1e-6)
        end

        end
        end
    end
end
