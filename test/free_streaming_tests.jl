@testset "Free streaming" begin
    @testset "z free streaming" begin
        @testset "reflection" begin
            @no_escape begin
            for device in supported_devices()
            dt = 0.001
            T = 0.2
            f0(z, vz) = (0.1 + 0.8exp(-z^2/0.01)) * (exp(-(vz-1.5)^2/2) + exp(-(vz+1.5)^2/2))

            characteristic(z, vz) = begin
                if -1 <= (z - vz*T) <= 1
                    (z - vz*T, vz)
                elseif (z - vz*T) >= 1
                    t = T - (z - 1) / vz
                    (1 + vz*t, -vz)
                elseif (z - vz*T) <= -1
                    t = T - (z + 1) / vz
                    (-1 + vz*t, -vz)
                end
            end

            errors = Float64[]
            Ns = [20, 40, 80] .* 4
            for Nz in Ns
                sim = single_species_1d1v_z(f0; Nz, Nvz=20, q=0.0, vdisc=:weno, device)

                actual0 = as_zvz(sim.u.x[1])
                runsim_lightweight!(sim, T, dt)
                actual = as_zvz(sim.u.x[1])

                (; Z) = sim.species[1].discretization.x_grid
                (; VZ) = sim.species[1].discretization.vdisc.grid
                expected = ((z, vz) -> f0(characteristic(z, vz)...)).(Z, VZ)
                expected = as_zvz(expected)
                
                error = norm(expected - actual) / norm(expected)
                push!(errors, error)
            end

            γ = estimate_log_slope(Ns, errors)
            @test -4 >= γ >= -5
            end
            end
        end
    end

    @testset "x free streaming" begin
        @no_escape begin
        for device in supported_devices()
        Nx = 32
        dt = 0.01
        T = 1.0
        n(x) = 1 + 0.2*exp((sin(x) + 0cos(2x)))
        sim = single_species_1d1v_x(; Nx, Nvx=20, Lx=4pi, vdisc=:weno, q=0.0, device) do x, vx
            n(x) * exp(-vx^2/2)
        end
        actual0 = copy(sim.u.x[1])
        runsim_lightweight!(sim, T, dt)
        actual = sim.u.x[1]
        (; X) = sim.species[1].discretization.x_grid
        (; VX) = sim.species[1].discretization.vdisc.grid

        display(as_xvx(actual0))
        display(as_xvx(actual))
        expected = (n.(X .- VX*T) .* exp.(-(VX).^2 ./ 2))
        display(as_xvx(expected))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-7
        end
        end
    end
    @testset "y free streaming" begin
        @no_escape begin
        for device in supported_devices()
        Ny = 32
        dt = 0.01
        T = 1.0
        n(y) = 1 + 0.2*exp((sin(y) + 0cos(2y)))
        sim = single_species_1d1v_y(; Ny, Nvy=20, Ly=4pi, vdisc=:weno, q=0.0, device) do y, vy
            n(y) * exp(-vy^2/2)
        end
        actual0 = copy(sim.u.x[1])
        runsim_lightweight!(sim, T, dt)
        actual = sim.u.x[1]
        (; Y) = sim.species[1].discretization.x_grid
        (; VY) = sim.species[1].discretization.vdisc.grid

        expected = (n.(Y .- VY*T) .* exp.(-(VY).^2 ./ 2))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-7
        end
        end
    end

end
