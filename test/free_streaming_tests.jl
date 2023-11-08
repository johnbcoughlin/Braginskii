@testset "Free streaming" begin
    @testset "z free streaming" begin
        @testset "reflection" begin
            @no_escape begin
            for device in supported_devices(), vdisc in [:hermite]
            dt = 0.001
            T = 0.2
            f0(z, vz) = (0.1 + 0.8exp(-(z-0.3)^2/0.01)) * (exp(-(vz-1.5)^2/2) + exp(-(vz+1.5)^2/2))

            characteristic_z(z, vz) = begin
                if -1 <= (z - vz*T) <= 1
                    z - vz*T
                elseif (z - vz*T) >= 1
                    t = T - (z - 1) / vz
                    1 + vz*t
                else # if (z - vz*T) <= -1
                    t = T - (z + 1) / vz
                    -1 + vz*t
                end
            end
            characteristic_vz(z, vz) = begin
                if -1 <= (z - vz*T) <= 1
                    vz
                elseif (z - vz*T) >= 1
                    t = T - (z - 1) / vz
                    -vz
                else #if (z - vz*T) <= -1
                    t = T - (z + 1) / vz
                    -vz
                end
            end

            errors = Float64[]
            Ns = [20, 40, 60] .* 2
            Nvz = 150
            for Nz in Ns
                sim = single_species_1d1v_z(f0; Nz, Nvz, q=0.0, vdisc, device, z_bcs=:reflecting)

                runsim_lightweight!(sim, T, dt / (Nz / 80 * Nvz / 100))

                disc = sim.species[1].discretization

                vgrid = vgrid_of(disc.vdisc, 50, 8.0)
                actual = expand_f(sim.u.x[1], disc, vgrid) |> as_zvz

                (; Z) = sim.species[1].discretization.x_grid
                (; VZ) = vgrid
                # Iterating like characteristic(z, vz)... not supported by CUDA
                expected = ((z, vz) -> f0(characteristic_z(z, vz), characteristic_vz(z, vz))).(Z, VZ)
                expected = as_zvz(expected)
                
                error = norm(expected - actual) / norm(expected)
                @show error
                push!(errors, error)
            end

            @show errors

            γ = estimate_log_slope(Ns, errors)
            @test -3 >= γ >= -5
            end
            end
        end

        @testset "reservoir" begin
            @no_escape begin
            for device in supported_devices(), vdisc in [:hermite]
            dt = 0.001
            T = 0.2
            f0(z, vz) = (0.1 + 0.8exp(-(z-0.3)^2/0.01)) * (exp(-(vz-1.5)^2/2) + exp(-(vz+1.5)^2/2))

            errors = Float64[]
            Ns = [20, 40, 60] .* 2
            Nvz = 150
            for Nz in Ns
                sim = single_species_1d1v_z(f0; Nz, Nvz, q=0.0, vdisc, device, z_bcs=:reservoir)

                runsim_lightweight!(sim, T, dt / (Nz / 80 * Nvz / 100))

                disc = sim.species[1].discretization

                vgrid = vgrid_of(disc.vdisc, 50, 8.0)
                actual = expand_f(sim.u.x[1], disc, vgrid) |> as_zvz

                (; Z) = sim.species[1].discretization.x_grid
                (; VZ) = vgrid
                # Iterating like characteristic(z, vz)... not supported by CUDA
                expected = f0.(Z .- VZ*T, VZ)
                expected = as_zvz(expected)
                
                error = norm(expected - actual) / norm(expected)
                @show error
                push!(errors, error)
            end

            @show errors

            γ = estimate_log_slope(Ns, errors)
            @test -3 >= γ >= -5
            end
            end
        end
    end

    @testset "x free streaming" begin
        @no_escape begin
        for device in supported_devices(), vdisc in [:weno, :hermite]
        Nx = 48
        dt = 0.01
        T = 1.0
        n(x) = 1 + 0.2*exp((sin(x) + 0cos(2x)))
        sim = single_species_1d1v_x(; Nx, Nvx=80, Lx=4pi, vdisc, q=0.0, device) do x, vx
            n(x) * exp(-vx^2/2)
        end
        actual0 = copy(sim.u.x[1])
        runsim_lightweight!(sim, T, dt)

        disc = sim.species[1].discretization

        vgrid = vgrid_of(disc.vdisc, 50)
        actual = expand_f(sim.u.x[1], disc, vgrid)
        (; X) = sim.species[1].discretization.x_grid
        (; VX) = vgrid

        expected = (n.(X .- VX*T) .* exp.(-(VX).^2 ./ 2))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-6
        end
        end
    end

    @testset "y free streaming" begin
        @no_escape begin
        for device in supported_devices(), vdisc in [:weno, :hermite]
        Ny = 32
        dt = 0.01
        T = 1.0
        n(y) = 1 + 0.2*exp((sin(y) + 0cos(2y)))
        sim = single_species_1d1v_y(; Ny, Nvy=60, Ly=4pi, vdisc, q=0.0, device) do y, vy
            n(y) * exp(-vy^2/2)
        end
        disc = sim.species[1].discretization
        vgrid = vgrid_of(disc.vdisc, 50)

        actual0 = expand_f(copy(sim.u.x[1]), disc, vgrid)
        runsim_lightweight!(sim, T, dt)

        actual = expand_f(copy(sim.u.x[1]), disc, vgrid)
        (; Y) = disc.x_grid
        (; VY) = vgrid

        expected = (n.(Y .- VY*T) .* exp.(-(VY).^2 ./ 2))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-6
        end
        end
    end

end
