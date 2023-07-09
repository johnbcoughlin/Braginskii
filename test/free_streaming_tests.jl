@testset "Free streaming" begin

    @testset "x free streaming" begin
        @testset "reflection" begin
            dt = 0.001
            T = 0.2
            f0(x, vx) = (0.1 + 0.8exp(-x^2/0.01)) * (exp(-(vx-1.5)^2/2) + exp(-(vx+1.5)^2/2))

            characteristic(x, vx) = begin
                if -1 <= (x - vx*T) <= 1
                    (x - vx*T, vx)
                elseif (x - vx*T) >= 1
                    t = T - (x - 1) / vx
                    (1 + vx*t, -vx)
                elseif (x - vx*T) <= -1
                    t = T - (x + 1) / vx
                    (-1 + vx*t, -vx)
                end
            end

            errors = Float64[]
            Ns = [20, 40, 80] .* 4
            @no_escape begin
                for Nx in Ns
                    sim = single_species_1d1v_x(f0; Nx, Nvx=20, q=0.0, vdisc=:weno)

                    actual0 = as_xvx(sim.u.x[1])
                    runsim_lightweight!(sim, T, dt)
                    actual = as_xvx(sim.u.x[1])

                    (; X) = sim.species[1].discretization.x_grid
                    (; VX) = sim.species[1].discretization.vdisc.grid
                    expected = ((x, vx) -> f0(characteristic(x, vx)...)).(X, VX)
                    expected = as_xvx(expected)
                    
                    error = norm(expected - actual) / norm(expected)
                    push!(errors, error)
                end
            end

            γ = estimate_log_slope(Ns, errors)
            @test -4 >= γ >= -5
        end
    end

    @testset "y free streaming" begin
        Ny = 32
        dt = 0.001
        T = 1.0
        n(y) = 1 + 0.2*exp((sin(y) + 0cos(2y)))
        sim = single_species_1d1v_y(; Ny, Nvy=20, Ly=4pi, vdisc=:weno, q=0.0) do y, vy
            n(y) * exp(-vy^2/2)
        end
        actual0 = sim.u.x[1]
        runsim_lightweight!(sim, T, dt)
        actual = sim.u.x[1]
        (; Y) = sim.species[1].discretization.x_grid
        (; VY) = sim.species[1].discretization.vdisc.grid

        expected = (n.(Y .- VY*T) .* exp.(-(VY).^2 ./ 2))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-7
    end

end
