@testset "Free streaming" begin

    @testset "x free streaming" begin
        @testset "reflection" begin
            Nx = 60
            dt = 0.01 / 2
            T = 0.4
            f0(x, vx) = (1.0 + 0.2exp(-x^2/.01)) * exp(-(vx-1)^2/2)
            sim = single_species_1d1v_x(f0, Nx, 20, -1., 1.)
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

            actual0 = as_xvx(sim.u.x[1])
            runsim_lightweight!(sim, T, dt)
            actual = as_xvx(sim.u.x[1])

            (; X, VX) = sim.species[1].grid
            expected = ((x, vx) -> f0(characteristic(x, vx)...)).(X, VX) |> as_xvx
            
            @show norm(expected - actual) / norm(expected)
        end
    end

    @testset "y free streaming" begin
        Ny = 16
        dt = 0.001
        T = 1.0
        n(y) = 1 + 0.2*exp((sin(y) + 0cos(2y)))
        sim = single_species_1d1v_y(Ny, 20) do y, vy
            n(y) * exp(-vy^2/2)
        end
        actual0 = sim.u.x[1]
        runsim_lightweight!(sim, T, dt)
        actual = sim.u.x[1]
        (; Y, VY) = sim.species[1].grid
        expected = (n.(Y .- VY*T) .* exp.(-(VY).^2 ./ 2))

        error = norm(actual - expected) / norm(expected)
        @test abs(error) < 1e-7
    end

end
