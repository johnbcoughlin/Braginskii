@testset "Hermite expansions" begin
    @testset "X-VX" begin
        xgrid = x_grid_3d(10, 1, 1)
        vdisc = hermite_disc(; Nvx=20)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(x, vx) = 1 / sqrt(2π) * exp(-vx^2/2)
        actual = approximate_f(f, disc, (1, 4)) |> as_xvx
        expected = zeros(10, 20)
        expected[:, 1] .= 1.0
        @test actual ≈ expected
    end

    @testset "Y-VY" begin
        xgrid = x_grid_3d(1, 10, 1)
        vdisc = hermite_disc(; Nvy=20)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(y, vy) = 1 / sqrt(2π) * exp(-vy^2/2)
        actual = approximate_f(f, disc, (2, 5)) |> as_yvy
        expected = zeros(10, 20)
        expected[:, 1] .= 1.0
        @test actual ≈ expected
    end

    @testset "XY-VXVY" begin
        xgrid = x_grid_3d(10, 10, 1)
        vdisc = hermite_disc(; Nvx=20, Nvy=20)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(x, y, vx, vy) = 1 / 2π * exp(-(vx^2 + vy^2)/2)
        actual = approximate_f(f, disc, (1, 2, 4, 5)) |> as_xyvxvy
        expected = zeros(10, 10, 20, 20)
        expected[:, :, 1, 1] .= 1.0
        @test actual ≈ expected
    end
end
