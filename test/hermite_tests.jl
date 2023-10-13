@testset "Hermite expansions" begin
    @testset "X-VX" begin
        xgrid = x_grid_3d(10, 1, 1)
        vdisc = hermite_disc(; Nvx=20, device=:cpu)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(x, vx) = 1 / sqrt(2π) * exp(-vx^2/2)
        coefs = approximate_f(f, disc, (1, 4), allocator(:cpu))
        expected = zeros(10, 20)
        expected[:, 1] .= 1.0
        @test as_xvx(coefs) ≈ expected

        test_grid = vgrid_of(vdisc, 50, 5.0, default_buffer())
        actual = expand_f(coefs, disc, test_grid)

        @test as_xvx(actual) ≈ f.(xgrid.x.nodes, test_grid.x.nodes')
    end

    @testset "Y-VY" begin
        xgrid = x_grid_3d(1, 10, 1)
        vdisc = hermite_disc(; Nvy=20, device=:cpu)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(y, vy) = 1 / sqrt(2π) * exp(-vy^2/2)
        coefs = approximate_f(f, disc, (2, 5), allocator(:cpu))
        expected = zeros(10, 20)
        expected[:, 1] .= 1.0
        @test as_yvy(coefs) ≈ expected

        test_grid = vgrid_of(vdisc, 50, 5.0, default_buffer())
        actual = expand_f(coefs, disc, test_grid)

        @test as_yvy(actual) ≈ f.(xgrid.y.nodes, test_grid.y.nodes')
    end

    @testset "XY-VXVY" begin
        xgrid = x_grid_3d(10, 10, 1)
        vdisc = hermite_disc(; Nvx=20, Nvy=20, device=:cpu)

        disc = Braginskii.XVDiscretization(xgrid, vdisc)

        # Trivial Maxwellian example
        f(x, y, vx, vy) = 1 / 2π * exp(-(vx^2 + vy^2)/2)
        coefs = approximate_f(f, disc, (1, 2, 4, 5), allocator(:cpu))
        expected = zeros(10, 10, 20, 20)
        expected[:, :, 1, 1] .= 1.0
        @test as_xyvxvy(coefs) ≈ expected

        test_grid =  vgrid_of(vdisc, 20, 5.0, default_buffer())
        actual = expand_f(coefs, disc, test_grid)

        (; X, Y) = xgrid
        (; VX, VY) = test_grid

        @test as_xyvxvy(actual) ≈ as_xyvxvy(f.(X, Y, VX, VY))
    end
end
