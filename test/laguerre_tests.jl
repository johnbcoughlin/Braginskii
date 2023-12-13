@testset "Hermite-Laguerre expansions" begin
    @testset "X-μ" begin
        buffer = allocator(:cpu)
        xgrid = x_grid_1d(10, 2π, buffer)
        gyro_v_disc = hermite_laguerre_disc(; Nμ=20, device=:cpu, μ0=1.2)

        disc = Braginskii.XVDiscretization(xgrid, gyro_v_disc)
        vgrid = vgrid_of(gyro_v_disc)

        T(x) = 1 + 0.1*sin(x)
        f(x, μ) = exp(-μ / T(x))
        coefs = approximate_f(f, disc, [1, 4], buffer)

        expected = f.(xgrid.x.nodes, vgrid.μ.nodes')
        actual = expand_bigfloat_laguerre_f(coefs, vgrid, gyro_v_disc.μ0, gyro_v_disc.vth) |> as_xvx

        @test actual ≈ expected
    end
end
