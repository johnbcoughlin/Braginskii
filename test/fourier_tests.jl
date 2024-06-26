@testset "dealias test" begin
    @testset "round trip 1D" begin
        g(x) = exp(sin(x))
        N = 16
        x = reshape(collect(LinRange(0, 2π, N)), (N, 1, 1))

        plans = Braginskii.plan_ffts(N, 1, 1, (), default_buffer())
        
        there = Braginskii.quadratic_dealias(g.(x), default_buffer(), plans)
        back = Braginskii.reverse_quadratic_dealias(there, default_buffer(), plans)

        error = norm(back .- g.(x)) / norm(g.(x))
        @test back ≈ g.(x)
    end

    @testset "quadratic" begin
        ref_grid = reshape(collect(LinRange(0, 2π-2π/1000, 1000)), (1000, 1, 1))

        g(x) = exp(sin(x))
        h(x) = 1 + cos(x)*cos(5x)
        gh(x) = g(x) * h(x)

        for N in [16, 20, 24, 30]
            plans = Braginskii.plan_ffts(N, 1, 1, (), default_buffer())

            x = reshape(collect(LinRange(0, 2π*(N-1)/N, N)), (N, 1, 1))
            x_double_1dim = reshape(collect(LinRange(0, 2π*(2N-2)/(2N-1), 2N-1)), (2N-1, 1, 1))
            x_double = hcat(x_double_1dim, x_double_1dim)
            g_nodes = g.(x)
            h_nodes = h.(x)
            ĝ = Braginskii.quadratic_dealias(g.(x), default_buffer(), plans)

            @test isapprox(ĝ, g.(x_double), rtol=1e-6)

            ĥ = Braginskii.quadratic_dealias(h.(x), default_buffer(), plans)

            gĥ = Braginskii.reverse_quadratic_dealias(ĝ .* ĥ, default_buffer(), plans)

            ref_rfft_modes = rfft(gh.(ref_grid), (1, 2))
            ref_truncated_modes = ref_rfft_modes[1:N÷2+1, 1, :]
            ref_nodes = irfft(ref_truncated_modes, N, (1, 2,)) * N / 1000

            error_from_direct = norm(ref_nodes .- gh.(x)) / norm(ref_nodes)
            error_from_dealiased = norm(ref_nodes .- gĥ) / norm(ref_nodes)

            @test error_from_dealiased < 1e-7
            @test error_from_direct >= error_from_dealiased * 1e4
        end
    end
end
