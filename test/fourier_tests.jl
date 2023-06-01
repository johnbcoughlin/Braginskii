@testset "dealias test" begin

    @testset "round trip" begin
        g(x) = exp(sin(x))
        N = 16
        x = reshape(collect(LinRange(0, 2π, N)), (1, N, 1))
        
        there = Braginskii.quadratic_dealias(g.(x), default_buffer())
        back = Braginskii.reverse_quadratic_dealias(there, default_buffer())

        error = norm(back .- g.(x)) / norm(g.(x))
        @test back ≈ g.(x)
    end

    @testset "quadratic" begin
        ref_grid = reshape(collect(LinRange(0, 2π-2π/1000, 1000)), (1, 1000, 1))

        g(x) = exp(sin(x))
        h(x) = 1 + cos(x)*cos(5x)
        gh(x) = g(x) * h(x)

        for N in [16, 20, 24, 30]
            x = reshape(collect(LinRange(0, 2π*(N-1)/N, N)), (1, N, 1))
            x_double = reshape(collect(LinRange(0, 2π*(2N-2)/(2N-1), 2N-1)), (1, 2N-1, 1))
            g_nodes = g.(x)
            h_nodes = h.(x)
            ĝ = Braginskii.quadratic_dealias(g.(x), default_buffer())

            @test isapprox(ĝ, g.(x_double), rtol=1e-6)

            ĥ = Braginskii.quadratic_dealias(h.(x), default_buffer())

            gĥ = Braginskii.reverse_quadratic_dealias(ĝ .* ĥ, default_buffer())

            ref_rfft_modes = rfft(gh.(ref_grid), (2,))
            ref_truncated_modes = ref_rfft_modes[:, 1:N÷2+1, :]
            ref_nodes = irfft(ref_truncated_modes, N, (2,)) * N / 1000

            error_from_direct = norm(ref_nodes .- gh.(x)) / norm(ref_nodes)
            error_from_dealiased = norm(ref_nodes .- gĥ) / norm(ref_nodes)

            @test error_from_dealiased < 1e-7
            @test error_from_direct >= error_from_dealiased * 1e4
        end
    end
end
