@testset "Poisson tests" begin
    @testset "Evaluate Laplacian" begin
        set_default_buffer_size!(100_000_000)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nx in Ns
            grid = x_grid_3d(Nx, 20, 20)

            ϕ = sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)

            ϕ_left(x, y) = 0
            ϕ_right(x, y) = 0

            ϕl = ϕ_left.(grid.Y, grid.Z)
            ϕr = ϕ_right.(grid.Y, grid.Z)

            actual = similar(ϕ)
            @time Braginskii.apply_laplacian!(actual, ϕ, ϕl, ϕr, grid, default_buffer())

            ϕ_xx = -π^2 * sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_yy = sin.(π * grid.X) .* -9cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_zz = sin.(π * grid.X) .* cos.(3grid.Y) .* -25cos.(5grid.Z)
            expected = ϕ_xx + ϕ_yy + ϕ_zz
            
            error = norm(actual - expected) / norm(expected)
            push!(errors, error)
        end
        @show γ = estimate_log_slope(Ns, errors)
        @test -3 >= γ
    end
end
