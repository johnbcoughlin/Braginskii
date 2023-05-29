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
            Braginskii.apply_laplacian!(actual, ϕ, ϕl, ϕr, grid, [:x, :y, :z], default_buffer())

            ϕ_xx = -π^2 * sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_yy = sin.(π * grid.X) .* -9cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_zz = sin.(π * grid.X) .* cos.(3grid.Y) .* -25cos.(5grid.Z)
            expected = ϕ_xx + ϕ_yy + ϕ_zz
            
            error = norm(actual - expected) / norm(expected)
            push!(errors, error)
        end
        γ = estimate_log_slope(Ns, errors)
        @test -3 >= γ
    end

    @testset "Poisson solve" begin
        set_default_buffer_size!(100_000_000)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nx in Ns
            grid = x_grid_3d(Nx, 20, 20)

            ϕ_xx = -π^2 * sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_yy = sin.(π * grid.X) .* -9cos.(3grid.Y) .* cos.(5grid.Z)
            ϕ_zz = sin.(π * grid.X) .* cos.(3grid.Y) .* -25cos.(5grid.Z)
            ρ = ϕ_xx + ϕ_yy + ϕ_zz

            ϕ_left(x, y) = 0
            ϕ_right(x, y) = 0

            ϕl = ϕ_left.(grid.Y, grid.Z)
            ϕr = ϕ_right.(grid.Y, grid.Z)

            Ex, Ey, Ez = Braginskii.poisson(ρ, ϕl, ϕr, grid, [:x, :y, :z], default_buffer())

            Ex_expected = π * cos.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)

            Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)
            push!(errors, Ex_error)
        end
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ
    end
end
