@testset "Poisson tests" begin
    @testset "Evaluate Laplacian" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        device == :cpu && set_default_buffer_size!(200_000_000)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nx in Ns
            @no_escape begin
                grid = x_grid_3d(Nx, 20, 20)
                (; X, Y, Z) = grid
                fft_plans = Braginskii.plan_ffts(grid)

                ϕ = sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)

                ϕ_left(x, y) = 0
                ϕ_right(x, y) = 0

                ϕl = ϕ_left.(grid.Y, grid.Z)
                ϕr = ϕ_right.(grid.Y, grid.Z)

                actual = similar(ϕ)
                Braginskii.apply_laplacian!(actual, ϕ, ϕl, ϕr, grid, [:x, :y, :z], buffer, fft_plans)

                ϕ_xx = @. -π^2 * sin(π * X) * cos(3 * Y) * cos(5 * Z)
                ϕ_yy = @. sin(π * X) * -9 * cos(3 * Y) * cos(5* Z)
                ϕ_zz = @. sin(π * X) * cos(3 * Y) * -25 * cos(5 * Z)
                expected = ϕ_xx + ϕ_yy + ϕ_zz
                
                error = norm(actual - expected) / norm(expected)
                push!(errors, error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test -3 >= γ
        end
    end

    @testset "Poisson solve" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        device == :cpu && set_default_buffer_size!(200_000_000)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nx in Ns
            @no_escape begin
                grid = x_grid_3d(Nx, 20, 20)
                fft_plans = Braginskii.plan_ffts(grid)

                ϕ_xx = -π^2 * sin.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)
                ϕ_yy = sin.(π * grid.X) .* -9cos.(3grid.Y) .* cos.(5grid.Z)
                ϕ_zz = sin.(π * grid.X) .* cos.(3grid.Y) .* -25cos.(5grid.Z)
                ρ = ϕ_xx + ϕ_yy + ϕ_zz

                ϕ_left(x, y) = 0
                ϕ_right(x, y) = 0

                ϕl = ϕ_left.(grid.Y, grid.Z)
                ϕr = ϕ_right.(grid.Y, grid.Z)

                Ex, Ey, Ez = Braginskii.poisson(ρ, ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer, zeros(size(grid)), fft_plans)

                Ex_expected = π * cos.(π * grid.X) .* cos.(3grid.Y) .* cos.(5grid.Z)

                Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)
                push!(errors, Ex_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ
    end
    end
end
