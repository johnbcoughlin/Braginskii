@testset "Poisson tests" begin
    @testset "Evaluate Laplacian" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nz in Ns
            @no_escape begin
                grid = x_grid_3d(20, 20, Nz, buffer)
                (; X, Y, Z) = grid
                helper = Braginskii.poisson_helper(grid.z.dx, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)

                ϕ = sin.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)

                ϕ_left(x, y) = 0
                ϕ_right(x, y) = 0

                ϕl = ϕ_left.(grid.X, grid.Y)
                ϕr = ϕ_right.(grid.X, grid.Y)

                actual = similar(ϕ)
                Braginskii.apply_laplacian!(actual, ϕ, ϕl, ϕr, grid, [:x, :y, :z], buffer, fft_plans, helper)

                ϕ_xx = @. -π^2 * sin(π * Z) * cos(3 * Y) * cos(5 * X)
                ϕ_yy = @. sin(π * Z) * -9 * cos(3 * Y) * cos(5* X)
                ϕ_zz = @. sin(π * Z) * cos(3 * Y) * -25 * cos(5 * X)
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
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nz in Ns
            @no_escape begin
                grid = x_grid_3d(20, 20, Nz, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)
                helper = Braginskii.poisson_helper(grid.z.dx, buffer)

                ϕ_xx = -π^2 * sin.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)
                ϕ_yy = sin.(π * grid.Z) .* -9cos.(3grid.Y) .* cos.(5grid.X)
                ϕ_zz = sin.(π * grid.Z) .* cos.(3grid.Y) .* -25cos.(5grid.X)
                ρ = ϕ_xx + ϕ_yy + ϕ_zz

                ϕ_left(x, y) = 0
                ϕ_right(x, y) = 0

                ϕl = ϕ_left.(grid.X, grid.Y)
                ϕr = ϕ_right.(grid.X, grid.Y)

                Δ = Braginskii.form_fourier_domain_poisson_operator(ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer, fft_plans, helper)

                Ex, Ey, Ez = Braginskii.poisson(ρ, ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer, alloc_zeros(Float64, buffer, size(grid)...), fft_plans, helper)

                Ez_expected = π * cos.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)

                Ez_error = norm(Ez - Ez_expected) / norm(Ez_expected)
                push!(errors, Ez_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ
    end
    end

    @testset "Poisson direct LU solve z" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nz in Ns
            @no_escape begin
                grid = x_grid_3d(20, 20, Nz, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)
                helper = Braginskii.poisson_helper(grid.z.dx, buffer)

                ϕ_xx = -π^2 * sin.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)
                ϕ_yy = sin.(π * grid.Z) .* -9cos.(3grid.Y) .* cos.(5grid.X)
                ϕ_zz = sin.(π * grid.Z) .* cos.(3grid.Y) .* -25cos.(5grid.X)
                ρ = ϕ_xx + ϕ_yy + ϕ_zz

                ϕ_left(x, y) = 0
                ϕ_right(x, y) = 0

                ϕl = ϕ_left.(grid.X, grid.Y)
                ϕr = ϕ_right.(grid.X, grid.Y)

                Δ = Braginskii.form_fourier_domain_poisson_operator(ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                Ex, Ey, Ez = Braginskii.poisson_direct(ρ, Δ_lu, ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer, alloc_zeros(Float64, buffer, size(grid)...), fft_plans, helper)

                Ez_expected = π * cos.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)

                Ez_error = norm(Ez - Ez_expected) / norm(Ez_expected)
                @show Ez_error
                push!(errors, Ez_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ
    end
    end

    @testset "Poisson direct LU solve x" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        Ns = [20, 40, 80, 160]
        errors = Float64[]
        for Nx in Ns
            @no_escape begin
                grid = Helpers.x_grid_1d(Nx, 2π, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)
                helper = Braginskii.poisson_helper(grid.x.dx, buffer)

                ϕ_xx = -25 * cos.(5grid.X)
                ρ = ϕ_xx

                Δ = Braginskii.form_fourier_domain_poisson_operator([0.], [0.], grid, [:x], 
                    buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                Ex, rest... = Braginskii.poisson_direct(ρ, Δ_lu, [0.], [0.], grid, [:x], 
                    buffer, alloc_zeros(Float64, buffer, size(grid)...), fft_plans, helper)

                Ex_expected = -5 * sin.(5grid.X)

                Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)
                @show Ex_error
                push!(errors, Ex_error)
            end
        end
        @show errors
        γ = estimate_log_slope(Ns, errors)
        @test -4 >= γ
    end
    end
end
