@testset "Poisson tests" begin
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

                Δ = Braginskii.form_fourier_domain_poisson_operator(grid, [:x, :y, :z], buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                Ex, Ey, Ez = Braginskii.poisson_direct(ρ, Δ_lu, ϕl, ϕr, grid, [:x, :y, :z], 
                    buffer, alloc_zeros(Float64, buffer, size(grid)...), fft_plans, helper)

                Ez_expected = π * cos.(π * grid.Z) .* cos.(3grid.Y) .* cos.(5grid.X)

                Ez_error = norm(Ez - Ez_expected) / norm(Ez_expected)
                push!(errors, Ez_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test -0.8 >= γ
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

                Δ = Braginskii.form_fourier_domain_poisson_operator(grid, [:x], buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                Ex, rest... = Braginskii.poisson_direct(ρ, Δ_lu, [0.], [0.], grid, [:x], 
                    buffer, alloc_zeros(Float64, buffer, size(grid)...), fft_plans, helper)

                Ex_expected = -5 * sin.(5grid.X)

                Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)
                push!(errors, Ex_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        @test norm(errors) < 1e-14
    end
    end

    @testset "Poisson direct LU solve xz" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        Ns = [40, 80]
        errors = Float64[]
        for N in Ns
            @no_escape begin
                grid = Helpers.xz_grid_2d(N, N, -1.0, 1.0, 3π, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)
                helper = Braginskii.poisson_helper(grid.z.dx, buffer)

                gaussian_z(z) = exp(-z^2/ 2T)
                T = 0.02

                ϕ(x, z) = cos(2x/3) * gaussian_z(z)
                ϕ_expected = ϕ.(grid.X, grid.Z) .+ 1.0
                ϕxx(x, z) = -4/9*cos(2x/3) * gaussian_z(z)
                ϕzz(x, z) = cos(2x/3) * (1/T) * ((1/T)z^2 - 1) * gaussian_z(z)
                Δϕ = ϕxx.(grid.X, grid.Z) + ϕzz.(grid.X, grid.Z)
                ρ = -Δϕ

                Δ = Braginskii.form_fourier_domain_poisson_operator(grid, [:x, :z], buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                ϕ = Braginskii.do_poisson_solve(Δ_lu, ρ, grid, 1.0, 1.0, helper, [:x, :z], fft_plans, buffer)
                Ex, _, Ez = Braginskii.poisson_direct(ρ, Δ_lu, [1.], [1.], grid, [:x, :z], 
                    buffer, ϕ, fft_plans, helper)

                Ex_expected = 2/3*sin.(2*grid.X/3) .* gaussian_z.(grid.Z)
                Ez_expected = -cos.(2*grid.X/3) .* ((1/T) .* -grid.Z .* gaussian_z.(grid.Z))

                ϕ_error = norm(ϕ - ϕ_expected) / norm(ϕ_expected)

                Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)

                Ez_error = norm(Ez - Ez_expected) / norm(Ez_expected)

                push!(errors, Ex_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        #@test norm(errors) < 1e-14
    end
    end

    @testset "Poisson fft solve xz" begin
        for device in supported_devices()
        buffer = Braginskii.allocator(device)
        Ns = [20, 40]
        errors = Float64[]
        for N in Ns
            @no_escape begin
                grid = Helpers.xz_grid_2d(N, N, -1.0, 1.0, 3π, buffer)
                fft_plans = Braginskii.plan_ffts(grid, buffer)
                helper = Braginskii.poisson_helper(grid.z.dx, buffer)

                gaussian_z(z) = exp(-z^2/ 2T)
                T = 0.02

                ϕ(x, z) = cos(2x/3) * gaussian_z(z)
                ϕ_expected = ϕ.(grid.X, grid.Z) .+ 1.0
                ϕxx(x, z) = -4/9*cos(2x/3) * gaussian_z(z)
                ϕzz(x, z) = cos(2x/3) * (1/T) * ((1/T)z^2 - 1) * gaussian_z(z)
                Δϕ = ϕxx.(grid.X, grid.Z) + ϕzz.(grid.X, grid.Z)
                ρ = -Δϕ

                Δ = Braginskii.form_fourier_domain_poisson_operator(grid, [:x, :z], buffer)
                Δ_lu = Braginskii.factorize_poisson_operator(Δ)

                ϕ = Braginskii.do_poisson_solve(Δ_lu, ρ, grid, 1.0, 1.0, helper, [:x, :z], fft_plans, buffer)
                ϕ_avg = sum(ϕ) / length(ϕ)
                Ex, _, Ez = Braginskii.poisson_fft(ρ, Δ_lu, 1., 1., grid, [:x, :z], 
                    buffer, ϕ, fft_plans, helper)

                Ex_expected = 2/3*sin.(2*grid.X/3) .* gaussian_z.(grid.Z)
                Ez_expected = -cos.(2*grid.X/3) .* ((1/T) .* -grid.Z .* gaussian_z.(grid.Z))

                ϕ_error = norm(ϕ - ϕ_expected) / norm(ϕ_expected)

                Ex_error = norm(Ex - Ex_expected) / norm(Ex_expected)

                Ez_error = norm(Ez - Ez_expected) / norm(Ez_expected)

                push!(errors, Ex_error)
            end
        end
        γ = estimate_log_slope(Ns, errors)
        #@test norm(errors) < 1e-14
    end
    end
end
