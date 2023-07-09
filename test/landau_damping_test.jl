@testset "Landau damping" begin
    Ny = 16
    Nvy = 50
    δ = 0.001
    q = 1.0
    k = 0.5
    sim = single_species_1d1v_y(; Ny, Nvy, Ly=2π/k, q, vdisc=:weno) do y, vy
        1 / sqrt(2π) * (1.0 + δ * cos(k*y)) * exp(-vy^2/2)
    end

    df = runsim_lightweight!(sim, 20.0, 0.01, 
        diagnostic=Braginskii.lightweight_diagnostics())

    line, γ, freq = Braginskii.find_damping_fit(df.t, df.electric_energy)
    @test -.152 > γ/2 > -.154
end

@testset "Inverse Landau damping" begin
    Ny = 16
    Nvy = 50
    δ = 0.01
    q = 3.0
    k = 1.0
    sim = single_species_1d1v_y(; Ny, Nvy, Ly=2π/k, q, vdisc=:weno) do y, vy
        0.5 / sqrt(2π) * (1.0 + δ * cos(k*y)) * (exp(-(vy-1.5)^2/1) + exp(-(vy+1.5)^2/1))
    end

    df = runsim_lightweight!(sim, 20.0, 0.01, 
        diagnostic=Braginskii.lightweight_diagnostics())

    @test 0.113 < maximum(df.electric_energy) < 0.117
    @test 11.6 < df.t[argmax(df.electric_energy)] < 11.7
end
