@testset "Landau damping" begin
    Ny = 24
    Nvy = 100
    δ = 0.001
    q = 1.0
    k = 0.5
    sim = single_species_1d1v_y(Ny, Nvy, 2π/k; q) do y, vy
        1 / sqrt(2π) * (1.0 + δ * cos(k*y)) * exp(-vy^2/2)
    end

    df = runsim_lightweight!(sim, 20.0, 0.01, 
        diagnostic=Braginskii.lightweight_diagnostics())

    line, γ, freq = Braginskii.find_damping_fit(df.t, df.electric_energy)
    @test -.152 > γ/2 > -.154
end
