struct FFTPlans{A, B, C, D}
    ky_rfft::A
    ky_irfft::B

    kz_rfft::C
    kz_irfft::D
end

function plan_ffts(grid)
    Nx, Ny, Nz = size(grid)

    arr = @StrideArray rand(size(grid))

    arr = reshape(arr, (Nx, Ny, :))
    ky_rfft = plan_rfft(arr, (2,))
    modes = ky_rfft * arr
    ky_irfft = plan_irfft(modes, Ny, (2,))

    arr = reshape(arr, (Nx*Ny, Nz, :))
    kz_rfft = plan_rfft(arr, (2,))
    modes = kz_rfft * arr
    kz_irfft = plan_irfft(modes, Nz, (2,))

    return FFTPlans(ky_rfft, ky_irfft, kz_rfft, kz_irfft)
end

function in_ky_domain!(g!, arr, buffer, plans)
    Nx, Ny = size(arr)

    arr = reshape(arr, (Nx, Ny, :))

    modes = alloc(ComplexF64, buffer, Nx, Ny÷2+1, size(arr, 3))
    mul!(modes, plans.ky_rfft, arr)

    g!(modes)

    mul!(arr, plans.ky_irfft, modes)
end

function in_kz_domain!(g!, arr, buffer)
    Nx, Ny, Nz = size(arr)

    arr = reshape(arr, (Nx*Ny, Nz, :))

    modes = alloc(ComplexF64, buffer, Nx*Ny, Nz÷2+1, size(arr, 3))
    plan = plan_rfft(arr, (2,))
    mul!(modes, plan, arr)

    g!(modes)

    plan = plan_irfft(modes, Nz, (2,))
    mul!(arr, plan, modes)
end
