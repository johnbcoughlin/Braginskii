struct FFTPlans{A, B, C, D, E, F}
    kx_rfft::A
    kx_irfft::B

    ky_rfft::C
    ky_irfft::D

    kz_rfft::E
    kz_irfft::F
end

function plan_ffts(grid)
    Nx, Ny, Nz = size(grid)

    arr = rand(size(grid)...)

    arr = reshape(arr, (Nx, :))
    kx_rfft = plan_rfft(arr, (2,))
    modes = kx_rfft * arr
    kx_irfft = plan_irfft(modes, Nx, (1,))

    arr = reshape(arr, (Nx, Ny, :))
    ky_rfft = plan_rfft(arr, (2,))
    modes = ky_rfft * arr
    ky_irfft = plan_irfft(modes, Ny, (2,))

    arr = reshape(arr, (Nx*Ny, Nz, :))
    kz_rfft = plan_rfft(arr, (2,))
    modes = kz_rfft * arr
    kz_irfft = plan_irfft(modes, Nz, (2,))

    return FFTPlans(kx_rfft, kx_irfft, ky_rfft, ky_irfft, kz_rfft, kz_irfft)
end

function in_kx_domain!(g!, arr, buffer, plans)
    Nx, = size(arr)

    arr = reshape(arr, (Nx, :))

    modes = alloc_array(ComplexF64, buffer, Nx÷2+1, size(arr, 2))
    mul!(modes, plans.kx_rfft, arr)

    g!(modes)

    mul!(arr, plans.kx_irfft, modes)
end

function in_ky_domain!(g!, arr, buffer, plans)
    Nx, Ny = size(arr)

    arr = reshape(arr, (Nx, Ny, :))

    modes = alloc_array(ComplexF64, buffer, Nx, Ny÷2+1, size(arr, 3))
    mul!(modes, plans.ky_rfft, arr)

    g!(modes)

    mul!(arr, plans.ky_irfft, modes)
end

function in_kz_domain!(g!, arr, buffer)
    Nx, Ny, Nz = size(arr)

    arr = reshape(arr, (Nx*Ny, Nz, :))

    modes = alloc_array(ComplexF64, buffer, Nx*Ny, Nz÷2+1, size(arr, 3))
    plan = plan_rfft(arr, (2,))
    mul!(modes, plan, arr)

    g!(modes)

    plan = plan_irfft(modes, Nz, (2,))
    mul!(arr, plan, modes)
end
