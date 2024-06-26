struct FFTPlans{A, B, C, D, E, F}
    kxy_rfft::A
    kxy_irfft::B

    kxy_double_irfft::C
    kxy_double_rfft::D

    kxyz_rfft::E
    kxyz_irfft::F
end

function plan_ffts(grid, buffer)
    Nx, Ny, Nz, rest... = size(grid)
    plan_ffts(Nx, Ny, Nz, rest, buffer)
end

function plan_ffts(Nx, Ny, Nz, rest, buffer)
    rest = prod(rest)

    arr = alloc_array(Float64, buffer, Nx, Ny, Nz, rest...)

    arr = reshape(arr, (Nx, Ny, :))
    kxy_rfft = plan_rfft(arr, (1, 2))
    modes = kxy_rfft * arr
    kxy_irfft = plan_irfft(modes, Nx, (1, 2))

    double_modes = alloc_zeros(Complex{Float64}, buffer, Nx, 2Ny, Nz*rest)
    kxy_double_irfft = plan_irfft(double_modes, 2Nx-1, (1, 2))

    U = kxy_double_irfft * double_modes
    kxy_double_rfft = plan_rfft(U, (1, 2))

    if rest == 1
        arr = reshape(arr, (Nx, Ny, Nz))
        kxyz_rfft = plan_rfft(arr, (1, 2, 3))
        modes = kxyz_rfft * arr
        kxyz_irfft = plan_irfft(modes, Nx, (1, 2, 3))
    else
        kxyz_rfft = nothing
        kxyz_irfft = nothing
    end

    return FFTPlans(kxy_rfft, kxy_irfft, kxy_double_irfft, kxy_double_rfft,
        kxyz_rfft, kxyz_irfft)
end

function in_kxy_domain!(g!, arr, buffer, plans)
    Nx, Ny, = size(arr)


    arr = reshape(arr, (Nx, Ny, :))

    modes = alloc_array(ComplexF64, buffer, Nx÷2+1, Ny, size(arr, 3))
    mul!(modes, plans.kxy_rfft, arr)

    g!(modes)

    mul!(arr, plans.kxy_irfft, modes)
end

plan_rfft(a::Array, args...) = FFTW.plan_rfft(a, args...)
plan_irfft(a::Array, args...) = FFTW.plan_irfft(a, args...)

plan_rfft(a::CuArray, args...) = CUDA.CUFFT.plan_rfft(a, args...)
plan_irfft(a::CuArray, args...) = CUDA.CUFFT.plan_irfft(a, args...)
