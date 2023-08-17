struct FFTPlans{A, B}
    kxy_rfft::A
    kxy_irfft::B
end

function plan_ffts(grid, buffer)
    Nx, Ny, Nz = size(grid)

    arr = alloc_array(Float64, buffer, size(grid)...)

    arr = reshape(arr, (Nx, Ny, :))
    kxy_rfft = plan_rfft(arr, (1, 2))
    modes = kxy_rfft * arr
    kxy_irfft = plan_irfft(modes, Nx, (1, 2))

    return FFTPlans(kxy_rfft, kxy_irfft)
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
