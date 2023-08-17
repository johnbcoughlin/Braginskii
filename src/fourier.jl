struct FFTPlans{A, B, C, D}
    ky_rfft::A
    ky_irfft::B

    kz_rfft::C
    kz_irfft::D
end

function plan_ffts(grid, buffer)
    Nx, Ny, Nz = size(grid)

    arr = alloc_array(Float64, buffer, size(grid)...)

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

plan_rfft(a::Array, args...) = FFTW.plan_rfft(a, args...)
plan_irfft(a::Array, args...) = FFTW.plan_irfft(a, args...)

plan_rfft(a::CuArray, args...) = CUDA.CUFFT.plan_rfft(a, args...)
plan_irfft(a::CuArray, args...) = CUDA.CUFFT.plan_irfft(a, args...)

function make_cufft_dim2_rfft()
    ref = Ref{CUDA.CUFFT.cufftHandle}()

    Nx = 2
    Ny = 10
    Nz = 1

    rank = 1
    batch = Nx
    n = [Ny]
    istride = ostride = Nx
    inembed = [Nz, Ny, Nx]
    onembed = [Nz, (Ny÷2)+1, Nx]
    idist = 1
    odist = 1

    cufftType = CUDA.CUFFT.CUFFT_D2Z

    a = rand(Nx, Ny, Nz)
    data = CUDA.zeros(Float64, Nx, Ny, Nz)
    copyto!(data, a)

    CUDA.CUFFT.cufftCreate(ref)

    CUDA.CUFFT.cufftPlanMany(ref, rank, n, inembed, istride, idist, onembed,
        ostride, odist, cufftType, batch)

    output = CUDA.rand(ComplexF64, Nx, (Ny÷2)+1, Nz)

    CUDA.initialize_context()

    @show typeof(data)
    @show typeof(output)

    CUDA.CUFFT.cufftExecD2Z(ref[], data, output)

    actual = Array(output)
    expected = FFTW.rfft(a, [2])
    
    display(expected)
    display(actual)
    display(expected - actual)
    @assert expected ≈ actual
end
