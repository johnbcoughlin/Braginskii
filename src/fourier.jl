function in_ky_domain!(g!, arr, buffer)
    Nx, Ny = size(arr)

    arr = reshape(arr, (Nx, Ny, :))

    modes = alloc(ComplexF64, buffer, Nx, Ny÷2+1, size(arr, 3))
    plan = plan_rfft(arr, (2,))
    mul!(modes, plan, arr)

    g!(modes)

    plan = plan_irfft(modes, Ny, (2,))
    mul!(arr, plan, modes)
end

function in_kz_domain!(g!, arr, buffer)
    Nx, Ny, Nz = size(arr)

    arr = reshape(arr, (Nx*Ny, Nz, :))

    modes = alloc(ComplexF64, buffer, Nx*Ny, Nz÷2+1, size(arr, 3))
    plan = plan_rfft(arr, (2,))
    @show size(arr)
    @show size(modes)
    mul!(modes, plan, arr)

    g!(modes)

    plan = plan_irfft(modes, Nz, (2,))
    mul!(arr, plan, modes)
end
