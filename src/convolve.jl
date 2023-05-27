function convolve_x!(dest::AbstractArray{T, N}, u::AbstractArray{T, N}, args...) where {T, N}
    Nx, = size(u)
    u = reshape(u, (Nx, :))

    Nx, = size(dest)
    dest = reshape(dest, (Nx, :))
    convolve_over_first!(dest, u, args...)
end

function convolve_vx!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz, Nvx, Nvy*Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(du)
    dest = reshape(dest, (Nx*Ny*Nz, Nvx, Nvy*Nvz))
    convolve_over_middle!(dest, u, args...)
end

function convolve_vy!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz*Nvx, Nvy, Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(du)
    dest = reshape(dest, (Nx*Ny*Nz*Nvx, Nvy, Nvz))
    convolve_over_middle!(dest, u, args...)
end

function convolve_vz!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz*Nvx*Nvy, Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(du)
    dest = reshape(dest, (Nx*Ny*Nz*Nvx*Nvy, Nvz))
    convolve_over_last!(dest, u, args...)
end

function convolve_over_first!(dest::AbstractArray{T, 2}, u::AbstractArray{T, 2}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (pad, 0), (Colon(), 1))
end

function convolve_over_middle!(dest::AbstractArray{T, 3}, u::AbstractArray{T, 3}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (0, pad, 0), (1, Colon(), 1))
end

function convolve_over_last!(dest::AbstractArray{T, 2}, u::AbstractArray{T, 2}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (0, pad), (1, Colon()))
end

function convolve_over!(
    dest::AbstractArray{T, N}, u::AbstractArray{T, N}, 
    stencil::AbstractVector{T}, has_boundary, buffer,
    pad_wrapper, stencil_shape) where {T, N}

    @assert isodd(length(stencil))
    pad = has_boundary ? 0 : length(stencil) ÷ 2

    u = reshape(u, (size(u)..., 1, 1))
    dest = reshape(dest, (size(dest)..., 1, 1))
    stencil = reshape(stencil, (stencil_shape..., 1, 1))

    cdims = DenseConvDims(size(u), size(stencil), padding=pad_wrapper(pad), flipkernel=true)

    col = alloc(Float64, buffer, prod(size(dest)), length(stencil), 1)
    conv!(dest, u, stencil, cdims; col)
end
