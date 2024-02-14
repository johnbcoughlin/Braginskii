function convolve_x!(dest::AbstractArray{T, N}, u::AbstractArray{T, N}, args...) where {T, N}
    Nx, Ny, Nz, = size(u)
    u = reshape(u, (Nx, :))

    Nx, = size(dest)
    dest = reshape(dest, (Nx, :))
    convolve_over_first!(dest, u, args...)
end

function convolve_z!(dest::AbstractArray{T, N}, u::AbstractArray{T, N1}, args...) where {T, N, N1}
    Nx, Ny, Nz, = size(u)
    u = reshape(u, (Nx*Ny, Nz, :))

    Nx, Ny, Nz = size(dest)
    dest = reshape(dest, (Nx*Ny, Nz, :))
    convolve_over_middle!(dest, u, args...)
end

function convolve_vx!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz, Nvx, Nvy*Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(dest)
    dest = reshape(dest, (Nx*Ny*Nz, Nvx, Nvy*Nvz))
    convolve_over_middle!(dest, u, args...)
end

function convolve_vy!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz*Nvx, Nvy, Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(dest)
    dest = reshape(dest, (Nx*Ny*Nz*Nvx, Nvy, Nvz))
    convolve_over_middle!(dest, u, args...)
end

function convolve_vz!(dest::AbstractArray{T, 6}, u::AbstractArray{T, 6}, args...) where {T}
    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(u)
    u = reshape(u, (Nx*Ny*Nz*Nvx*Nvy, Nvz))

    Nx, Ny, Nz, Nvx, Nvy, Nvz = size(dest)
    dest = reshape(dest, (Nx*Ny*Nz*Nvx*Nvy, Nvz))
    convolve_over_last!(dest, u, args...)
end

function convolve_over_first!(dest::AbstractArray{T, 2}, u::AbstractArray{T, 2}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (pad, 0), (Colon(), 1))
end

function convolve_over_middle!(dest::AbstractArray{T, 4}, u::AbstractArray{T, 3}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (0, pad, 0), (1, Colon(), 1))
end

function convolve_over_middle!(dest::AbstractArray{T, 3}, u::AbstractArray{T, 3}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (0, pad, 0), (1, Colon(), 1))
end

function convolve_over_last!(dest::AbstractArray{T, 2}, u::AbstractArray{T, 2}, args...) where {T}
    convolve_over!(dest, u, args..., pad -> (0, pad), (1, Colon()))
end

function reshape_stencil(stencil::AbstractVector, stencil_shape)
    reshape(stencil, (stencil_shape..., 1, 1))
end

function convolve_over!(
    dest::AbstractArray{T, N}, u::AbstractArray{T, N}, 
    args...) where {T, N, N1}
    convolve_over!(reshape(dest, (size(dest)..., 1)), u, args...)
end

function convolve_over!(
    dest::AbstractArray{T, N}, u::AbstractArray{T, N}, 
    stencil::AbstractArray{T}, has_boundary, buffer,
    pad_wrapper, stencil_shape) where {T, N}

    @assert isodd(size(stencil, 1))
    pad = has_boundary ? 0 : length(stencil) ÷ 2

    reshaped_u = reshape(u, (size(u)..., 1, 1))
    reshaped_dest = reshape(dest, (size(dest)..., 1, 1))
    reshaped_stencil = reshape(convert_stencil(stencil, typeof(u)), (stencil_shape..., 1, 1))

    cdims = DenseConvDims(size(reshaped_u), size(reshaped_stencil), 
        padding=pad_wrapper(pad), flipkernel=true)

    if (isa(u, Array))
        conv!(reshaped_dest, reshaped_u, reshaped_stencil, cdims)
    else
        conv!(reshaped_dest, reshaped_u, reshaped_stencil, cdims)
    end
end

function convolve_over!(
    dest::AbstractArray{T, N1}, u::AbstractArray{T, N}, 
    stencil::AbstractArray{T}, has_boundary, buffer,
    n_channels,
    pad_wrapper, stencil_shape) where {T, N, N1}

    @assert isodd(size(stencil, 1))
    pad = has_boundary ? 0 : length(stencil) ÷ 2

    reshaped_u = reshape(u, (size(u)..., 1, 1))
    reshaped_dest = reshape(dest, (size(dest)..., 1))
    reshaped_stencil = reshape(convert_stencil(stencil, typeof(u)), (stencil_shape..., n_channels, 1))

    cdims = DepthwiseConvDims(size(reshaped_u), size(reshaped_stencil), 
        padding=pad_wrapper(pad), flipkernel=true)


    if (isa(u, Array))
        depthwiseconv!(reshaped_dest, reshaped_u, reshaped_stencil, cdims)
    else
        depthwiseconv!(reshaped_dest, reshaped_u, reshaped_stencil, cdims)
    end
end

convert_stencil(stencil, ::Type{<:CuArray}) = CuArray(stencil)
convert_stencil(stencil, _) = stencil
