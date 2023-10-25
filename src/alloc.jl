import Bumper.no_escape

export allocator

function allocator(device)
    if device == :cpu
        return Bumper.default_buffer()
    elseif device == :gpu
        return GPUAllocator()
    end
end

mutable struct GPUAllocator
    pools::Dict
    checked_out::Vector{CuArray}
end

arraytype(device::Symbol) = begin
    if device == :cpu
        return Array
    elseif device == :gpu
        return CuArray
    end
end
arraytype(::Bumper.AllocBuffer) = Array
arraytype(::GPUAllocator) = CuArray
arraytype(::Array) = Array
arraytype(::CuArray) = CuArray
sparsearraytype(::Bumper.AllocBuffer) = SparseMatrixCSC{Float64, Int64}
sparsearraytype(::GPUAllocator) = CuSparseMatrixCSR{Float64, Int64}

GPUAllocator() = GPUAllocator(Dict(), CuArray[])

next_greatest_multiple(x, b) = (((x-1) ÷ b)+1) * b

alloc_zeros(::Type{T}, args...) where {T} = begin
    res = alloc_array(T, args...)
    res .= zero(T)
    res
end

alloc_array(::Type{T}, buffer::Bumper.AllocBuffer, s...) where {T} = begin
    ptr = Bumper.Internals.alloc_ptr(buffer, prod(s) * sizeof(T))
    buffer.offset = next_greatest_multiple(buffer.offset, 16)
    unsafe_wrap(Array, convert(Ptr{T}, ptr), s)
end

alloc_array(::Type{T}, buffer::GPUAllocator, s::Vararg{Int64, N}) where {T, N} = begin
    @assert !isnothing(buffer.checked_out)

    pool_key = (tuple(s...), T)
    pool = get!(buffer.pools, pool_key, CuArray{T}[])
    arr = if length(pool) == 0
        CuArray{T}(undef, s...)
    else
        pop!(pool)
    end
    push!(buffer.checked_out, arr)
    return arr::CuArray{T, N, CUDA.Mem.DeviceBuffer}
end

function Bumper.no_escape(f, b::GPUAllocator)
    # Store the checkout list from the parent scope, to restore it later.
    previous_checkout_list = b.checked_out

    # Keep track of the arrays checked out from pools in this scope
    checked_out = CuArray[]
    b.checked_out = checked_out

    # Do the work
    res = f()

    # Return checked out arrays to their pools
    for arr in checked_out
        key = (size(arr), eltype(arr))
        push!(b.pools[key], arr)
    end

    b.checked_out = previous_checkout_list

    res
end
