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
    arrays::Vector{CuArray}
    offset::UInt
end

arraytype(::Bumper.AllocBuffer) = Array
arraytype(::GPUAllocator) = CuArray
arraytype(::Array) = CuArray
arraytype(::CuArray) = CuArray

GPUAllocator() = GPUAllocator(CuArray[], 0)

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

alloc_array(::Type{T}, buffer::GPUAllocator, s...) where {T} = begin
    arr = CuArray{T}(undef, s...)
    push!(buffer.arrays, arr)
    arr
end

function Bumper.no_escape(f, b::GPUAllocator)
    offset = b.offset
    res = f()
    while length(b.arrays) > offset
        a = pop!(b.arrays)
        CUDA.unsafe_free!(a)
    end
    res
end
