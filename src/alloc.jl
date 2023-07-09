alloc_zeros(::Type{T}, args...) where {T} = begin
    res = alloc_array(T, args...)
    res .= zero(T)
    res
end

alloc_array(::Type{T}, s...) where {T} = alloc_array(T, default_buffer(), s...)

alloc_array(::Type{T}, buffer::Bumper.AllocBuffer, s...) where {T} = begin
    ptr = Bumper.alloc_ptr(buffer, prod(s) * sizeof(T))
    unsafe_wrap(Array, convert(Ptr{T}, ptr), s)
end
