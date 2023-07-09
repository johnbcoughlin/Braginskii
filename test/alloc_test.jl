@testset "Allocator tests" begin
    @no_escape begin
        buffer = default_buffer()
        A = alloc_zeros(Float64, buffer, 10, 10)
        @test A == zeros(10, 10)
    end
end
