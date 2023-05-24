function apply_fd!(y, x::OffsetArray, K)
    @turbo thread=8 for i in 1:K
        y[i] += 1/30 * x[i-2] - 13/60 * x[i-1] + 47/60 * x[i] + 9/20 * x[i+1] - 1/20 * x[i+2]
    end
    return y
end

function apply_fd_2!(y, x::OffsetArray, K)
    axpy!(1/30, (@view x[-2:end-5]), y)
    axpy!(-13/60, (@view x[-1:end-4]), y)
    return y
end

function apply_fd_3!(y, x::OffsetArray, K)
    stencil = @SVector [1/30, -13/60, 47/60, 9/20, -1/20]
    @turbo thread=4 for i in 1:K
        for s in eachindex(stencil)
            y[i] += stencil[s] * x[i-3+s]
        end
    end
    return y
end

function apply_fd_4!(y1, y2, x::OffsetArray, K)
    stencil1 = @SVector [1/30, -13/60, 47/60, 9/20, -1/20, 0.]
    stencil2 = @SVector [0., 1/30, -13/60, 47/60, 9/20, -1/20]
    @turbo thread=2 for i in 1:K
        for s in eachindex(stencil1)
            y1[i] += stencil1[s] * x[i-3+s]
            y2[i] += stencil2[s] * x[i-3+s]
        end
    end
    return y1, y2
end

function apply_fd_5!(y, x::OffsetArray, K)
    stencil = [1/30, -13/60, 47/60, 9/20, -1/20]
    Kx, Ky, Kz, Kvx, Kvy, Kvz = K
    x_R = reshape(x, (Kx*Ky*Kz, :, Kvy*Kvz)) |> Origin(1, -2, 1)
    y_R = reshape(y, (Kx*Ky*Kz, :, Kvy*Kvz)) |> Origin(1, -2, 1)
    @turbo thread=2 for λvx in 1:Kvx
        for λ1 in 1:Kx*Ky*Kz, λ3 in 1:Kvy*Kvz
            y_R[λ1, λvx, λ3] += 1/30 * x_R[λ1, λvx-2, λ3] - 13/60 * x_R[λ1, λvx-1, λ3] + 47/60 * x_R[λ1, λvx, λ3] + 9/20 * x_R[λ1, λvx+1, λ3] - 1/20 * x_R[λ1, λvx+2, λ3]
        end
    end
    return y
end

