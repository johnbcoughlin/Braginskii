struct RK4StageCache{A}
    u′::A
    k¹::A
    k²::A
    k³::A
    k⁴::A
end

rk_4stage_cache(sz) = begin
    RK4StageCache(zeros(sz), zeros(sz), zeros(sz), zeros(sz), zeros(sz))
end

rk_4stage_cache(arr::AbstractArray) = begin
    RK4StageCache(similar(arr), similar(arr), similar(arr), similar(arr), similar(arr))
end

function alloc_vec(buffer, template)
    return ArrayPartition((map(template.x) do tmp
        alloc_zeros(Float64, buffer, size(tmp)...)
    end)...)
end

"""
The four-stage RK method from https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html
http://ketch.github.io/numipedia/methods/SSPRK43.html
"""
function ssp_rk43(F!, uⁿ, p, t, dt,  CFL_max, buffer)
    Bumper.no_escape(buffer) do
        start = copy(uⁿ.x[1][1, :, 1, 1, :, 1])

        u′ = alloc_vec(buffer, uⁿ)
        k¹ = alloc_vec(buffer, uⁿ)
        k² = alloc_vec(buffer, uⁿ)
        k³ = alloc_vec(buffer, uⁿ)
        k⁴ = alloc_vec(buffer, uⁿ)

        u′ .= uⁿ
        F!(k¹, u′, p, t)
        #error("exiting early")

        sf = safety_factor(dt, p, CFL_max)
        sf < 1 || return false, sf

        # u′ = uⁿ + 0.5 k¹
        # @. u′ += 0.5 * dt * k¹
        rk_update_12(u′, k¹, dt)
        F!(k², u′, p, t + dt/2)

        sf = safety_factor(dt, p, CFL_max)
        sf < 1 || return false, sf

        # u′ = uⁿ + 0.5 k¹ + 0.5 k²
        # @. u′ += 0.5 * dt * k²
        rk_update_12(u′, k², dt)
        F!(k³, u′, p, t + dt)

        sf = safety_factor(dt, p, CFL_max)
        sf < 1 || return false, sf

        # u′ = uⁿ + dt/6(k¹ + k² + k³)
        # @. u′ = uⁿ + (dt/6) * (k¹ + k² + k³)
        rk_update_3(u′, uⁿ, k¹, k², k³, dt)
        F!(k⁴, u′, p, t + dt/2)


        sf = safety_factor(dt, p, CFL_max)
        sf < 1 || return false, sf

        # @. uⁿ = uⁿ + dt * (1/6 * (k¹ + k² + k³) + 1/2 * k⁴)
        rk_update_4(uⁿ, k¹, k², k³, k⁴, dt)
        return true, sf
    end
end


function rk_update_12(u′::AbstractArray, k¹²::AbstractArray, dt)
    @.. u′ += 0.5 * dt * k¹²
end

function rk_update_12(u′::ArrayPartition, k¹²::ArrayPartition, dt)
    for i in 1:length(u′.x)
        rk_update_12(u′.x[i], k¹².x[i], dt)
    end
end

function rk_update_12(u′::OffsetArray, k¹²::OffsetArray, dt)
    rk_update_12(parent(u′), parent(k¹²), dt)
end


function rk_update_3(u′::AbstractArray, uⁿ::AbstractArray, k¹::AbstractArray, k²::AbstractArray, k³::AbstractArray, dt)
    @.. u′ = uⁿ + (dt/6) * (k¹ + k² + k³)
end

function rk_update_3(u′::ArrayPartition, uⁿ::ArrayPartition, k¹::ArrayPartition, k²::ArrayPartition, k³::ArrayPartition, dt)
    for i in 1:length(u′.x)
        rk_update_3(u′.x[i], uⁿ.x[i], k¹.x[i], k².x[i], k³.x[i], dt)
    end
end

function rk_update_3(u′::OffsetArray, uⁿ::OffsetArray, k¹::OffsetArray, k²::OffsetArray, k³::OffsetArray, dt)
    rk_update_3(parent(u′), parent(uⁿ), parent(k¹), parent(k²), parent(k³), dt)
end

function rk_update_4(uⁿ::AbstractArray, k¹::AbstractArray, k²::AbstractArray, k³::AbstractArray, k⁴::AbstractArray, dt)
    @.. uⁿ = uⁿ + dt * (1/6 * (k¹ + k² + k³) + 1/2 * k⁴)
end

function rk_update_4(uⁿ::ArrayPartition, k¹::ArrayPartition, k²::ArrayPartition, k³::ArrayPartition, k⁴::ArrayPartition, dt)
    for i in 1:length(uⁿ.x)
        rk_update_4(uⁿ.x[i], k¹.x[i], k².x[i], k³.x[i], k⁴.x[i], dt)
    end
end

function rk_update_4(uⁿ::OffsetArray, k¹::OffsetArray, k²::OffsetArray, k³::OffsetArray, k⁴::OffsetArray, dt)
    rk_update_4(parent(uⁿ), parent(k¹), parent(k²), parent(k³), parent(k⁴), dt)
end


function safety_factor(dt, p, CFL_max)
    return dt * p.λmax[] / CFL_max
end
