module CUDAProfiling

using TimerOutputs
using Braginskii
using CUDA

function vlasov_fokker_planck!(du, f, sim, λmax, buffer)
    λmax[] = 0.0

    no_escape(buffer) do
        @timeit "poisson" Ex, Ey, Ez = poisson(sim, f, buffer)

        @timeit "collisional moments" collisional_moments!(sim, f, buffer)

        for i in eachindex(sim.species)
            α = sim.species[i]

            df = du.x[i]
            df .= 0

            if sim.free_streaming
                @timeit "free streaming" free_streaming!(df, f.x[i], α, buffer)
            end
            @timeit "electrostatic" electrostatic!(df, f.x[i], Ex, Ey, Ez, sim.By, α, buffer, sim.fft_plans)

            @timeit "dfp" dfp!(df, f.x[i], α, sim, buffer)
        end
    end
end


function do_free_streaming(sim)
    buffer = sim.buffer
    f = sim.u
    df = similar(sim.u)
    α = sim.species[1]

    buf = IOBuffer()

    CUDA.@profile io=buf begin
        for i in 1:100
            Braginskii.free_streaming!(df.x[1], f.x[1], α, buffer)
        end
    end

    open("free_streaming.txt", "w") do f
        write(f, take!(buf))
    end
end

end
