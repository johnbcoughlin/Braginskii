module BragAnalysis

using Braginskii
using StaticArrays
using SparseArrays
using JLD2
using LinearAlgebra
using TOML

export x_comp, z_comp, xx_comp, zz_comp, xz_comp

struct Context
    campaign_dir::String
    prefix::String
    vt::Float64
    Nz::Int
    Nx::Int
    dx::Float64
    dz::Float64
    gz::Float64
end

struct Snapshot
    dict::Dict
end

struct Writeout
    array::Array{Float64, 6}
end

# Loader functions

function ions_snapshot(sim_id, frame, ctx)
    jldopen(joinpath(ctx.campaign_dir, "$(ctx.prefix)-$(sim_id)", "snapshots.jld2")) do file
        Snapshot(file["snapshot_$(frame)_ions"])
    end
end

function ions_writeout(sim_id, frame, ctx)
    jldopen(joinpath(ctx.campaign_dir, "$(ctx.prefix)-$(sim_id)", "writeouts.jld2")) do file
        Writeout(file["frame_$frame"]["ions"]["f"])
    end
end

function E_field_writeout(sim_id, frame, ctx)
    jldopen(joinpath(ctx.campaign_dir, "$(ctx.prefix)-$(sim_id)", "writeouts.jld2")) do file
        Ex, Ez = (file["frame_$frame"]["Ex"][:, 1, :], file["frame_$frame"]["Ez"][:, 1, :])

        func = (ux, uy) -> @SVector [ux, uy]
        func.(Ex, Ez)
    end
end

function writeout_time(sim_id, frame, ctx)
    jldopen(joinpath(ctx.campaign_dir, "$(ctx.prefix)-$(sim_id)", "writeouts.jld2")) do file
        file["frame_$frame"]["t"]
    end
end

function sim_params(sim_id, ctx)
    TOML.parsefile(joinpath(ctx.campaign_dir, "$(ctx.prefix)-$(sim_id)", "params.toml"))
end

function oct(sim_id, ctx)
    sim_params(sim_id, ctx)["ωcτ"]
end

function opt(sim_id, ctx)
    sim_params(sim_id, ctx)["ωpτ"]
end

function get_rL_over_alpha(sim_id, ctx)
    vti = ctx.vt
    alpha = 0.04
    rL = vti / get_oct(sim_id)

    return rL / alpha
end

D1(N, d) = begin
    result = spdiagm(-1 => -ones(N-1), 1 => ones(N-1))
    result[1, 1:2] .= [-2, 2]
    result[end, end-1:end] .= [-2, 2]
    result ./ (2d)
end
D1_dirichlet(N, d) = begin
    result = spdiagm(-1 => -ones(N-1), 1 => ones(N-1))
    result ./ (2d)
end
D1_periodic(N, d) = begin
    result = spdiagm(-1 => -ones(N-1), 1 => ones(N-1))
    result[1, end] =-1
    result[end, 1] = 1
    result ./ (2d)
end    

d_dz(A, ctx) = begin
    ddz = kron(D1(ctx.Nz, ctx.dz), I(ctx.Nx));
    reshape(ddz * vec(A), size(A))
end

d_dx(A, ctx) = begin
    ddx = kron(I(ctx.Nz), D1_periodic(ctx.Nx, ctx.dx));
    reshape(ddx * vec(A), size(A))
end

#### Snapshots

function density(snapshot::Snapshot)
    return snapshot.dict["n"] |> as_xz
end

function temp(snapshot::Snapshot)
    return snapshot.dict["T"] |> as_xz
end

function pressure(s::Snapshot)
    return density(s) .* temp(s)
end

function u(snapshot::Snapshot)
    func = (ux, uy) -> @SVector [ux, uy]
    func.(u_x(snapshot), u_z(snapshot))
end

function u_x(snapshot::Snapshot)
    return snapshot.dict["u_x"] |> as_xz
end

function u_z(snapshot::Snapshot)
    return snapshot.dict["u_z"] |> as_xz
end

function q(snapshot::Snapshot)
    func = (qx, qy) -> @SVector [qx, qy]
    func.(q_x(snapshot), q_z(snapshot))
end

function q_x(snapshot::Snapshot)
    return (snapshot.dict["q_x"] |> as_xz) / 2
end

function q_z(snapshot::Snapshot)
    return (snapshot.dict["q_z"] |> as_xz) / 2
end

function Tperp0(snapshot::Snapshot)
    T = temp(snapshot)
    n = density(snapshot)
    ux = u_x(snapshot)
    uy = u_y(snapshot)

    return @. T + n * (ux^2 + uy^2) / 2
end

function grad(scalar, ctx)
    dx = d_dx(scalar, ctx)
    dz = d_dz(scalar, ctx)
    func(x, z) = @SVector[x, z]
    func.(dx, dz)
end

function div(vector, ctx)
    dx = d_dx(x_comp(vector), ctx)
    dz = d_dz(z_comp(vector), ctx)
    dx .+ dz
end

function dot(vec1, vec2)
    func(a, b) = a ⋅ b
    func.(vec1, vec2)
end

function vector(ux, uz)
    func(x, z) = @SVector[x, z]
    func.(ux, uz)
end

function tensor(A_xx, A_xz, A_zx, A_zz)
    func(xx, xz, zx, zz) = @SMatrix[xx  xz;
                                    zx  zz]
    func.(A_xx, A_xz, A_zx, A_zz)
end

function crossB(field)
    func(u) = begin
        ux, uy = u
        @SVector[uy, -ux]
    end
    return func.(field)
end

function x_comp(field)
    return (s -> s[1]).(field)
end
function xx_comp(field)
    return (s -> s[1, 1]).(field)
end
function z_comp(field)
    return (s -> s[2]).(field)
end
function zz_comp(field)
    return (s -> s[2, 2]).(field)
end
function xz_comp(field)
    return (s -> s[1, 2]).(field)
end

Dcal(vector, ctx) = begin
    vx = x_comp(vector)
    vz = z_comp(vector)
    tensor(
        -d_dz(vx, ctx) - d_dx(vz, ctx),
        d_dx(vx, ctx) - d_dz(vz, ctx),
        d_dx(vx, ctx) - d_dz(vz, ctx),
        d_dz(vx, ctx) + d_dx(vz, ctx)
    )
end

widehat(u, v) = begin
    ux = x_comp(u); uz = z_comp(u);
    vx = x_comp(v); vz = z_comp(v);
    tensor(
        (@. ux * vx - uz*vz),
        (@. ux * vz + uz*vx),
        (@. uz * vx + ux*vz),
        (@. -ux * vx + uz*vz)
    )
end
widehat2(u, v) = begin
    ux = x_comp(u); uz = z_comp(u);
    vx = x_comp(v); vz = z_comp(v);
    tensor(
        (@. ux * vx + uz*vz),
        (@. ux * vz + uz*vx),
        (@. ux * vx - uz*vz),
        (@. uz * vx + ux*vz),
    )
end


#### Writeouts

function density(f::Writeout)
    return f.array[:, 1, :, 1, 1, 1]
end

function u_x(f::Writeout, ctx)
    n = density(f)
    nuix = ctx.vt * f.array[:, 1, :, 2, 1, 1];
    nuix ./ n
end

function u_z(f::Writeout, ctx)
    n = density(f)
    nuiz = ctx.vt * f.array[:, 1, :, 1, 1, 2];
    nuiz ./ n
end

function Tperp0(f::Writeout, ctx)
    n = density(f)
    Mxx = ctx.vt^2 * (sqrt(2) * f.array[:, 1, :, 3, 1, 1] + n)
    Mzz = ctx.vt^2 * (sqrt(2) * f.array[:, 1, :, 1, 1, 3] + n)
    return (Mxx + Mzz) ./ 2n
end

function temp(f::Writeout, ctx)
    n = density(f)
    ux = u_x(f, ctx)
    uz = u_z(f, ctx)

    T0 = Tperp0(f, ctx)
    p0 = T0 .* n
    p = p0 - n .* (ux.^2 + uz.^2)/2
    p ./ n
end

function uD(writeout, E, sim_id, ctx)
    T0 = Tperp0(writeout, ctx)
    n = density(writeout)
    p0 = T0 .* n

    ud = crossB(grad(p0, ctx)) ./ n
    for i in eachindex(ud)
        ud[i] = ud[i] + @SVector[-ctx.gz, 0.0]
    end
    ExB = opt(sim_id, ctx) * crossB(E)
    return (ud - ExB) / oct(sim_id, ctx)
end

function uD_proper(writeout, E, sim_id, ctx)
    T = temp(writeout, ctx)
    n = density(writeout)
    p = T .* n

    ud = crossB(grad(p, ctx)) ./ n
    for i in eachindex(ud)
        ud[i] = ud[i] + @SVector[-ctx.gz, 0.0]
    end
    ExB = opt(sim_id, ctx) * crossB(E)
    return (ud - ExB) / oct(sim_id, ctx)
end

function ug(writeout, sim_id, ctx)
    n = density(writeout)
    -map(n) do ni 
        @SVector[-ctx.gz, 0.0]
    end / oct(sim_id, ctx)
end

function uE(writeout, E, sim_id, ctx)
    ExB = opt(sim_id, ctx) * crossB(E)
    return -ExB / oct(sim_id, ctx)
end

function uT(writeout, sim_id, ctx)
    T0 = Tperp0(writeout, ctx)
    n = density(writeout)

    return crossB(grad(T0, ctx)) / oct(sim_id, ctx)
end

function un(writeout, sim_id, ctx)
    T0 = Tperp0(writeout, ctx)
    n = density(writeout)

    return (T0 ./ n) .* crossB(grad(n, ctx)) / oct(sim_id, ctx)
end

function all_moments(wr::Writeout, ctx)
    vth = ctx.vt
    moments = Braginskii.moments_for_wsindy(wr.array, vth, [:vx, :vz], allocator(:cpu))
    n, ux, _, uz, T, Pi_xx, _, Pi_zz, _, Pi_xz, _, qx, _, qz = map(as_xz, moments);

    u = vector(ux, uz)
    q = vector(qx, qz)
    Pi = tensor(Pi_xx, Pi_xz, Pi_xz, Pi_zz)

    return n, u, T, Pi, q
end

end
