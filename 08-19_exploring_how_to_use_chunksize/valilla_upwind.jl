#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    a simple 1D FV upwind!,  with a uniform mesh and periodic boundaries.
    Artificially create a simple cache object holding intermediate values for the FV fluxes.
=#

using Enzyme
using ForwardDiff

const C = 0.2 # C = Δt/Δx

function upwind!(du::Vector, u::Vector, cache)
    cache.numerical_flux .= u .* cache.v
	for i = 2:length(u)
        du[i] = - C * (cache.numerical_flux[i] - cache.numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (cache.numerical_flux[1] - cache.numerical_flux[end])
    return nothing
end

function upwind!(du::Vector, u::Vector)
    v = 1.0
    numerical_flux = u .* v
	for i = 2:length(u)
        du[i] = - C * (numerical_flux[i] - numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (numerical_flux[1] - numerical_flux[end])
    return nothing
end

# %%
function jacobian_ad_forward_enzyme_cache_upwind_right(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    cache = (;v=1.0, numerical_flux=zeros(length(x)))

    dy = Tuple(zeros(size(du_ode)) for _ in 1:length(u_ode))
    dx = Enzyme.onehot(u_ode)
    cache_shadows =Tuple((;v=1.0, numerical_flux=zeros(length(x))) for i=1:length(x))

    # cache is passed to upwind!
    Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.BatchDuplicated(du_ode, dy), Enzyme.BatchDuplicated(u_ode, dx), Enzyme.BatchDuplicated(cache, cache_shadows))
    return stack(dy)
end

x = -1.0:0.5:1.0
@time jacobian_ad_forward_enzyme_cache_upwind_right(x);


# %%

function gradients_ad_forward_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    numerical_flux= zeros(length(x))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    numerical_flux_shadow= zeros(length(x))
    dys = zeros(length(du_ode), length(du_ode))
    velocity = 1.0

    inner_func!(du, u, numerical_flux) = let
        v = velocity
        upwind!(du, u, (;v=velocity, numerical_flux=numerical_flux))
    end

    for i in 1:length(x)
        dx[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Forward, inner_func!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx), Enzyme.Duplicated(numerical_flux, numerical_flux_shadow))
        dys[:, i] .= dy
        dx[i] = 0.0
    end
    return dys
end

# %%

x=-1.0:0.1:1
@time J1 = gradients_ad_forward_enzyme_cache_upwind(x);
@time J2= jacobian_ad_forward_enzyme_cache_upwind_reuse(x);

function jacobian_ad_forward_enzyme_cache_upwind_reuse(x::AbstractVector)

    CHUNKSIZE = 12
    N = CHUNKSIZE

    u_ode = zeros(length(x))
    du_ode = zeros(length(x))

    xlen = length(x)
    remainder = xlen % N
    lastchunksize = ifelse(remainder == 0, N, remainder)
    lastchunkindex = xlen - lastchunksize + 1
    middlechunks = 2:div(xlen - lastchunksize, N)
    dys = zeros(length(du_ode), length(du_ode))
    dx = Tuple(zeros(size(x)) for _ in 1:N)
    dy = Tuple(zeros(size(x)) for _ in 1:N)
    for i = 1:N
        dx[i][i] = 1.0
    end
    Enzyme.autodiff(Forward, upwind!, BatchDuplicated(du_ode, dy), BatchDuplicated(u_ode, dx))
    for i = 1:N
        dys[:, i] .= dy[i]
        dx[i][i] = 0.0
    end

    for c in middlechunks
        i = ((c - 1) * N + 1)
        for j = 1:N
            dx[j][j+i-1] = 1.0
        end
        #copyto!(dys, CartesianIndices((1:size(dys, 1), i:i+N-1)), dy, CartesianIndices(dy))
        Enzyme.autodiff(Forward, upwind!, BatchDuplicated(du_ode, dy), BatchDuplicated(u_ode, dx))
        for j = 1:N
            dys[:, i+j-1] .= dy[j]
        end
        for j = 1:N
            dx[j][j+i-1] = 0.0
        end
    end

    dx = Tuple(zeros(size(x)) for _ in 1:lastchunksize)
    dy = Tuple(zeros(size(x)) for _ in 1:lastchunksize)
    for j = 1:lastchunksize
        dx[j][j+lastchunkindex-1] = 1.0
    end
    Enzyme.autodiff(Forward, upwind!, BatchDuplicated(du_ode, dy), BatchDuplicated(u_ode, dx))
    for j = 1:lastchunksize
        dys[:, lastchunkindex+j-1] .= dy[j]
    end

    return dys
end
function gradients_ad_forward_enzyme_cache_upwind2(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    dys = zeros(length(du_ode), length(du_ode))

    for i in 1:length(x)
        dx[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx))
        dys[:, i] .= dy
        dx[i] = 0.0
    end
    return dys
end

# %%
# https://github.com/EnzymeAD/Enzyme.jl/pull/1545/files
function pick_batchsize(x)
    totalsize = length(x)
    return min(totalsize, 12)
end

function jacobian_ad_forward_enzyme_cache_upwind(x::AbstractVector, ::Val{chunk};
    dy = chunkedonehot(x, Val(chunk)),
    dx = chunkedonehot(x, Val(chunk))
    ) where {chunk}
   if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))

    tmp = ntuple(length(dx)) do i
        Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.BatchDuplicated(du_ode, dy[i]), Enzyme.BatchDuplicated(u_ode, dx[i]))
        dy[i]
    end

    cols = Enzyme.tupleconcat(tmp...)
    return reduce(hcat, cols)
end
# %%

@time J2 =jacobian_ad_forward_enzyme_cache_upwind(x, Val(pick_batchsize(x)));
@btime J1=gradients_ad_forward_enzyme_cache_upwind2($x);
J1=gradients_ad_forward_enzyme_cache_upwind2(x)
@time J1=gradients_ad_forward_enzyme_cache_upwind(x); # 0.000217 seconds (8 allocations: 326.328 KiB)
@time jacobian_ad_forward_enzyme_cache_upwind(x);


@btime J2 =jacobian_ad_forward_enzyme_cache_upwind($x, Val(pick_batchsize(x)));

# %%

function gradients_ad_reverse_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    # cache = (;v=1.0, numerical_flux=zeros(length(x)))
    numerical_flux=zeros(length(x))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    dxs = zeros(length(du_ode), length(du_ode))
    # cache_shadow = (;v=1.0,numerical_flux= zeros(length(x)))
    # show(stdout, "text/plain", dy)
    numerical_flux_shadow=zeros(length(x))

    # cache_shadow.numerical_flux[1] = 0.0
    for i in 1:length(x)
        dy[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Reverse, upwind!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx), Const(1.0),Enzyme.Duplicated(numerical_flux, numerical_flux_shadow))
        dxs[:, i] .= dx # it costs more time
        dx .= 0 # important! TODO it costs time
    end
    return dxs'
end

# %%

function gradients_ad_reverse_enzyme_cache_upwind2(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    # cache = (;v=1.0, numerical_flux=zeros(length(x)))
    numerical_flux=zeros(length(x))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    dxs = zeros(length(du_ode), length(du_ode))
    # cache_shadow = (;v=1.0,numerical_flux= zeros(length(x)))
    # show(stdout, "text/plain", dy)
    numerical_flux_shadow=zeros(length(x))
    velocity = 1.0

    inner_func!(du, u, numerical_flux) = let
        v = velocity
        cache = (;v=velocity, numerical_flux=numerical_flux)
        upwind!(du, u, cache)
    end

    # cache_shadow.numerical_flux[1] = 0.0
    for i in 1:length(x)
        dy[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Reverse, inner_func!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx),Enzyme.Duplicated(numerical_flux, numerical_flux_shadow))
        dxs[:, i] .= dx # it costs more time
        dx .= 0 # important! TODO it costs time
    end
    println("hello")
    return dxs'
end
# %%

x = -1.0:0.01:1.0
gradients_ad_reverse_enzyme_cache_upwind(x)
@btime J1=gradients_ad_forward_enzyme_cache_upwind($x);

@btime J1=gradients_ad_forward_enzyme_cache_upwind($x);
@btime J2=gradients_ad_reverse_enzyme_cache_upwind($x);
J1 == J2


# %%

# ForwardDiff

x = -1.0:0.01:1.0
u = zeros(length(x))
du = zeros(length(x))

# %%

@time let
cfg = ForwardDiff.JacobianConfig(nothing, du, u, ForwardDiff.Chunk(12))

J = ForwardDiff.jacobian(upwind!, du, u, cfg);

end; # 0.000183 seconds (42 allocations: 707.375 KiB)

# %%
@time let
cfg = ForwardDiff.JacobianConfig(nothing, du, u, ForwardDiff.Chunk(1))

J = ForwardDiff.jacobian(upwind!, du, u, cfg);

end; # 0.000374 seconds (207 allocations: 988.234 KiB)

# %%
@time let
cfg = ForwardDiff.JacobianConfig(nothing, du, u, ForwardDiff.Chunk(12))

uEltype = eltype(cfg)
nan_uEltype=convert(uEltype, NaN)
numerical_flux=fill(nan_uEltype, length(u))

J = ForwardDiff.jacobian(du, u, cfg) do du_ode, u_ode
    upwind!(du_ode, u_ode, (;v=1.0, numerical_flux))
end

end; # 0.023251 seconds (12.53 k allocations: 1.246 MiB, 98.24% compilation time)

# %%
@time let
cfg = ForwardDiff.JacobianConfig(nothing, du, u, ForwardDiff.Chunk(1))

uEltype = eltype(cfg)
nan_uEltype=convert(uEltype, NaN)
numerical_flux=fill(nan_uEltype, length(u))

J = ForwardDiff.jacobian(du, u, cfg) do du_ode, u_ode
    upwind!(du_ode, u_ode, (;v=1.0, numerical_flux))
end

end; # 0.326764 seconds (1.09 M allocations: 75.013 MiB, 7.28% gc time, 99.87% compilation time)
