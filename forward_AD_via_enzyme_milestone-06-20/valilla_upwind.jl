#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    
=#

using Enzyme

const C = 0.2
const Δx= 0.007
# C = Δt/Δx
const Δt = Δx * C

function initial_condition!(u::Vector, x::AbstractVector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
    return nothing
end

struct Cache{T}
    v::T # velocity
    numerical_flux::Vector{T}
end


function upwind!(du::Vector, u::Vector, cache::Cache)
    cache.numerical_flux .= u .* cache.v
	for i = 2:length(u)
        du[i] = - C * (cache.numerical_flux[i] - cache.numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (cache.numerical_flux[1] - cache.numerical_flux[end])
end

function jacobian_ad_forward_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    cache = Cache(1.0, zeros(length(x)))

    dy = Tuple(zeros(size(du_ode)) for _ in 1:length(u_ode))
    dx = Enzyme.onehot(u_ode)
    cache_shadows =Tuple(Cache(1.0, zeros(length(x))) for i=1:length(x))

    # cache is passed to upwind!
    Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.BatchDuplicated(du_ode, dy), Enzyme.BatchDuplicated(u_ode, dx), Enzyme.BatchDuplicated(cache, cache_shadows))
    return stack(dy)
end

function gradients_ad_forward_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    cache = Cache(1.0, zeros(length(x)))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    cache_shadow = Cache(1.0, zeros(length(x)))
    dys = zeros(length(du_ode), length(du_ode))

    for i in 1:length(x)
        dx .= 0.0
        dx[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx), Enzyme.Duplicated(cache, cache_shadow))
        dys[:, i] .= dy
    end
    return dys
end

x = -1.0:0.01:1.0
J1 = jacobian_ad_forward_enzyme_cache_upwind(x)
J2 = gradients_ad_forward_enzyme_cache_upwind(x)


