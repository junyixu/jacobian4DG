using Trixi
using SimpleUnPack

# equation with a advection_velocity of `1`.
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# create DG solver with flux lax friedrichs and LGL basis
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# distretize domain with `TreeMesh`
coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4, # number of elements = 2^4
                n_cells_max=30_000)

# create initial condition and semidiscretization
initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

tf = fieldnames ∘ typeof

semi.cache |> typeof |> fieldnames
semi.cache[:elements]
semi.cache[:elements] |> tf
semi.cache[:elements].node_coordinates
semi.cache[:elements].surface_flux_values

semi.cache[:elements].node_coordinates

semi.cache[:elements]._node_coordinates

for e in semi.cache[:elements] |> tf
    @eval show(semi.cache[:elements].$e)
end


t0 = zero(real(semi))
u0_ode = compute_coefficients(t0, semi)
u0_ode_plain = reinterpret(eltype(eltype(u0_ode)), u0_ode)
du_ode_plain = similar(u0_ode_plain)

# %%

import Enzyme  # AD backends you want to use 
using ForwardDiff
function my_jacobian_forward!(f!::Function, y::AbstractVector, x::AbstractVector)
    dx = Enzyme.onehot(x)
    dy = Tuple(zeros(size(x)) for _ in 1:length(x))
    Enzyme.autodiff(Enzyme.Forward, f!, Enzyme.BatchDuplicated(y, dy), Enzyme.BatchDuplicated(x, dx))
    return stack(dy)
end
# %%

Trixi.rhs!(du_ode_plain, u0_ode_plain, semi, t0)

f(du, u0) = Trixi.rhs!(du, u0, semi, t0)
my_jacobian_forward!(f, du_ode_plain, u0_ode_plain) # 1.246 ms (19911 allocations: 1.63 MiB)
# ForwardDiff.jacobian(f, du_ode_plain, u0_ode_plain)

J = jacobian_ad_forward(semi);


function foo!(du_ode, u_ode, semi)
    Trixi.rhs!(du_ode, u_ode, semi, t0)
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

u_ode = u0_ode
dot_u_ode = zero(u0_ode)
dot_u_ode[1] = 1.0

du_ode = du_ode_plain
ddu_ode=zero(du_ode)

dx = Enzyme.onehot(u_ode)

dy = Tuple(zeros(size(du_ode)) for _ in 1:length(u_ode))
semi_zero=Enzyme.make_zero(semi)

j1=stack(dy)

@btime Enzyme.autodiff(Enzyme.Forward, $foo!, Enzyme.BatchDuplicated($du_ode, $dy), Enzyme.BatchDuplicated($u_ode, $dx), BatchDuplicated($semi, $tuple_semi))

dot_u_ode = zero(u0_ode)
dot_u_ode[2] = 1.0
autodiff(Forward, foo!, Duplicated(du_ode, ddu_ode), Duplicated(u_ode, dot_u_ode), Duplicated(semi, Enzyme.make_zero(semi)))


tuple_semi=Tuple(Enzyme.make_zero(semi) for i=1:64)

@btime j2=jacobian_ad_forward($semi)

j1 .- j2

t0 = zero(real(semi))
u_ode = compute_coefficients(t0, semi)
du_ode = similar(u_ode)


function my_jacobian_forward!(f!::Function, y::AbstractVector, x::AbstractVector)
    dx = Enzyme.onehot(x)
    dy = Tuple(zeros(size(y)) for _ in 1:length(x))
    Enzyme.autodiff(Enzyme.Forward, f!, Enzyme.BatchDuplicated(y, dy), Enzyme.BatchDuplicated(x, dx))
    return stack(dy)
end
