using BenchmarkTools
using DifferentiationInterface
import ForwardDiff, Enzyme  # AD backends you want to use 

include("DG.jl")

function bar!(du_ode, u0_ode)
    u0 = reshape(u0_ode, :, 16)
    du = reshape(du_ode, :, 16)
    rhs!(du, u0)
    return nothing
end
function my_jacobian_forward!(f!::Function, y::AbstractVector, x::AbstractVector)
    dx = Enzyme.onehot(x)
    dy = Tuple(zeros(size(x)) for _ in 1:length(x))
    Enzyme.autodiff(Enzyme.Forward, f!, Enzyme.BatchDuplicated(y, dy), Enzyme.BatchDuplicated(x, dx))
    return stack(dy)
end
function my_jacobian_reverse!(f!::Function, y::AbstractVector, x::AbstractVector)
    bx = Tuple(zeros(size(x)) for _ in 1:length(x))
    by = Enzyme.onehot(y)
    Enzyme.autodiff(Enzyme.Reverse, f!, Enzyme.BatchDuplicated(y, by), Enzyme.BatchDuplicated(x, bx))
    return stack(bx)'
end

du = similar(u0)

du = zeros(size(u0))
du_ode = vec(du)
u0_ode = vec(u0)

f = bar!

# %%

du .= zero(eltype(du))

my_jacobian_reverse!(f, du_ode, u0_ode) # 1.246 ms (19911 allocations: 1.63 MiB)

my_jacobian_forward!(f, du_ode, u0_ode) # 1.246 ms (19911 allocations: 1.63 MiB)

# @btime my_jacobian_reverse!($f, $du_ode, $u0_ode) # 1.246 ms (19911 allocations: 1.63 MiB)

ForwardDiff.jacobian(f, du_ode, u0_ode)

@btime ForwardDiff.jacobian($f, $du_ode, $u0_ode)
@btime jacobian($f, $du_ode, AutoForwardDiff(), $u0_ode)
@btime jacobian($f, $du_ode,  AutoEnzyme(;mode=Enzyme.Forward), $u0_ode) # 3.213 ms  (45680 allocations: 2.94 MiB)
@btime my_jacobian_forward!($f, $du_ode, $u0_ode) # 1.246 ms (19911 allocations: 1.63 MiB)
