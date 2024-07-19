struct MyStruct
    cell_ids::Vector
end

function test_func1(x::Vector, c::Vector)
    tmp = MyStruct(zero(x))
    tmp_const = 1:length(x)
    test_func2(tmp, tmp_const)
    return c[1]*x[1]^2 + c[2]*x[2]
end

function test_func2(tmp::MyStruct, x)
    tmp.cell_ids .= x
    return nothing
end

let
    x = rand(2)
    bx = zero(x)
    c = collect(1:2)
    Enzyme.autodiff(Reverse, test_func1, Duplicated(x, bx), Const(c))
    bx
end
