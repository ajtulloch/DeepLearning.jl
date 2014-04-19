import Base.length
import Base.show
import Base.size

flatten(arr) = reshape(arr, length(arr))

type Vol
    w::Array{Float64, 3}
    dw::Array{Float64, 3}

    function Vol(dimensions, value::Float64)
        w = Array(Float64, dimensions)
        dw = Array(Float64, dimensions)
        new(ones(dimensions) * value, zeros(dimensions))
    end

    Vol(w::Array{Float64, 3}, dw::Array{Float64, 3}) = new(w, dw)

    function Vol(dimensions)
        (x, y, depth) = dimensions
        scale = sqrt(1.0 / (x * y * depth))
        new(randn(x, y, depth) * scale, zeros(x, y, depth))
    end
end

function show(io::IO, v::Vol)
    println("Vol")
    println(io, "w: ", v.w)
    println(io, "dw: ", v.dw)
end

function length(v::Vol)
    @assert size(v.w) == size(v.dw)
    length(v.w)
end

function size(v::Vol)
    @assert size(v.w) == size(v.dw)
    size(v.w)
end

