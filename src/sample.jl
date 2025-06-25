_index_fix(i, ::Colon) = i
_index_fix(i, dims::Int) = CartesianIndex(i)[dims]
_index_fix(i::CartesianIndex, dims::Tuple{Vararg{Int}}) = CartesianIndex(getindex.(i, dims)...)

"""
    logitsample([rng], logits)

Sample an index from a logit distribution using the Gumbel argmax trick.
"""
function logitsample(rng::AbstractRNG, x::AbstractArray; dims=:)
    u = rand!(rng, similar(x))
    indices = argmax(.-log.(.-log.(u)) .+ x; dims)
    return _index_fix.(indices, Ref(dims))
end

logitsample(x; kws...) = logitsample(Random.default_rng(), x; kws...)
