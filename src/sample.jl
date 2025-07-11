_index_fix(i, ::Colon) = i
_index_fix(i, dims::Int) = CartesianIndex(i)[dims]
_index_fix(i::CartesianIndex, dims::Tuple{Vararg{Int}}) = CartesianIndex(getindex.(i, dims)...)

"""
    logitsample([rng], logits)

Sample an index from a logit distribution using the Gumbel argmax trick.
"""
function logitsample(rng::AbstractRNG, x::AbstractArray, u=similar(x); dims=:)
    indices = argmax(.-log.(.-log.(rand!(rng, u))) .+ x; dims)
    return _index_fix.(indices, Ref(dims))
end

logitsample(xs::AbstractArray...; kws...) = logitsample(Random.default_rng(), xs...; kws...)
