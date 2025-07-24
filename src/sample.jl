_index_fix(i, ::Colon) = i
_index_fix(i::Union{Int,CartesianIndex}, dims::Int) = CartesianIndex(i)[dims]
_index_fix(i::CartesianIndex, dims::Tuple{Vararg{Int}}) = CartesianIndex(getindex.((i,), dims)...)
_index_fix(i::AbstractArray, dims) = _index_fix.(i, (dims,))

"""
    logitsample([rng], logits; dims=:)

Sample an index from a logit distribution using the Gumbel argmax trick.

# Examples

```jldoctest
julia> logitsample([-Inf, -10.0])
2

julia> logitsample([-Inf -10.0; 20 -Inf])
CartesianIndex(2, 1)

julia> logitsample([-Inf -10.0; 20 -Inf], dims=1)
1×2 Matrix{Int64}:
 2  1
```
"""
function logitsample(rng::AbstractRNG, x::AbstractArray, u=similar(x); dims=:)
    indices = argmax(.-log.(.-log.(rand!(rng, u))) .+ x; dims)
    return _index_fix(indices, dims)
end

logitsample(xs::AbstractArray...; kws...) = logitsample(Random.default_rng(), xs...; kws...)
