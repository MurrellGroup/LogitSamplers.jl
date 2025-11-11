"""
    logitsample([rng], logits; dims=:)

Sample indices from a logit distribution using the Gumbel argmax trick.

See also [`logitsample_categorical`](@ref).

# Examples

```jldoctest
julia> logitsample([-Inf, -10.0])
2

julia> logits = [-Inf -10
                   30  10];

julia> logitsample(logits)
CartesianIndex(2, 1)

julia> logitsample(logits, dims=1)
1×2 Matrix{CartesianIndex{2}}:
 CartesianIndex(2, 1)  CartesianIndex(2, 2)

julia> logitsample(logits, dims=2)
2×1 Matrix{CartesianIndex{2}}:
 CartesianIndex(1, 2)
 CartesianIndex(2, 1)
```
"""
function logitsample(rng::AbstractRNG, x::AbstractArray; dims=:)
    gumbel_noise = GumbelNoise(; rng)
    return argmax(gumbel_noise(x); dims)
end

logitsample(x::AbstractArray; kws...) = logitsample(Random.default_rng(), x; kws...)

function get_tokens(indices; dims::Int=1)
    getdims = t -> t[dims]
    return getdims.(Tuple.(indices))
end

"""
    logitsample_categorical([rng], logits; dims::Int=1)

Sample indices from a logit distribution using the Gumbel argmax trick,
and return the corresponding indices over the specified dimension.

See also [`logitsample`](@ref).

# Examples

```jldoctest
julia> logitsample_categorical([-Inf, -10.0])
1-element Vector{Int64}:
 2

julia> logits = [-Inf -10
                   30  10];

julia> logitsample_categorical(logits) # dims=1 by default
1×2 Matrix{Int64}:
 2  2

julia> logitsample_categorical(logits, dims=1)
1×2 Matrix{Int64}:
 2  2

julia> logitsample_categorical(logits, dims=2)
2×1 Matrix{Int64}:
 2
 1
```
"""
function logitsample_categorical(args...; dims::Int=1)
    indices = logitsample(args...; dims)
    return get_tokens(indices; dims)
end
