"""
    logitsample([rng], logits) -> Int

Sample an index from a logit distribution using the Gumbel argmax trick.
"""
function logitsample(rng::AbstractRNG, x::AbstractVector)
    u = rand!(rng, similar(x))
    argmax(-log.(-log.(u)) + x)
end

logitsample(x::AbstractVector) = logitsample(Random.default_rng(), x)


apply_mask(x::AbstractVector{T}, mask::AbstractVector{Bool}) where T<:AbstractFloat =
    T(-Inf) * .!mask + x

function create_mask(x::AbstractVector, indices::AbstractVector{Int})
    mask = similar(x, Bool) .= false
    mask[indices] .= true
    return mask
end
