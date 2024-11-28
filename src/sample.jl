"""
    logitsample([rng], logits, [buffer=similar(logits)]) -> Int

Sample an index from a logit distribution using the Gumbel argmax trick.

Alternatively pass a buffer to avoid allocating a new array when creating
the random numbers.
"""
function logitsample(rng::AbstractRNG, x::AbstractVector{T}, u::AbstractVector{T}=similar(x)) where T<:AbstractFloat
    length(x) == length(u) || throw(DimensionMismatch("Expected buffer of same length as logits"))
    rand!(rng, u)
    argmax(-log.(-log.(u)) + x)
end

@inline logitsample(args...) = logitsample(Random.default_rng(), args...)
