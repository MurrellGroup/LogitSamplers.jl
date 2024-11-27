"""
    logitsample([rng], logits) -> Int

Sample an index from a logit distribution using the Gumbel argmax trick.
"""
function logitsample(rng::AbstractRNG, x::AbstractVector)
    u = rand!(rng, similar(x))
    argmax(-log.(-log.(u)) + x)
end

logitsample(x::AbstractVector) = logitsample(Random.default_rng(), x)
