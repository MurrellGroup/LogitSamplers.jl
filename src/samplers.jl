#To do: refactor into a combination of modified_softmax and sample. This way we can viz the result of the modified logits without having to sample.
#This won't be visible to the user. Any method that doesn't fit this interface can be implemented directly.

function argmax_sampler(logits::AbstractVector; device = identity)
    return argmax(device(logits))
end

"""
    argmax_sampler(; device = identity)

Returns a function that samples most likely token.
"""
argmax_sampler(; device = identity) = logits -> argmax_sampler(logits; device = device)

function top_pk_sampler(logits::AbstractVector; p = 0.5f0, k = 5, device = identity)
    probs = device(softmax(logits))
    perm = partialsortperm(probs, 1:k, rev=true)
    sorted_probs = probs[perm]
    cumsum_probs = cumsum(sorted_probs)
    if cumsum_probs[1] > p
        return perm[1]
    else
        cutoff = findlast(cumsum_probs .< p)
        return sample(perm[1:cutoff], Weights(sorted_probs[1:cutoff]))
    end
end

"""
    top_pk_sampler(; p = 0.5f0, k = 5, device = identity)

Returns a function that samples from the most probable tokens. A combination of the top-k and top-p sampling methods, where you can sample from the top tokens with cumulative probability `p`, with a max number of tokens `k`.
"""
top_pk_sampler(;p = 0.5f0, k = 5, device = identity) = logits -> top_pk_sampler(logits; p, k, device)

#https://arxiv.org/pdf/2407.01082
function min_p_sampler(logits::AbstractVector{T}; pbase::T = 0.5f0, device = identity) where T
    probs = device(softmax(logits))
    pmax = maximum(probs)
    pscaled = pbase * pmax
    mask = probs .>= pscaled
    if !any(mask)
        mask[argmax(probs)] = true
    end
    probs[.!mask] .= zero(T)
    return sample(1:length(probs), Weights(probs))
end

"""
    min_p_sampler(; pbase = 0.5f0, device = identity)

Returns a function that samples from the most probable tokens using the min-p strategy. See: https://arxiv.org/pdf/2407.01082
"""
min_p_sampler(; pbase = 0.5f0, device = identity) = logits -> min_p_sampler(logits; pbase, device)

# https://arxiv.org/pdf/2411.07641
function top_nσ_sampler(logits::AbstractVector{T}; temperature::T = 1.0f0, n::T = 1.0f0, device = identity) where T
    scaled_logits = logits ./ temperature
    M = maximum(scaled_logits)
    σ = std(scaled_logits)
    threshold = M - n * σ
    mask = scaled_logits .>= threshold
    scaled_logits[.!mask] .= -Inf
    probs = device(softmax(scaled_logits))
    return sample(1:length(probs), Weights(probs))
end

"""
    top_nσ_sampler(; temperature = 1.0f0, n = 1.0f0, device = identity)

Returns a function that samples from the most probable tokens using the top-nσ strategy. See: https://arxiv.org/pdf/2411.07641
"""
top_nσ_sampler(; temperature = 1.0f0, n = 1.0f0, device = identity) = logits -> top_nσ_sampler(logits; temperature, n, device)
