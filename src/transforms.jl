abstract type LogitTransform <: Function end

"""
    Top_p(p = 0.5f0, k = 5)

Returns a function that samples from the most probable tokens.
A combination of the top-p and top-k sampling methods, where you can sample from
the top tokens with cumulative probability `p`, with a max number of tokens `k`.
"""
struct Top_p{T} <: LogitTransform
    p::T
    k::Int
end

function (t::Top_p)(logits::AbstractVector{T}) where T
    logits = convert(Vector, logits)
    top_k_indices = partialsortperm(logits, 1:min(t.k, length(logits)), rev=true)
    top_k_logits = logits[top_k_indices]
    top_k_probs = NNlib.softmax(top_k_logits)
    last_index = findfirst(>(T(t.p)), cumsum(top_k_probs))
    top_p_indices = isnothing(last_index) ? top_k_indices : top_k_indices[1:last_index]
    mask = create_mask(logits, top_p_indices)
    return apply_mask(logits, mask)
end


"""
    Min_p(pbase)

Returns a function that samples from the most probable tokens using the min-p strategy.

See: https://arxiv.org/pdf/2407.01082
"""
struct Min_p{T} <: LogitTransform
    pbase::T
end

function (f::Min_p)(logits::AbstractVector{T}) where T
    p = NNlib.softmax(logits)
    mask = p .> T(f.pbase) * maximum(p)
    return apply_mask(logits, mask)
end


"""
    Top_nσ(temp, n)

Returns a function that samples from the most probable tokens using the top-nσ strategy.

See: https://arxiv.org/pdf/2411.07641
"""
struct Top_nσ{T} <: LogitTransform
    T::T
    n::T
end

function (f::Top_nσ)(logits::AbstractVector)
    logits′ = logits / f.T
    M, σ = maximum(logits′), std(logits′)
    mask = logits′ .>= M - f.n * σ
    return apply_mask(logits, mask)
end
