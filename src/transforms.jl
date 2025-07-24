abstract type LogitTransform <: Function end

Base.show(io::IO, ::MIME"text/plain", t::LogitTransform) = show(io, t)


"""
    Temperature(T)

A logit transform that scales (divides) the logits by a temperature parameter.
"""
mutable struct Temperature{T<:Real} <: LogitTransform
    T::T
end

(t::Temperature)(logits::AbstractVector{T}) where T = logits ./ T(t.T)


"""
    Top_pk(p, k)

A logit transform that masks out logits outside the top `p` cumulative probability *or* top `k` logits.
"""
mutable struct Top_pk{P<:Real,K<:Union{Integer,Nothing}} <: LogitTransform
    p::P
    k::K
end

_sort(x::AbstractVector; dims, kwargs...) = sort(x; kwargs...)
_sort(x::AbstractArray; kwargs...) = sort(x; kwargs...)
function (t::Top_pk)(logits::AbstractArray{T}) where T<:AbstractFloat
    0 < t.p <= 1 || throw(DomainError(t.p, "p must be in the interval (0, 1]"))
    probs = softmax(logits, dims=1)
    sorted_probs = _sort(probs, dims=1, rev=true)
    cutoff_k = t.k isa Integer ? copy(selectdim(sorted_probs, 1, t.k:t.k)) : zero(T)
    sorted_probs[cumsum(sorted_probs, dims=1) .< t.p] .= 0
    cutoff_p = maximum(sorted_probs, dims=1, init=zero(T))
    return ifelse.(probs .>= max.(cutoff_p, cutoff_k), logits, T(-Inf))
end

Top_p(p) = Top_pk(p, nothing)
Top_k(k) = Top_pk(1, k)


"""
    Min_p(pbase)

A logit transform that masks out logits below `pbase` times the maximum probability.

See: https://arxiv.org/pdf/2407.01082
"""
mutable struct Min_p{T<:Real} <: LogitTransform
    pbase::T
end

function (t::Min_p)(logits::AbstractArray{T}) where T<:AbstractFloat
    logp = logsoftmax(logits, dims=1)
    return ifelse.(logp .>= log(t.pbase) .+ maximum(logp, dims=1), logits, T(-Inf))
end


"""
    Top_nσ(n)

A logit transform that masks out logits below `n` standard deviations of the maximum logit.

Top-nσ is temperature-invariant, i.e. the candidate set does not change with temperature.

See: https://arxiv.org/pdf/2411.07641
"""
mutable struct Top_nσ{T<:Real} <: LogitTransform
    n::T
end

function (t::Top_nσ)(logits::AbstractArray{T}) where T<:AbstractFloat
    M, σ = maximum(logits, dims=1), std(logits, dims=1)
    return ifelse.(logits .>= M .- t.n .* σ, logits, T(-Inf))
end
