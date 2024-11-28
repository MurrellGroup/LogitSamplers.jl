@deprecate argmax_sampler(logits; device=identity) Top_k(1)(device(logits))
@deprecate argmax_sampler(; kwargs...) logits -> argmax_sampler(logits; kwargs...)

@deprecate top_pk_sampler(logits; p = 0.5f0, k = 5, device = identity) Top_pk(p, k)(device(logits))
@deprecate top_pk_sampler(; kwargs...) logits -> top_pk_sampler(logits; kwargs...)

@deprecate min_p_sampler(logits; pbase = 0.5f0, device = identity) Min_p(pbase)(device(logits))
@deprecate min_p_sampler(; kwargs...) logits -> min_p_sampler(logits; kwargs...)

@deprecate top_nσ_sampler(logits; temperature = 1.0f0, n = 1.0f0, device = identity) Top_nσ(temperature, n)(device(logits))
@deprecate top_nσ_sampler(; kwargs...) logits -> top_nσ_sampler(logits; kwargs...)
