@deprecate argmax_sampler(logits; device=identity) logits |> device |> Top_k(1) |> logitsample
@deprecate argmax_sampler(; kwargs...) logits -> argmax_sampler(logits; kwargs...)

@deprecate top_pk_sampler(logits; p = 0.5f0, k = 5, device = identity) logits |> device |> Top_pk(p, k) |> logitsample
@deprecate top_pk_sampler(; kwargs...) logits -> top_pk_sampler(logits; kwargs...)

@deprecate min_p_sampler(logits; pbase = 0.5f0, device = identity) logits |> device |> Min_p(pbase) |> logitsample
@deprecate min_p_sampler(; kwargs...) logits -> min_p_sampler(logits; kwargs...)

@deprecate top_nσ_sampler(logits; temperature = 1.0f0, n = 1.0f0, device = identity) logits |> device |> Temperature(temperature) |> Top_nσ(n) |> logitsample
@deprecate top_nσ_sampler(; kwargs...) logits -> top_nσ_sampler(logits; kwargs...)
