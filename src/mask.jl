apply_mask(x::AbstractVector{T}, mask::AbstractVector{Bool}) where T<:AbstractFloat =
    T(-Inf) * .!mask + x

function create_mask(x::AbstractVector, indices::AbstractVector{Int})
    mask = similar(x, Bool) .= false
    mask[indices] .= true
    return mask
end