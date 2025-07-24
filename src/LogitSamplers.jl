module LogitSamplers

using NNlib: softmax, logsoftmax
using Random
using Statistics: std

include("sample.jl")
export logitsample

include("transforms.jl")
export LogitTransform
export Temperature
export Top_pk, Top_p, Top_k
export Min_p
export Top_nσ

include("samplers.jl")
export argmax_sampler, top_pk_sampler, min_p_sampler, top_nσ_sampler

end
