module LogitSamplers

using NNlib: softmax, logsoftmax
using Random
using Statistics: std

include("sample.jl")
export logitsample
export logitsample_categorical

include("transforms.jl")
export LogitTransform
export Temperature
export GumbelNoise
export Top_pk, Top_p, Top_k
export Min_p
export Top_nσ

end
