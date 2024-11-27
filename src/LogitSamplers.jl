module LogitSamplers

using NNlib
using Random
using Statistics

include("mask.jl")

include("sample.jl")
export logitsample

include("transforms.jl")
export LogitTransform
export Top_p, Min_p, Top_nσ

end
