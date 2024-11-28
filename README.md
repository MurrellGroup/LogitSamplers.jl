# LogitSamplers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/LogitSamplers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/LogitSamplers.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/LogitSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/LogitSamplers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/LogitSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/LogitSamplers.jl)

A Julia package for GPU-friendly sampling from logit distributions with various transformation methods commonly used in language models.

## Usage

The package provides a set of logit transforms to modify the distributions in the log domain.

```julia
using LogitSamplers

# Create a temperature transform
temperature = Temperature(1.5)

# Create a top-p transform
top_p = Top_p(0.5)

# Compose a function that first applies temperature, then top-p
transform = top_p ∘ temperature

# Create a token index sampler function from the transform
sampler = logitsample ∘ transform

# or equivalently:
sampler = logits -> logitsample(top_p(temperature(logits)))

logits = randn(100)

# Get token probabilities from the sampler
probs = softmax(transform(logits))

# Sample a logit index from the sampler
index = sampler(logits)
```
