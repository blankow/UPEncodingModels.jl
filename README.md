# UPEncodingModels

[![Build Status](https://github.com/blankow/UPEncodingModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/blankow/UPEncodingModels.jl/actions/workflows/CI.yml?query=branch%3Amain)


# A Poisson GLM pipeline for neural encoding analysis with:
##   • Raised cosine (log-time) basis for spike history
##    • Raised cosine (linear-time) basis for task encoding
##   • Full design matrix construction
##   • Ridge-regularized Poisson GLM fitting via IRLS and L-BFGS
##    • Deviance-based variance decomposition
##    • Cross-validated effect size estimation