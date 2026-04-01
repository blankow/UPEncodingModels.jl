

# ============================================================================
# predictions.jl: Functions for generating predictions and evaluating model fit
# ============================================================================

"""
    predict(fit::GLMFit, X::Matrix{Float64}; type=:rate) → Vector{Float64}

Generate predictions from a fitted model.

- `type = :rate`   → returns λ̂ = exp(Xw)  (firing rate in Hz)
- `type = :count`  → returns μ̂ = Δ · exp(Xw)  (expected spike count per bin)
- `type = :linear` → returns Xw  (log-rate, i.e. the linear predictor)
"""
function predict(
    fit::GLMFit,
    X::Matrix{Float64};
    type::Symbol = :rate
)
    η = X * fit.w
    if type == :linear
        return η
    elseif type == :rate
        return exp.(η)
    elseif type == :count
        error("For count predictions, use predict_count(fit, X, Δ)")
    else
        error("Unknown prediction type: $type")
    end
end


"""
    null_deviance(y, Δ) → Float64

Deviance of the null model (constant rate = mean firing rate).
"""
function null_deviance(y::Vector{Int}, Δ::Float64)
    μ_bar = mean(y)  # mean count per bin
    λ_null = μ_bar / Δ
    λ_vec = fill(λ_null, length(y))
    return poisson_deviance(y, λ_vec, Δ)
end

# Inline mean to avoid depending on Statistics.jl
function mean(x)
    return sum(x) / length(x)
end


"""
    deviance_explained(y, λ_hat, Δ) → Float64

Fraction of deviance explained (pseudo-R²):

    1 − D_model / D_null

where D_null is the deviance of a constant-rate model.
"""
function deviance_explained(
    y::Vector{Int},
    λ_hat::Vector{Float64},
    Δ::Float64
)
    D_model = poisson_deviance(y, λ_hat, Δ)
    D_null  = null_deviance(y, Δ)
    return 1.0 - D_model / D_null
end