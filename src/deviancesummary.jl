#=
    deviance_summary.jl

    Helper function that computes the three-part deviance decomposition
    for a fitted Poisson GLM:

        D_total  =  D_explained  +  D_residual

    where:
        D_total     = null deviance (constant-rate model vs. saturated)
        D_explained = deviance accounted for by the model
        D_residual  = Poisson deviance around the model prediction

    Usage:
        include("deviance_summary.jl")

        result = deviance_summary(y, λ_hat, Δ)
        # or equivalently:
        result = deviance_summary(dm, fit; dt_ms=1.0)
=#


"""
    deviance_summary(y::Vector{Int}, λ_hat::Vector{Float64}, Δ::Float64)
        → NamedTuple

Compute the three-part Poisson deviance decomposition.

## Arguments
- `y`: observed spike counts per bin (integers)
- `λ_hat`: predicted firing rates from the fitted model (in Hz)
- `Δ`: time bin width in seconds (e.g., 0.001 for 1-ms bins)

## Returns a NamedTuple with:
- `D_total`: the deviance in the data — total Poisson deviance of a
  constant-rate (null) model relative to the saturated model.
  Measures the total variability in the spike counts.

      D_total = 2 Σᵢ [ yᵢ log(yᵢ / ȳ) − (yᵢ − ȳ) ]

  where ȳ is the mean count per bin.

- `D_explained`: the deviance accounted for by the model prediction —
  the reduction in deviance from null to fitted model.

      D_explained = D_total − D_residual

  This is the Poisson analogue of the model sum of squares. It
  quantifies how much of the total variability the model captures.

- `D_residual`: the Poisson deviance around the model prediction —
  the deviance of the fitted model relative to the saturated model.

      D_residual = 2 Σᵢ [ yᵢ log(yᵢ / μ̂ᵢ) − (yᵢ − μ̂ᵢ) ]

  where μ̂ᵢ = λ̂ᵢ · Δ is the predicted expected count. This is the
  residual variability not explained by the model.

## Relationship

    D_total  =  D_explained  +  D_residual

The fraction of deviance explained is D_explained / D_total, which is
the pseudo-R² that we've been using for variance decomposition.
"""
function deviance_summary(
    y::Vector{Int},
    λ_hat::Vector{Float64},
    Δ::Float64
)
    N = length(y)

    # Mean count per bin (for the null model)
    ȳ = sum(y) / N

    # D_total: null deviance
    # = 2 Σ [ y log(y/ȳ) − (y − ȳ) ]
    D_total = 0.0
    @inbounds for i in 1:N
        if y[i] > 0
            D_total += y[i] * log(y[i] / ȳ) - (y[i] - ȳ)
        else
            D_total += ȳ   # since 0·log(0) = 0 by convention
        end
    end
    D_total *= 2.0

    # D_residual: model deviance
    # = 2 Σ [ y log(y/μ̂) − (y − μ̂) ]  where μ̂ = λ̂·Δ
    D_residual = 0.0
    @inbounds for i in 1:N
        μ = λ_hat[i] * Δ
        μ = max(μ, 1e-12)   # numerical guard
        if y[i] > 0
            D_residual += y[i] * log(y[i] / μ) - (y[i] - μ)
        else
            D_residual += μ
        end
    end
    D_residual *= 2.0

    # D_explained: the difference
    D_explained = D_total - D_residual
    

    return (
        D_total     = D_total,
        D_explained = D_explained,
        D_residual  = D_residual,
        frac_explained = D_explained / D_total,
        n_obs = N,
    )
end


"""
    deviance_summary(dm, fit; dt_ms=1.0)

Convenience method that accepts a DesignMatrix and GLMFit directly.
"""
function deviance_summary(dm, fit; dt_ms::Float64 = 1.0)
    Δ = dt_ms / 1000.0
    λ_hat = exp.(dm.X * fit.w)
    return deviance_summary(dm.y, λ_hat, Δ)
end