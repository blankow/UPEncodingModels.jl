

# ============================================================================
# utils.jl: Utility functions for GLM encoding models
# ============================================================================

"""
    extract_encoding_weights(fit::GLMFit, dm::DesignMatrix,
        factor::Symbol, n_levels::Int, encoding_basis::RaisedCosineBasis
    ) → Matrix{Float64}

Reconstruct the time-varying encoding weight β_f(τ) for each level of a
discrete factor.

Returns a (T × n_levels) matrix, where T is the number of time points
in the encoding basis, and each column is the reconstructed coefficient
time course for one factor level.

The raw parameters for factor f with C levels are stored as
(C−1) × n_basis weights (effect-coded). To recover the weight for each
level, we reconstruct from the effect coding:

  β_c(τ) = Σ_l  w_{c,l} · b_l(τ)     for c = 1, …, C−1
  β_C(τ) = −Σ_{c=1}^{C-1} β_c(τ)      (reference level, from sum-to-zero)
"""
function extract_encoding_weights(
    fit::GLMFit,
    dm::DesignMatrix,
    factor::Symbol,
    n_levels::Int,
    encoding_basis::RaisedCosineBasis
)
    cols = dm.groups[factor]
    n_enc = encoding_basis.n_basis
    T = size(encoding_basis.B, 1)

    # Extract raw basis weights: (n_levels−1) × n_enc
    raw_weights = reshape(fit.w[cols], n_enc, n_levels - 1)'  # (n_levels-1) × n_enc

    # Reconstruct time courses
    β = zeros(T, n_levels)
    for c in 1:(n_levels - 1)
        β[:, c] = encoding_basis.B * raw_weights[c, :]
    end
    # Reference level = negative sum of others
    β[:, n_levels] = -sum(β[:, 1:(n_levels-1)], dims=2)[:]

    return β
end


"""
    extract_spike_history_filter(fit::GLMFit, dm::DesignMatrix,
        history_basis::RaisedCosineBasis
    ) → Vector{Float64}

Reconstruct the spike history filter h(δ) from the fitted basis weights.
Returns a vector of length equal to history_basis.timepoints.

Interpreting the result:
  • Negative values at short lags (1–5 ms) = refractoriness ✓
  • Positive bump at 2–5 ms = bursting
  • Slow negative component at 10–200 ms = adaptation
  • Filter should be net negative for stability
"""
function extract_spike_history_filter(
    fit::GLMFit,
    dm::DesignMatrix,
    history_basis::RaisedCosineBasis
)
    cols = dm.groups[:history]
    h_weights = fit.w[cols]
    return history_basis.B * h_weights
end

"""
    compute_local_drift(trial_rates_hz, window=5) → Vector{Float64}

For each trial t, compute the log of the mean firing rate across the
`window` neighboring trials on each side (excluding trial t itself).
Used as a per-trial constant offset in the model to absorb slow drift
in baseline firing rate.

For edge trials (first/last `window` trials), uses only the available
neighbors in that direction.

Returns a vector of log-rate offsets, one per trial.
"""
function compute_local_drift(
    trial_rates_hz::Vector{Float64},
    window::Int = 5
)
    n = length(trial_rates_hz)
    log_offsets = Vector{Float64}(undef, n)

    for t in 1:n
        lo = max(1, t - window)
        hi = min(n, t + window)
        neighbor_rates = vcat(
            lo <= t-1 ? trial_rates_hz[lo:t-1] : Float64[],
            t+1 <= hi ? trial_rates_hz[t+1:hi] : Float64[]
        )
        mean_rate = isempty(neighbor_rates) ? trial_rates_hz[t] :
                        sum(neighbor_rates) / length(neighbor_rates)
        log_offsets[t] = log(max(mean_rate, 1e-6))
    end

    return log_offsets
end


# ============================================================================
# EXTRACT EVENT KERNEL (updated for per-event basis)
# ============================================================================

"""
    extract_event_kernel(fit, dm, event_name, event_basis) → NamedTuple

Reconstruct the event response kernel κ(δ) from the fitted parameters.

Returns:
- `lags`: vector of lag times (ms) from the event basis
- `kernel`: the reconstructed kernel h(δ) = B · w_event

For click events, this kernel represents the average response to a
single click. Because the model is linear in log-rate space, the
response to N overlapping clicks is N × kernel (in log-rate), which
corresponds to a multiplicative scaling of the firing rate by exp(kernel)^N.
"""
function extract_event_kernel(
    fit,  # GLMFit
    dm,   # DesignMatrix
    event_name::Symbol,
    event_basis::RaisedCosineBasis
)
    cols = dm.groups[event_name]
    w_evt = fit.w[cols]
    kernel = event_basis.B * w_evt

    return (
        lags = event_basis.timepoints,
        kernel = kernel
    )
end