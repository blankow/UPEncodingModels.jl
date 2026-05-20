

# ============================================================================
# rcbfs.jl: Raised Cosine Basis Functions for GLM Encoding Models
# ============================================================================

"""
    RaisedCosineBasis

Stores a precomputed basis matrix and its evaluation points.

Fields:
- `B::Matrix{Float64}` — (n_timepoints × n_basis) matrix of basis function values
- `timepoints::Vector{Float64}` — the time values at which the basis was evaluated
- `centers::Vector{Float64}` — the centers φ_l of each basis function (in transformed space)
- `n_basis::Int` — number of basis functions
"""
struct RaisedCosineBasis
    B::Matrix{Float64}
    timepoints::Vector{Float64}
    centers::Vector{Float64}
    n_basis::Int
end


"""
    make_spike_history_basis(;
        n_basis    = 8,
        max_lag_ms = 200.0,
        dt_ms      = 1.0,
        offset     = 5.0       # controls log-spacing near zero
    ) → RaisedCosineBasis

Construct a raised cosine basis on a LOG time axis, following Pillow et al. (2005).

The transformation from lag δ (in ms) to the log-scaled coordinate is:

    φ(δ) = log(δ + offset)

Basis functions are half-cosine bumps placed uniformly in φ-space:

    b_l(δ) = ½ cos( (φ(δ) − c_l) · π / Δc ) + ½

where c_l is the center of the l-th basis function in φ-space, and Δc is the
half-width (= spacing between adjacent centers). The function is nonzero only
when |φ(δ) − c_l| < Δc.

The `offset` parameter controls how compressed the short-lag end is:
  • Small offset (1–3):  very fine resolution at short lags, coarse at long lags.
                         Good for fast-spiking neurons where refractoriness details matter.
  • Large offset (10+):  more uniform spacing, approaching a linear-time basis.
                         Better for neurons with slow dynamics or when short-lag
                         detail is less critical.

The choice of 5.0 as a default is a reasonable middle ground. You should
examine the resulting basis (plot columns of B vs. timepoints) and adjust
if the coverage doesn't match your neurons' autocorrelation timescale.
"""
function make_spike_history_basis(;
    n_basis::Int       = 8,
    max_lag_ms::Float64 = 200.0,
    dt_ms::Float64     = 1.0,
    offset::Float64    = 5.0
)
    # Evaluation points: lags from dt_ms to max_lag_ms in steps of dt_ms
    lags = collect(dt_ms:dt_ms:max_lag_ms)
    n_pts = length(lags)

    # Transform to log-space
    ϕ = log.(lags .+ offset)
    ϕ_min = log(dt_ms + offset)
    ϕ_max = log(max_lag_ms + offset)

    # Place centers uniformly in ϕ-space
    # We extend slightly beyond the range so edge basis functions
    # have support covering the boundary
    centers = range(ϕ_min, ϕ_max, length=n_basis)
    Δc = centers[2] - centers[1]  # half-width = spacing

    # Evaluate each basis function at each lag
    B = zeros(n_pts, n_basis)
    for l in 1:n_basis
        for i in 1:n_pts
            z = (ϕ[i] - centers[l]) / Δc
            if abs(z) < 1.0
                B[i, l] = 0.5 * cos(π * z) + 0.5
            end
        end
    end

    return RaisedCosineBasis(B, lags, collect(centers), n_basis)
end


"""
    make_trial_time_basis(;
        n_basis      = 15,
        trial_dur_ms = 3000.0,
        dt_ms        = 1.0,
        spacing      = :linear    # :linear or :log
    ) → RaisedCosineBasis

Construct a raised cosine basis on a LINEAR (or log) time axis spanning
the trial duration. Used for:
  • Baseline temporal profile h(τ)
  • Time-varying encoding weights β_f(τ)
  • Event response kernels (pass a shorter trial_dur_ms, e.g. 500 ms)

For trial-time encoding, linear spacing is usually appropriate because
the timescale of interest (hundreds of ms to seconds) doesn't span
the orders of magnitude that spike history does. Use :log spacing
if you need finer resolution near trial onset.
"""
function make_trial_time_basis(;
    n_basis::Int         = 15,
    trial_dur_ms::Float64 = 3000.0,
    dt_ms::Float64       = 1.0,
    spacing::Symbol      = :linear
)
    timepoints = collect(0.0:dt_ms:trial_dur_ms)
    n_pts = length(timepoints)

    if spacing == :log
        offset = 10.0
        ϕ = log.(timepoints .+ offset)
        ϕ_min = log(0.0 + offset)
        ϕ_max = log(trial_dur_ms + offset)
    else  # :linear
        ϕ = timepoints
        ϕ_min = 0.0
        ϕ_max = trial_dur_ms
    end

    centers = range(ϕ_min, ϕ_max, length=n_basis)
    Δc = centers[2] - centers[1]

    B = zeros(n_pts, n_basis)
    for l in 1:n_basis
        for i in 1:n_pts
            z = (ϕ[i] - centers[l]) / Δc
            if abs(z) < 1.0
                B[i, l] = 0.5 * cos(π * z) + 0.5
            end
        end
    end

    return RaisedCosineBasis(B, timepoints, collect(centers), n_basis)
end



