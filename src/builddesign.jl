

# ============================================================================
# builddesign.jl: Functions to build design matrices for GLM encoding models
# ============================================================================

"""
    TrialData

Container for the data from a single trial. All time-series fields are vectors
of length T (number of time bins in the trial).

Fields:
- `spike_counts::Vector{Int}` — binned spike counts, length T
- `condition::Dict{Symbol, Int}` — maps factor name → level index for this trial
    e.g. Dict(:stimulus => 3, :reward => 1, :choice => 2)
- `event_times::Dict{Symbol, Float64}` — maps event name → time (ms) within trial
    e.g. Dict(:stim_on => 500.0, :go_cue => 1500.0, :reward => 2500.0)
- `continuous_covariates::Dict{Symbol, Vector{Float64}}` — maps name → time series
    e.g. Dict(:hand_velocity => [...], :eye_x => [...])
"""
struct TrialData
    spike_counts::Vector{Int}
    condition::Dict{Symbol, Int}
    event_times::Dict{Symbol, Float64}
    continuous_covariates::Dict{Symbol, Vector{Float64}}
end


"""
    DesignSpec

Specifies what goes into the design matrix.

Fields:
- `factors::Dict{Symbol, Int}` — factor name → number of levels.
    Encoded as (n_levels − 1) indicators using effect coding.
- `events::Vector{Symbol}` — names of discrete events to model with kernels
- `continuous::Vector{Symbol}` — names of continuous time-varying covariates
- `baseline_basis::RaisedCosineBasis` — basis for h(τ)
- `encoding_basis::RaisedCosineBasis` — basis for β_f(τ)
- `history_basis::RaisedCosineBasis` — basis for spike history filter
- `event_basis::RaisedCosineBasis` — basis for event response kernels
- `dt_ms::Float64` — time bin width in ms
"""
struct DesignSpec
    factors::Dict{Symbol, Int}
    events::Vector{Symbol}
    continuous::Vector{Symbol}
    baseline_basis::RaisedCosineBasis
    encoding_basis::RaisedCosineBasis
    history_basis::RaisedCosineBasis
    event_basis::RaisedCosineBasis
    dt_ms::Float64
end


"""
    DesignMatrix

The assembled design matrix along with metadata for interpreting the columns.

Fields:
- `X::Matrix{Float64}` — (total_timebins × n_params) design matrix
- `y::Vector{Int}` — (total_timebins,) concatenated spike counts
- `groups::Dict{Symbol, UnitRange{Int}}` — maps component name → column indices
    e.g. :baseline => 1:15, :stimulus => 16:38, :history => 70:79
- `trial_boundaries::Vector{UnitRange{Int}}` — row ranges for each trial
- `n_params::Int` — total number of columns / parameters
"""
struct DesignMatrix
    X::Matrix{Float64}
    y::Vector{Int}
    groups::Dict{Symbol, UnitRange{Int}}
    trial_boundaries::Vector{UnitRange{Int}}
    n_params::Int
end


"""
    effect_code(level::Int, n_levels::Int) → Vector{Float64}

Generate effect-coded indicators for a factor with `n_levels` levels.
Returns a vector of length (n_levels − 1).

Effect coding (sum-to-zero):
  level k ∈ {1, …, n_levels−1}  → indicator k = +1, all others = 0
  level n_levels (reference)     → all indicators = −1

This ensures that the estimated coefficients represent deviations from
the grand mean, and the reference level's effect is −Σ(other effects).
"""
function effect_code(level::Int, n_levels::Int)
    code = zeros(n_levels - 1)
    if level < n_levels
        code[level] = 1.0
    else
        # Reference level: all indicators = −1
        code .= -1.0
    end
    return code
end


"""
    build_spike_history_features(spike_counts, history_basis) → Matrix{Float64}

For a single trial's spike train, compute the spike-history features at each
time bin. Returns a (T × n_history_basis) matrix.

At each time bin τ, the l-th feature is:

    h_l(τ) = Σ_δ  b_l^(hist)(δ) · y(τ − δ)

i.e. the convolution of the spike train with the l-th history basis function,
evaluated causally (only past spikes contribute).

Implementation note: this is computed as a causal filtering operation. For
each basis function, we convolve it with the spike train and take only the
causal part. Equivalently, at each time bin we form the dot product of the
basis function values with the relevant lagged spike counts.
"""
function build_spike_history_features(
    spike_counts::Vector{Int},
    hb::RaisedCosineBasis
)
    T = length(spike_counts)
    n_lags = size(hb.B, 1)   # number of lag bins in the basis
    n_basis = hb.n_basis
    H = zeros(T, n_basis)

    for τ in 2:T
        # Determine how many lags we can look back
        max_δ = min(τ - 1, n_lags)
        for l in 1:n_basis
            acc = 0.0
            @inbounds for δ in 1:max_δ
                acc += hb.B[δ, l] * spike_counts[τ - δ]
            end
            H[τ, l] = acc
        end
    end

    return H
end


"""
    build_design_matrix(trials::Vector{TrialData}, spec::DesignSpec) → DesignMatrix

Assemble the full design matrix from trial data and the design specification.

The column layout is:
  [baseline basis | factor1 encoding | factor2 encoding | ... |
   continuous1 encoding | ... | spike history | event1 kernel | event2 kernel | ...]

Each factor f with C_f levels contributes (C_f − 1) × n_enc_basis columns.
Each continuous covariate contributes n_enc_basis columns (time-varying weight)
  or 1 column (if you prefer time-invariant encoding — set separately).
Spike history contributes n_hist_basis columns.
Each event contributes n_event_basis columns.
"""
function build_design_matrix(
    trials::Vector{TrialData},
    spec::DesignSpec
)
    n_trials = length(trials)
    T_per_trial = length(trials[1].spike_counts)  # assuming equal length
    N = n_trials * T_per_trial  # total observations

    # ---- Count parameters and assign column groups ----
    groups = Dict{Symbol, UnitRange{Int}}()
    col = 1

    # Baseline: n_base_basis columns
    n_base = spec.baseline_basis.n_basis
    groups[:baseline] = col:(col + n_base - 1)
    col += n_base

    # Experimental factors: each factor f → (n_levels_f − 1) × n_enc_basis
    n_enc = spec.encoding_basis.n_basis
    for (fname, n_levels) in spec.factors
        n_cols = (n_levels - 1) * n_enc
        groups[fname] = col:(col + n_cols - 1)
        col += n_cols
    end

    # Continuous covariates: each → n_enc_basis columns (time-varying weight)
    for cname in spec.continuous
        groups[cname] = col:(col + n_enc - 1)
        col += n_enc
    end

    # Spike history: n_hist_basis columns
    n_hist = spec.history_basis.n_basis
    groups[:history] = col:(col + n_hist - 1)
    col += n_hist

    # Event kernels: each event → n_event_basis columns
    n_evt = spec.event_basis.n_basis
    for ename in spec.events
        groups[ename] = col:(col + n_evt - 1)
        col += n_evt
    end

    n_params = col - 1

    # ---- Check for name collisions ----
    all_names = Symbol[]
    push!(all_names, :baseline)
    append!(all_names, collect(keys(spec.factors)))
    append!(all_names, spec.continuous)
    push!(all_names, :history)
    append!(all_names, spec.events)
    if length(all_names) != length(unique(all_names))
        dupes = [n for n in all_names if count(==(n), all_names) > 1]
        error("Name collision in design matrix groups: $(unique(dupes)). " *
              "Factor names, event names, continuous covariate names, " *
              "'baseline', and 'history' must all be unique.")
    end

    # ---- Allocate and fill ----
    X = zeros(N, n_params)
    y = zeros(Int, N)
    trial_boundaries = Vector{UnitRange{Int}}(undef, n_trials)

    for t in 1:n_trials
        trial = trials[t]
        T_t = length(trial.spike_counts)
        row_start = (t - 1) * T_per_trial + 1
        row_end = row_start + T_t - 1
        rows = row_start:row_end
        trial_boundaries[t] = rows

        # Response
        y[rows] = trial.spike_counts

        # --- Baseline temporal profile ---
        # Each time bin τ maps to a row in the baseline basis matrix
        # baseline_basis.B is (T_per_trial × n_base)
        X[rows, groups[:baseline]] .= spec.baseline_basis.B[1:T_t, :]

        # --- Experimental factor encoding ---
        for (fname, n_levels) in spec.factors
            level = trial.condition[fname]
            indicators = effect_code(level, n_levels)  # length (n_levels − 1)

            # For each indicator c, the columns are:
            #   X[τ, col_start + (c-1)*n_enc + l] = indicator_c × b_l^enc(τ)
            cols = groups[fname]
            for c in 1:(n_levels - 1)
                for l in 1:n_enc
                    col_idx = cols[1] + (c - 1) * n_enc + (l - 1)
                    for τ in 1:T_t
                        X[row_start + τ - 1, col_idx] = indicators[c] * spec.encoding_basis.B[τ, l]
                    end
                end
            end
        end

        # --- Continuous covariates (time-varying coefficient) ---
        for cname in spec.continuous
            cov_ts = trial.continuous_covariates[cname]  # length T_t
            cols = groups[cname]
            for l in 1:n_enc
                col_idx = cols[1] + (l - 1)
                for τ in 1:T_t
                    X[row_start + τ - 1, col_idx] = cov_ts[τ] * spec.encoding_basis.B[τ, l]
                end
            end
        end

        # --- Spike history ---
        H = build_spike_history_features(trial.spike_counts, spec.history_basis)
        X[rows, groups[:history]] .= H[1:T_t, :]

        # --- Event kernels ---
        for ename in spec.events
            if !haskey(trial.event_times, ename)
                continue  # event didn't occur on this trial
            end
            t_event_ms = trial.event_times[ename]
            t_event_bin = round(Int, t_event_ms / spec.dt_ms) + 1

            cols = groups[ename]
            event_B = spec.event_basis.B  # (n_event_timepts × n_evt)
            n_event_pts = size(event_B, 1)

            for l in 1:n_evt
                col_idx = cols[1] + (l - 1)
                for δ_bin in 1:n_event_pts
                    τ = t_event_bin + δ_bin - 1
                    if 1 ≤ τ ≤ T_t
                        X[row_start + τ - 1, col_idx] += event_B[δ_bin, l]
                    end
                end
            end
        end
    end

    return DesignMatrix(X, y, groups, trial_boundaries, n_params)
end
