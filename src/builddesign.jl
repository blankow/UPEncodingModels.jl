

# ============================================================================
# builddesign.jl: Functions to build design matrices for GLM encoding models
# ============================================================================

"""
    TrialData

Container for the data from a single trial.

Changes from original:
  • event_times is now Dict{Symbol, Vector{Float64}} — each event type
    maps to a VECTOR of occurrence times within the trial. For single
    events (go cue, reward), pass a length-1 vector: [1500.0].
    For click trains, pass all click times: [102.3, 187.5, 245.1, ...].

Fields:
- `spike_counts::Vector{Int}` — binned spike counts, length T
- `condition::Dict{Symbol, Int}` — factor name → level index
- `event_times::Dict{Symbol, Vector{Float64}}` — event name → vector
    of occurrence times (ms) within trial.
    e.g. Dict(:left_click  => [102.3, 187.5, 245.1, 312.8, 401.0],
              :right_click => [148.9, 201.4, 289.3, 355.7],
              :go_cue      => [1500.0])
- `continuous_covariates::Dict{Symbol, Vector{Float64}}` — name → time series
"""
struct TrialData
    spike_counts::Vector{Int}
    condition::Dict{Symbol, Int}
    event_times::Dict{Symbol, Vector{Float64}}
    continuous_covariates::Dict{Symbol, Vector{Float64}}
end


"""
    DesignSpec

Specifies what goes into the design matrix.

Changes from original:
  • event_basis (single) replaced by event_bases (Dict), mapping each
    event name to its own RaisedCosineBasis. This lets you use a short,
    finely-resolved kernel for clicks and a longer kernel for go cue.

Fields:
- `factors::Dict{Symbol, Int}` — factor name → number of levels
- `events::Vector{Symbol}` — names of all event types to model
- `continuous::Vector{Symbol}` — names of continuous covariates
- `baseline_basis::RaisedCosineBasis` — basis for h(τ)
- `encoding_basis::RaisedCosineBasis` — basis for β_f(τ)
- `history_basis::RaisedCosineBasis` — basis for spike history filter
- `event_bases::Dict{Symbol, RaisedCosineBasis}` — event name → basis.
    Every name in `events` must have a corresponding entry here.
- `dt_ms::Float64` — time bin width in ms
"""
struct DesignSpec
    factors::Dict{Symbol, Int}
    events::Vector{Symbol}
    continuous::Vector{Symbol}
    baseline_basis::RaisedCosineBasis
    encoding_basis::RaisedCosineBasis
    history_basis::RaisedCosineBasis
    event_bases::Dict{Symbol, RaisedCosineBasis}
    dt_ms::Float64
    trial_offsets::Vector{Symbol}
    # Event-aligned factors: each factor gets a basis evaluated on a lag axis
    # relative to a specific event.  The tuple is (event_name, condition_key, n_levels).
    # Filling is backwards: δ_bin = t_event_bin − τ + 1, so basis index 1 = at the
    # event, 2 = one bin before, etc.  Use :log spacing for denser coverage near
    # the event and sparser coverage further back.
    event_aligned_factors::Dict{Symbol, Tuple{Symbol, Symbol, Int}}
    event_aligned_factor_bases::Dict{Symbol, RaisedCosineBasis}
    # Anticausal event kernels: backward-looking kernels aligned to events that
    # may occur outside (after) the trial window.  δ_bin = t_event_bin − τ + 1,
    # so basis index 1 = at the event, 2 = one bin before, etc.
    # Every name here must also appear in event_bases.
    anticausal_events::Vector{Symbol}
end

function DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
                    history_basis, event_bases, dt_ms)
    DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
               history_basis, event_bases, dt_ms, Symbol[],
               Dict{Symbol, Tuple{Symbol, Symbol, Int}}(),
               Dict{Symbol, RaisedCosineBasis}(),
               Symbol[])
end

function DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
                    history_basis, event_bases, dt_ms, trial_offsets)
    DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
               history_basis, event_bases, dt_ms, trial_offsets,
               Dict{Symbol, Tuple{Symbol, Symbol, Int}}(),
               Dict{Symbol, RaisedCosineBasis}(),
               Symbol[])
end

function DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
                    history_basis, event_bases, dt_ms, trial_offsets,
                    event_aligned_factors, event_aligned_factor_bases)
    DesignSpec(factors, events, continuous, baseline_basis, encoding_basis,
               history_basis, event_bases, dt_ms, trial_offsets,
               event_aligned_factors, event_aligned_factor_bases,
               Symbol[])
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

# ============================================================================
# EVENT TRAIN FEATURE BUILDER
# ============================================================================

"""
    build_event_train_features(
        event_times_ms::Vector{Float64},
        basis::RaisedCosineBasis,
        T::Int,
        dt_ms::Float64
    ) → Matrix{Float64}

Compute the event-train features for one event type on one trial.
Returns a (T × n_basis) matrix.

At each time bin τ, the l-th feature is the superposition of the
basis function over all preceding event occurrences:

    f_l(τ) = Σⱼ  b_l(τ − t_event_j)  ·  𝟙(τ ≥ t_event_j)

This is a causal convolution of the event train (a sum of delta
functions at the event times) with each basis function. It's
computed the same way as the spike history features, except:

  • The "spike train" is the event train (sparse: one impulse per click)
  • The basis functions are the event kernel basis (not the history basis)
  • The lag is measured from each event time, not from each spike

The superposition is linear in log-rate space: if two clicks occur
close together, their kernels add, predicting a larger rate increase
than a single click. This is the standard assumption for brief
sensory events (linear temporal summation).

## Arguments
- `event_times_ms`: vector of event occurrence times in ms (within trial)
- `basis`: the RaisedCosineBasis for this event type
- `T`: number of time bins in the trial
- `dt_ms`: time bin width in ms
"""
function build_event_train_features(
    event_times_ms::Vector{Float64},
    basis::RaisedCosineBasis,
    T::Int,
    dt_ms::Float64
)
    n_basis = basis.n_basis
    n_kernel_pts = size(basis.B, 1)  # number of lag bins the kernel spans
    F = zeros(T, n_basis)

    # Convert event times to bin indices
    event_bins = [round(Int, t / dt_ms) + 1 for t in event_times_ms]

    # For each event occurrence, superimpose the kernel
    for t_evt in event_bins
        for l in 1:n_basis
            for δ_bin in 1:n_kernel_pts
                τ = t_evt + δ_bin - 1
                if 1 ≤ τ ≤ T
                    @inbounds F[τ, l] += basis.B[δ_bin, l]
                end
            end
        end
    end

    return F
end


"""
    build_anticausal_event_features(
        event_times_ms::Vector{Float64},
        basis::RaisedCosineBasis,
        T::Int,
        dt_ms::Float64
    ) → Matrix{Float64}

Anticausal (backward-looking) version of build_event_train_features.
At each bin τ, superimposes the kernel for all events at t_event > τ:

    f_l(τ) = Σⱼ  b_l(t_event_j − τ)  ·  𝟙(τ ≤ t_event_j)

δ_bin = t_event_bin − τ + 1, so δ_bin=1 at the event, δ_bin=2 one bin before, etc.
The event may lie outside the trial window (t_event_bin > T); the kernel still
fills bins inside the window that are within the kernel span.
"""
function build_anticausal_event_features(
    event_times_ms::Vector{Float64},
    basis::RaisedCosineBasis,
    T::Int,
    dt_ms::Float64
)
    n_basis = basis.n_basis
    n_kernel_pts = size(basis.B, 1)
    F = zeros(T, n_basis)

    event_bins = [round(Int, t / dt_ms) + 1 for t in event_times_ms]

    for t_evt in event_bins
        for l in 1:n_basis
            for δ_bin in 1:n_kernel_pts
                τ = t_evt - δ_bin + 1
                if 1 ≤ τ ≤ T
                    @inbounds F[τ, l] += basis.B[δ_bin, l]
                end
            end
        end
    end

    return F
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


# ============================================================================
# MODIFIED BUILD_DESIGN_MATRIX
# ============================================================================

"""
    build_design_matrix(trials::Vector{TrialData}, spec::DesignSpec) → DesignMatrix

Assemble the full design matrix. This version supports:
  • Multiple event occurrences per event type per trial (click trains)
  • Per-event-type basis functions

The column layout is:
  [baseline | factor1 encoding | factor2 encoding | ... |
   continuous1 | ... | spike history | event1 kernel | event2 kernel | ...]

Each event type can now have a DIFFERENT number of columns, since each
has its own basis with potentially different n_basis.
"""
function build_design_matrix(
    trials::Vector{TrialData},
    spec::DesignSpec
)
    n_trials = length(trials)
    T_per_trial = length(trials[1].spike_counts)
    N = n_trials * T_per_trial

    # ---- Validate: every event in spec.events must have a basis ----
    for ename in spec.events
        if !haskey(spec.event_bases, ename)
            error("Event :$ename is listed in spec.events but has no " *
                  "entry in spec.event_bases.")
        end
    end
    for ename in spec.anticausal_events
        if !haskey(spec.event_bases, ename)
            error("Anticausal event :$ename has no entry in spec.event_bases.")
        end
    end

    # ---- Count parameters and assign column groups ----
    groups = Dict{Symbol, UnitRange{Int}}()
    col = 1

    # Baseline
    n_base = spec.baseline_basis.n_basis
    groups[:baseline] = col:(col + n_base - 1)
    col += n_base

    # Experimental factors (trial-onset aligned)
    n_enc = spec.encoding_basis.n_basis
    for (fname, n_levels) in spec.factors
        n_cols = (n_levels - 1) * n_enc
        groups[fname] = col:(col + n_cols - 1)
        col += n_cols
    end

    # Event-aligned factors (each uses its own lag basis)
    for (fname, (_, _, n_levels)) in spec.event_aligned_factors
        n_ea = spec.event_aligned_factor_bases[fname].n_basis
        n_cols = (n_levels - 1) * n_ea
        groups[fname] = col:(col + n_cols - 1)
        col += n_cols
    end

    # Continuous covariates
    for cname in spec.continuous
        groups[cname] = col:(col + n_enc - 1)
        col += n_enc
    end

    # Per-trial scalar offsets (single column each)
    for oname in spec.trial_offsets
        groups[oname] = col:col
        col += 1
    end

    # Spike history
    n_hist = spec.history_basis.n_basis
    groups[:history] = col:(col + n_hist - 1)
    col += n_hist

    # Event kernels — each event type gets its own basis size
    for ename in spec.events
        n_evt = spec.event_bases[ename].n_basis
        groups[ename] = col:(col + n_evt - 1)
        col += n_evt
    end

    # Anticausal event kernels
    for ename in spec.anticausal_events
        n_evt = spec.event_bases[ename].n_basis
        groups[ename] = col:(col + n_evt - 1)
        col += n_evt
    end

    n_params = col - 1

    # ---- Check for name collisions ----
    all_names = Symbol[]
    push!(all_names, :baseline)
    append!(all_names, collect(keys(spec.factors)))
    append!(all_names, collect(keys(spec.event_aligned_factors)))
    append!(all_names, spec.continuous)
    append!(all_names, spec.trial_offsets)
    push!(all_names, :history)
    append!(all_names, spec.events)
    append!(all_names, spec.anticausal_events)
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
        X[rows, groups[:baseline]] .= spec.baseline_basis.B[1:T_t, :]

        # --- Experimental factor encoding ---
        for (fname, n_levels) in spec.factors
            level = trial.condition[fname]
            # level == 0 means "unknown/missing" (e.g., previous_choice on trial 1):
            # contribute nothing (zero vector) rather than erroring or biasing.
            indicators = level == 0 ? zeros(n_levels - 1) : effect_code(level, n_levels)

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

        # --- Event-aligned factor encoding ---
        for (fname, (ename, ckey, n_levels)) in spec.event_aligned_factors
            if !haskey(trial.event_times, ename) || isempty(trial.event_times[ename])
                continue
            end
            t_event_ms = trial.event_times[ename][1]
            t_event_bin = round(Int, t_event_ms / spec.dt_ms) + 1

            basis = spec.event_aligned_factor_bases[fname]
            n_ea = basis.n_basis
            n_kernel_pts = size(basis.B, 1)

            level = get(trial.condition, ckey, 0)
            indicators = level == 0 ? zeros(n_levels - 1) : effect_code(level, n_levels)

            cols = groups[fname]
            for c in 1:(n_levels - 1)
                for l in 1:n_ea
                    col_idx = cols[1] + (c - 1) * n_ea + (l - 1)
                    for τ in 1:T_t
                        # δ_bin=1 at the event, δ_bin=2 one bin before, etc.
                        δ_bin = t_event_bin - τ + 1
                        if 1 ≤ δ_bin ≤ n_kernel_pts
                            X[row_start + τ - 1, col_idx] = indicators[c] * basis.B[δ_bin, l]
                        end
                    end
                end
            end
        end

        # --- Continuous covariates ---
        for cname in spec.continuous
            cov_ts = trial.continuous_covariates[cname]
            cols = groups[cname]
            for l in 1:n_enc
                col_idx = cols[1] + (l - 1)
                for τ in 1:T_t
                    X[row_start + τ - 1, col_idx] = cov_ts[τ] * spec.encoding_basis.B[τ, l]
                end
            end
        end

        # --- Per-trial scalar offsets ---
        for oname in spec.trial_offsets
            offset_val = trial.continuous_covariates[oname][1]
            X[rows, groups[oname][1]] .= offset_val
        end

        # --- Spike history ---
        H = build_spike_history_features(trial.spike_counts, spec.history_basis)
        X[rows, groups[:history]] .= H[1:T_t, :]

        # --- Event kernels (supports trains + per-event basis) ---
        for ename in spec.events
            if !haskey(trial.event_times, ename)
                continue  # this event type didn't occur on this trial
            end

            evt_times = trial.event_times[ename]
            if isempty(evt_times)
                continue  # empty vector — no occurrences
            end

            evt_basis = spec.event_bases[ename]
            F = build_event_train_features(evt_times, evt_basis, T_t, spec.dt_ms)

            # Write into the design matrix
            cols = groups[ename]
            X[rows, cols] .= F[1:T_t, :]
        end

        # --- Anticausal event kernels (backward-looking; event may be outside window) ---
        for ename in spec.anticausal_events
            if !haskey(trial.event_times, ename)
                continue
            end
            evt_times = trial.event_times[ename]
            if isempty(evt_times)
                continue
            end
            evt_basis = spec.event_bases[ename]
            F = build_anticausal_event_features(evt_times, evt_basis, T_t, spec.dt_ms)
            cols = groups[ename]
            X[rows, cols] .= F[1:T_t, :]
        end
    end

    return DesignMatrix(X, y, groups, trial_boundaries, n_params)
end





# """
#     build_design_matrix(trials::Vector{TrialData}, spec::DesignSpec) → DesignMatrix

# Assemble the full design matrix from trial data and the design specification.

# The column layout is:
#   [baseline basis | factor1 encoding | factor2 encoding | ... |
#    continuous1 encoding | ... | spike history | event1 kernel | event2 kernel | ...]

# Each factor f with C_f levels contributes (C_f − 1) × n_enc_basis columns.
# Each continuous covariate contributes n_enc_basis columns (time-varying weight)
#   or 1 column (if you prefer time-invariant encoding — set separately).
# Spike history contributes n_hist_basis columns.
# Each event contributes n_event_basis columns.
# """
# function build_design_matrix(
#     trials::Vector{TrialData},
#     spec::DesignSpec
# )
#     n_trials = length(trials)
#     T_per_trial = length(trials[1].spike_counts)  # assuming equal length
#     N = n_trials * T_per_trial  # total observations

#     # ---- Count parameters and assign column groups ----
#     groups = Dict{Symbol, UnitRange{Int}}()
#     col = 1

#     # Baseline: n_base_basis columns
#     n_base = spec.baseline_basis.n_basis
#     groups[:baseline] = col:(col + n_base - 1)
#     col += n_base

#     # Experimental factors: each factor f → (n_levels_f − 1) × n_enc_basis
#     n_enc = spec.encoding_basis.n_basis
#     for (fname, n_levels) in spec.factors
#         n_cols = (n_levels - 1) * n_enc
#         groups[fname] = col:(col + n_cols - 1)
#         col += n_cols
#     end

#     # Continuous covariates: each → n_enc_basis columns (time-varying weight)
#     for cname in spec.continuous
#         groups[cname] = col:(col + n_enc - 1)
#         col += n_enc
#     end

#     # Spike history: n_hist_basis columns
#     n_hist = spec.history_basis.n_basis
#     groups[:history] = col:(col + n_hist - 1)
#     col += n_hist

#     # Event kernels: each event → n_event_basis columns
#     n_evt = spec.event_basis.n_basis
#     for ename in spec.events
#         groups[ename] = col:(col + n_evt - 1)
#         col += n_evt
#     end

#     n_params = col - 1

#     # ---- Check for name collisions ----
#     all_names = Symbol[]
#     push!(all_names, :baseline)
#     append!(all_names, collect(keys(spec.factors)))
#     append!(all_names, spec.continuous)
#     push!(all_names, :history)
#     append!(all_names, spec.events)
#     if length(all_names) != length(unique(all_names))
#         dupes = [n for n in all_names if count(==(n), all_names) > 1]
#         error("Name collision in design matrix groups: $(unique(dupes)). " *
#               "Factor names, event names, continuous covariate names, " *
#               "'baseline', and 'history' must all be unique.")
#     end

#     # ---- Allocate and fill ----
#     X = zeros(N, n_params)
#     y = zeros(Int, N)
#     trial_boundaries = Vector{UnitRange{Int}}(undef, n_trials)

#     for t in 1:n_trials
#         trial = trials[t]
#         T_t = length(trial.spike_counts)
#         row_start = (t - 1) * T_per_trial + 1
#         row_end = row_start + T_t - 1
#         rows = row_start:row_end
#         trial_boundaries[t] = rows

#         # Response
#         y[rows] = trial.spike_counts

#         # --- Baseline temporal profile ---
#         # Each time bin τ maps to a row in the baseline basis matrix
#         # baseline_basis.B is (T_per_trial × n_base)
#         X[rows, groups[:baseline]] .= spec.baseline_basis.B[1:T_t, :]

#         # --- Experimental factor encoding ---
#         for (fname, n_levels) in spec.factors
#             level = trial.condition[fname]
#             indicators = effect_code(level, n_levels)  # length (n_levels − 1)

#             # For each indicator c, the columns are:
#             #   X[τ, col_start + (c-1)*n_enc + l] = indicator_c × b_l^enc(τ)
#             cols = groups[fname]
#             for c in 1:(n_levels - 1)
#                 for l in 1:n_enc
#                     col_idx = cols[1] + (c - 1) * n_enc + (l - 1)
#                     for τ in 1:T_t
#                         X[row_start + τ - 1, col_idx] = indicators[c] * spec.encoding_basis.B[τ, l]
#                     end
#                 end
#             end
#         end

#         # --- Continuous covariates (time-varying coefficient) ---
#         for cname in spec.continuous
#             cov_ts = trial.continuous_covariates[cname]  # length T_t
#             cols = groups[cname]
#             for l in 1:n_enc
#                 col_idx = cols[1] + (l - 1)
#                 for τ in 1:T_t
#                     X[row_start + τ - 1, col_idx] = cov_ts[τ] * spec.encoding_basis.B[τ, l]
#                 end
#             end
#         end

#         # --- Spike history ---
#         H = build_spike_history_features(trial.spike_counts, spec.history_basis)
#         X[rows, groups[:history]] .= H[1:T_t, :]

#         # --- Event kernels ---
#         for ename in spec.events
#             if !haskey(trial.event_times, ename)
#                 continue  # event didn't occur on this trial
#             end
#             t_event_ms = trial.event_times[ename]
#             t_event_bin = round(Int, t_event_ms / spec.dt_ms) + 1

#             cols = groups[ename]
#             event_B = spec.event_basis.B  # (n_event_timepts × n_evt)
#             n_event_pts = size(event_B, 1)

#             for l in 1:n_evt
#                 col_idx = cols[1] + (l - 1)
#                 for δ_bin in 1:n_event_pts
#                     τ = t_event_bin + δ_bin - 1
#                     if 1 ≤ τ ≤ T_t
#                         X[row_start + τ - 1, col_idx] += event_B[δ_bin, l]
#                     end
#                 end
#             end
#         end
#     end

#     return DesignMatrix(X, y, groups, trial_boundaries, n_params)
# end
