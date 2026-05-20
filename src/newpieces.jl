#=
╔══════════════════════════════════════════════════════════════════════════════╗
║  event_trains_patch.jl                                                     ║
║                                                                            ║
║  Modified structs and functions to support:                                ║
║    • Multiple event occurrences per trial (click trains)                   ║
║    • Per-event-type basis functions (short kernels for clicks,             ║
║      longer kernels for go cue / reward delivery)                          ║
║                                                                            ║
║  This file contains drop-in replacements for TrialData, DesignSpec,        ║
║  build_design_matrix, and a new build_event_train_features function.       ║
║  Integrate these into your modified NeuralEncodingGLM module by            ║
║  replacing the corresponding structs and functions.                        ║
║                                                                            ║
║  Backward compatible: single-event trials work by passing length-1         ║
║  vectors, e.g. Dict(:go_cue => [1500.0]).                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
=#


# ============================================================================
# MODIFIED STRUCTS
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

    # ---- Count parameters and assign column groups ----
    groups = Dict{Symbol, UnitRange{Int}}()
    col = 1

    # Baseline
    n_base = spec.baseline_basis.n_basis
    groups[:baseline] = col:(col + n_base - 1)
    col += n_base

    # Experimental factors
    n_enc = spec.encoding_basis.n_basis
    for (fname, n_levels) in spec.factors
        n_cols = (n_levels - 1) * n_enc
        groups[fname] = col:(col + n_cols - 1)
        col += n_cols
    end

    # Continuous covariates
    for cname in spec.continuous
        groups[cname] = col:(col + n_enc - 1)
        col += n_enc
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
        X[rows, groups[:baseline]] .= spec.baseline_basis.B[1:T_t, :]

        # --- Experimental factor encoding ---
        for (fname, n_levels) in spec.factors
            level = trial.condition[fname]
            indicators = effect_code(level, n_levels)

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
    end

    return DesignMatrix(X, y, groups, trial_boundaries, n_params)
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

#=
# --- Define per-event bases ---

# Click kernel: short, finely resolved (8 basis functions over 100 ms)
# Clicks produce brief transients; the kernel captures the impulse
# response to a single click.
click_basis = make_trial_time_basis(
    n_basis      = 8,
    trial_dur_ms = 100.0,    # kernel spans 100 ms post-click
    dt_ms        = 1.0,
    spacing      = :linear
)

# Go cue / reward kernel: longer, coarser (10 basis functions over 400 ms)
go_basis = make_trial_time_basis(
    n_basis      = 10,
    trial_dur_ms = 400.0,
    dt_ms        = 1.0,
    spacing      = :linear
)

# --- Build the DesignSpec ---

spec = DesignSpec(
    Dict(:reward => 2),                          # factors
    [:left_click, :right_click, :go_cue],        # event types
    Symbol[],                                     # continuous covariates
    baseline_basis,
    encoding_basis,
    history_basis,
    Dict(                                         # per-event bases
        :left_click  => click_basis,
        :right_click => click_basis,              # same basis for both click types
        :go_cue      => go_basis,                 # different basis for go cue
    ),
    1.0                                           # dt_ms
)

# --- Build trial data ---

# Each trial has a train of left and right clicks at irregular times,
# plus a single go cue.
trial = TrialData(
    spike_counts,                                 # your spike count vector
    Dict(:reward => 1),                           # condition for this trial
    Dict(
        :left_click  => [102.3, 187.5, 245.1, 312.8, 401.0, 523.6, 614.2],
        :right_click => [148.9, 201.4, 289.3, 355.7, 478.1, 590.0],
        :go_cue      => [1500.0],                 # single event, length-1 vector
    ),
    Dict{Symbol, Vector{Float64}}()               # no continuous covariates
)

# --- After fitting, extract the kernels ---

left_kernel  = extract_event_kernel(fit, dm, :left_click,  click_basis)
right_kernel = extract_event_kernel(fit, dm, :right_click, click_basis)
go_kernel    = extract_event_kernel(fit, dm, :go_cue,      go_basis)

# left_kernel.lags    → [0.0, 1.0, 2.0, ..., 100.0]  (ms)
# left_kernel.kernel  → the impulse response to a single left click

# Compare left vs right click kernels to assess lateralized encoding:
# If the neuron prefers left clicks, the left kernel will have a larger
# peak than the right kernel.
=#