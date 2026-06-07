# Poisson GLM Neural Encoding Model — Project Context

## Overview

This project implements a Poisson GLM for decomposing variance in neural firing rates
across experimental factors in a high-dimensional neurophysiology dataset. The code is
written in Julia and organized as a module called `UPEncodingModels`.

## The experiment

Animals perform an auditory decision-making task. On each trial:
- Trains of auditory clicks are delivered to the left and right ears (Poisson click trains)
- A stereoclick marks the onset of the stimulus period
- The animal makes a left/right choice based on which side had more clicks
- Neural activity (spike counts in 2ms bins) is recorded from thousands of neurons simultaneously

Key experimental variables:
- `l_clicks`: times of individual left clicks (vector of times per trial)
- `r_clicks`: times of individual right clicks
- `clicks_on`: stereoclick onset time (single event per trial)
- `clicks_off`: end of click stimulus
- `cpoke_out`: time the animal leaves the center port to make its choice
- `choice`: binary, left (1) or right (2)

## The model

The Poisson GLM predicts the instantaneous firing rate as:

    log λ(τ) = baseline(τ) + Σ_f β_f(τ)·x_f + Σ_l h(δ)·y(τ−δ) + Σ_e κ_e(τ−t_e)

Components:
- **Baseline**: smooth temporal profile over the trial, expanded in raised cosine basis (15 functions, linear spacing)
- **Encoding factors**: time-varying weights for discrete trial-level factors (e.g., choice), expanded in raised cosine basis (15 functions, linear spacing). Effect-coded (sum-to-zero).
- **Spike history filter**: causal filter over the neuron's own recent spikes, expanded in raised cosine basis on LOG time axis (8 functions, 150ms window). Captures refractoriness, bursting, adaptation.
- **Event kernels**: transient responses to discrete events. Different basis per event type:
  - Individual clicks (l_clicks, r_clicks): 9 basis functions over 100ms, log spacing
  - Other events (clicks_on, clicks_off, cpoke_out): 12 basis functions over 500ms, log spacing

## Key data structures

### TrialData
```julia
struct TrialData
    spike_counts::Vector{Int}           # binned spike counts, length T
    condition::Dict{Symbol, Int}        # e.g., Dict(:choice => 1)
    event_times::Dict{Symbol, Vector{Float64}}  # event name → vector of times IN MILLISECONDS
    continuous_covariates::Dict{Symbol, Vector{Float64}}
end
```

### DesignSpec
```julia
struct DesignSpec
    factors::Dict{Symbol, Int}              # factor name → number of levels
    events::Vector{Symbol}                  # event type names
    continuous::Vector{Symbol}              # continuous covariate names
    baseline_basis::RaisedCosineBasis
    encoding_basis::RaisedCosineBasis
    history_basis::RaisedCosineBasis
    event_bases::Dict{Symbol, RaisedCosineBasis}  # per-event-type basis
    dt_ms::Float64                          # bin width in ms
end
```

### DesignMatrix
```julia
struct DesignMatrix
    X::Matrix{Float64}                      # (total_timebins × n_params)
    y::Vector{Int}                          # concatenated spike counts
    groups::Dict{Symbol, UnitRange{Int}}    # component name → column indices
    trial_boundaries::Vector{UnitRange{Int}} # row ranges per trial
    n_params::Int
end
```

### GLMFit
```julia
struct GLMFit
    w::Vector{Float64}          # fitted parameters
    log_likelihood::Float64
    deviance::Float64
    converged::Bool
    n_iter::Int
end
```

## Julia scripting gotchas (HPC scripts)

- **Backslash is NOT a line continuation in Julia.** Inside macro calls (`@printf`, `@save`, etc.) a trailing `\` is parsed as the adjoint operator and produces a `MethodError: no method matching adjoint(::String)`. Always write long `@printf` calls on a single line.
- **`using JLD2: jldsave` does NOT import `@save`** — use them separately if needed.
- **BLAS threading**: always set `BLAS.set_num_threads(1)` at the top of any script using `Threads.@threads`.
- **Headless plotting**: add `ENV["GKSwstype"] = "100"` before `using Plots` in any script that saves figures without a display.
- **Top-level loop assignments** shadow globals — use `global x = val` or wrap in a function.

## Critical implementation details and known gotchas

### Units
- **Time bin width**: 2ms (binsize = 0.002 seconds, dt_ms = 2.0)
- **Event times in TrialData are in MILLISECONDS**. The raw data from the experiment is in seconds, and the conversion happens during TrialData construction (multiply by 1000). Do NOT convert again downstream.
- **The model's linear predictor is log(rate in Hz)**. The Poisson likelihood uses expected counts μ = λ·Δ where Δ = dt_ms/1000 = 0.002 seconds.

### Name collision prevention
The `groups` dictionary uses Symbol keys for all components (factors, events, baseline, history). Factor names, event names, covariate names, `:baseline`, and `:history` must all be unique. The `build_design_matrix` function checks for collisions and throws an informative error.

### BLAS threading
When using Julia's `Threads.@threads` for parallel neuron fitting, BLAS must be set to single-threaded to avoid oversubscription:
```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```
This should be set at the top of any script that uses multithreading.

### IRLS fitting
- Always use `verbose=false` when fitting in threaded loops (stdout contention causes hangs)
- Ridge regularization is per-group via the `λ_ridge` Dict. Typical values:
  - baseline: 0.1, encoding factors: 0.1, spike history: 2.0, event kernels: 0.1
- The spike history filter should be net-inhibitory (negative integral) for model stability

### Design matrix construction
- Event trains (multiple events per trial) are handled by `build_event_train_features`, which superposes each event's kernel linearly in log-rate space
- Effect coding (sum-to-zero) is used for discrete factors
- The baseline must always be included; removing it for analysis of deviance requires replacing it with a single intercept column, not dropping it entirely

## Analysis pipeline

### Per-neuron fitting
1. Construct TrialData for each trial (shared across neurons in a session)
2. Build DesignSpec with basis functions
3. Build DesignMatrix (per-neuron, since spike counts differ)
4. Fit via IRLS: `fit_poisson_glm(dm; dt_ms=2.0, λ_ridge=..., verbose=false)`
5. Compute deviance summary: `deviance_summary(y, λ_hat, Δ)`

### Analysis of deviance (Type III)
For each variable to test: rebuild the DesignSpec without that variable, rebuild the design matrix, refit the model, and compare deviance to the full model. This is a proper refit, NOT parameter zeroing. The deviance difference ΔD = D_reduced − D_full follows χ² with df = number of removed parameters.

Baseline and spike history should remain in every reduced model (they're nuisance controls).

### Shapley values (planned)
For k variable groups, fit all 2^k subset models and compute the weighted average marginal contribution of each variable across all possible entry orderings. With k=4 groups (l_clicks, r_clicks, clicks_on, choice), this requires 16 model fits per neuron.

### Variance explained reporting
The deviance_summary function returns:
- `D_total`: null deviance (total variability)
- `D_explained`: deviance accounted for by the model
- `D_residual`: remaining deviance
- `frac_explained`: D_explained / D_total (pseudo-R²)

These satisfy: D_total = D_explained + D_residual

## HPC execution

- Parallelism: `Threads.@threads` across neurons within a session
- One SLURM job per session, array jobs for multiple sessions
- Typical resource request: 100 CPUs, 500GB memory, 4 hours
- Results saved as JLD2 files; plots saved as PNGs (sequential I/O after parallel fitting)

## File organization

Key source files in the module:
- Core GLM: TrialData, DesignSpec, basis functions, design matrix construction, IRLS fitting
- Event trains patch: modified structs supporting click trains and per-event bases
- PSTH: compute_psth, compute_psth_by_condition, compare_kernel_to_psth
- Model PSTHs: compute_predicted_rates, model_vs_data_psths (predicted vs observed overlays)
- Deviance summary: deviance_summary (three-part decomposition)
- Plotting: lookatkernels_and_psths (kernel vs PSTH comparison panels)

## Current status

- Core fitting pipeline is working and tested on real data
- Analysis of deviance implemented via manual DesignSpec modification and refitting
- PSTH comparison plots (data vs kernels, data vs model predictions) implemented
- Running on HPC cluster with multithreaded parallelism
- Shapley value computation planned but not yet implemented
- Hierarchical analysis across sessions/animals not yet implemented