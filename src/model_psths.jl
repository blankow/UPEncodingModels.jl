#=
╔══════════════════════════════════════════════════════════════════════════════╗
║  model_psths.jl                                                            ║
║                                                                            ║
║  Generate PSTHs of the model's predicted firing rates, decomposed by       ║
║  component, and overlay them on the observed data PSTHs.                   ║
║                                                                            ║
║  The key idea: compute the model's predicted rate at every (trial, timebin)║
║  then align those predictions to events and average across trials, exactly  ║
║  like a standard PSTH — but using predicted rates instead of observed       ║
║  spikes. This gives a direct visual comparison: does the model's           ║
║  prediction look like the data?                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
=#


"""
    compute_predicted_rates(fit, dm; dt_ms=2.0) → NamedTuple

Compute the full model prediction and per-component contributions
for every time bin across all trials.

Returns:
- `full_rate::Vector{Float64}` — full model predicted rate (Hz) at every bin
- `component_logrates::Dict{Symbol, Vector{Float64}}` — each component's
    contribution to log-rate (before exponentiation) at every bin
- `baseline_lograte::Vector{Float64}` — baseline component of log-rate
- `n_obs::Int` — total number of time bins
"""
function compute_predicted_rates(fit, dm; dt_ms::Float64 = 2.0)
    X = dm.X
    w = fit.w

    # Full prediction
    full_lograte = X * w
    full_rate = exp.(full_lograte)

    # Per-component contributions in log-rate space
    # Each component contributes X[:, cols] * w[cols] to the log-rate
    component_logrates = Dict{Symbol, Vector{Float64}}()
    for (gname, cols) in dm.groups
        component_logrates[gname] = X[:, cols] * w[cols]
    end

    return (
        full_rate = full_rate,
        full_lograte = full_lograte,
        component_logrates = component_logrates,
        n_obs = length(full_rate)
    )
end


"""
    compute_predicted_psth(
        predicted_rates::Vector{Float64},
        trials_ms::Vector,  # TrialData with event_times in ms
        dm,
        event_name::Symbol;
        dt_ms        = 2.0,
        window_pre   = 0.0,
        window_post  = 100.0,
        smooth_ms    = 0.0
    ) → NamedTuple

Compute a PSTH of the model's predicted firing rates, aligned to an event.

Same logic as compute_psth but uses the model's predicted rate vector
instead of observed spike counts. The predicted rates are continuous
(not integer counts), so no count-to-rate conversion is needed.

Returns:
- `time_ms` — peri-event time axis
- `rate_hz` — mean predicted rate across all event occurrences
"""
function compute_predicted_psth(
    predicted_rates::Vector{Float64},
    trials_ms::Vector,
    dm,
    event_name::Symbol;
    dt_ms::Float64        = 2.0,
    window_pre::Float64   = 0.0,
    window_post::Float64  = 100.0,
    smooth_ms::Float64    = 0.0
)
    pre_bins  = round(Int, window_pre / dt_ms)
    post_bins = round(Int, window_post / dt_ms)
    n_bins = pre_bins + post_bins + 1
    time_ms = collect((-pre_bins):(post_bins)) .* dt_ms

    snippets = Vector{Vector{Float64}}()
    n_events = 0

    for (t_idx, trial) in enumerate(trials_ms)
        if !haskey(trial.event_times, event_name)
            continue
        end
        evt_times = trial.event_times[event_name]
        if isempty(evt_times)
            continue
        end

        # Get the row range for this trial's predicted rates
        trial_rows = dm.trial_boundaries[t_idx]
        trial_rates = predicted_rates[trial_rows]
        T_t = length(trial_rates)

        for t_evt_ms in evt_times
            t_evt_bin = round(Int, t_evt_ms / dt_ms) + 1

            snippet = zeros(n_bins)
            valid = true
            for i in 1:n_bins
                τ = t_evt_bin + (i - 1 - pre_bins)
                if 1 ≤ τ ≤ T_t
                    snippet[i] = trial_rates[τ]
                else
                    valid = false
                    break
                end
            end

            if valid
                push!(snippets, snippet)
                n_events += 1
            end
        end
    end

    if n_events == 0
        return (time_ms=time_ms, rate_hz=zeros(n_bins), n_events=0)
    end

    M = reduce(hcat, snippets)'
    mean_rate = vec(sum(M, dims=1)) ./ n_events

    if smooth_ms > 0
        mean_rate = _psth_gaussian_smooth(mean_rate, smooth_ms / dt_ms)
    end

    return (time_ms=time_ms, rate_hz=mean_rate, n_events=n_events)
end


"""
    compute_component_rate(predictions, component_name, baseline_name=:baseline)
        → Vector{Float64}

Convert a component's log-rate contribution to a rate (Hz) by adding it
to the baseline and exponentiating.

The model is: log λ = baseline + component₁ + component₂ + ...
So the rate due to baseline + one component is:
    λ_component = exp(baseline_lograte + component_lograte)

And the component's effect on rate (relative to baseline) is:
    Δλ = exp(baseline + component) − exp(baseline)
"""
function compute_component_rate(predictions, component_name::Symbol;
                                 baseline_name::Symbol = :baseline)
    baseline_lr = predictions.component_logrates[baseline_name]
    component_lr = predictions.component_logrates[component_name]
    return exp.(baseline_lr .+ component_lr)
end

function compute_component_delta_rate(predictions, component_name::Symbol;
                                       baseline_name::Symbol = :baseline)
    baseline_lr = predictions.component_logrates[baseline_name]
    component_lr = predictions.component_logrates[component_name]
    rate_with = exp.(baseline_lr .+ component_lr)
    rate_without = exp.(baseline_lr)
    return rate_with .- rate_without
end


"""
    assemble_held_out_predictions(cv_result) → (preds, dummy_dm)

Pool per-fold held-out predictions from `kfold_crossval` into a single
`preds` NamedTuple and a matching dummy design matrix, both ordered by
original trial index (trial 1 first, ..., trial N last).

Pair the returned `(preds, dummy_dm)` with the original `ourtrials_s`
(in original trial order) and call:

    model_vs_data_psths(preds, dummy_dm, ourtrials_s, spec;
        kernel_fit = cv_result.insample_fit,
        kernel_dm  = cv_result.insample_dm)
"""
function assemble_held_out_predictions(cv_result)
    fold_assignments = cv_result.fold_assignments
    n_trials = length(fold_assignments)

    trial_rates   = Vector{Vector{Float64}}(undef, n_trials)
    trial_comp_lr = Vector{Dict{Symbol, Vector{Float64}}}(undef, n_trials)

    for fold_k in cv_result.folds
        for (i, t_orig) in enumerate(fold_k.test_indices)
            rows = fold_k.test_boundaries[i]
            trial_rates[t_orig] = fold_k.test_predicted_rates[rows]
            comp_t = Dict{Symbol, Vector{Float64}}()
            for (gname, lr_vec) in fold_k.test_component_logrates
                comp_t[gname] = lr_vec[rows]
            end
            trial_comp_lr[t_orig] = comp_t
        end
    end

    # Build dummy DM: trial_boundaries matching the concatenated order
    boundaries = Vector{UnitRange{Int}}(undef, n_trials)
    offset = 0
    for t in 1:n_trials
        T_t = length(trial_rates[t])
        boundaries[t] = (offset + 1):(offset + T_t)
        offset += T_t
    end
    dummy_dm = (trial_boundaries = boundaries,)

    full_rate = vcat(trial_rates...)

    comp_names = keys(trial_comp_lr[1])
    pooled_comp_lr = Dict{Symbol, Vector{Float64}}()
    for gname in comp_names
        pooled_comp_lr[gname] = vcat([trial_comp_lr[t][gname] for t in 1:n_trials]...)
    end

    preds = (
        full_rate          = full_rate,
        full_lograte       = log.(max.(full_rate, 1e-30)),
        component_logrates = pooled_comp_lr,
        n_obs              = length(full_rate),
    )

    return preds, dummy_dm
end


"""
    model_vs_data_psths(fit, dm, ourtrials, spec; binsize, label)
    model_vs_data_psths(preds, dm, ourtrials, spec; binsize, label, kernel_fit, kernel_dm)

Two calling conventions:

1. **In-sample**: pass `fit` (GLMFit) and `dm` (DesignMatrix); predictions are
   computed from `fit.w` applied to `dm.X`.

2. **CV pooled**: pass `preds` (NamedTuple from `assemble_held_out_predictions`)
   and `dm` (the matching dummy_dm).  For the weight-kernel panel (panel 8),
   pass `kernel_fit` and `kernel_dm` pointing at the in-sample fit so the
   kernel shape is shown even though per-fold weights differ.

In both cases `ourtrials` should have event times in **seconds** (they are
multiplied by 1000 internally).
"""
function model_vs_data_psths(
    fit_or_preds, dm, ourtrials, spec;
    binsize::Float64     = 0.002,
    label::String        = "",
    kernel_fit           = nothing,
    kernel_dm            = nothing,
)
    dt_ms = binsize * 1000
    title_suffix = isempty(label) ? "" : ", $label"

    # ---- Convert event times to ms ----
    ourtrials_ms = map(ourtrials) do trial
        new_times = Dict(k => v .* 1000.0 for (k, v) in trial.event_times)
        new_times[:trial_start] = [0.0]
        TrialData(trial.spike_counts, trial.condition, new_times,
                  trial.continuous_covariates)
    end

    # ---- Resolve predictions ----
    # If fit_or_preds is a GLMFit, compute predictions from it.
    # Otherwise treat it as a pre-computed preds NamedTuple.
    if fit_or_preds isa GLMFit
        preds = compute_predicted_rates(fit_or_preds, dm; dt_ms=dt_ms)
        # For the in-sample case, kernel panels use the same fit and dm
        if isnothing(kernel_fit)
            kernel_fit = fit_or_preds
            kernel_dm  = dm
        end
    else
        preds = fit_or_preds
    end

    # Component-specific rate contributions (baseline + component)
    rate_with_lclicks = compute_component_rate(preds, :l_clicks)
    rate_with_rclicks = compute_component_rate(preds, :r_clicks)

    # Delta rates for click components (effect above baseline)
    delta_lclicks = compute_component_delta_rate(preds, :l_clicks)
    delta_rclicks = compute_component_delta_rate(preds, :r_clicks)

    # ==================================================================
    # Panel 1: Left clicks — data PSTH vs model predicted PSTH
    # ==================================================================

    data_psth_l = compute_psth(ourtrials_ms, :l_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    model_psth_l_full = compute_predicted_psth(
        preds.full_rate, ourtrials_ms, dm, :l_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    model_psth_l_component = compute_predicted_psth(
        delta_lclicks, ourtrials_ms, dm, :l_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    p1 = plot(title="Left clicks$title_suffix",
              xlabel="Time from click (ms)", ylabel="Rate (Hz)")
    plot!(p1, data_psth_l.time_ms, data_psth_l.rate_hz,
          lw=2.0, ribbon=data_psth_l.sem_hz, fillalpha=0.2,
          label="Data PSTH", color=:gray40)
    plot!(p1, model_psth_l_full.time_ms, model_psth_l_full.rate_hz,
          lw=2.5, label="Model (full)", color=:teal)
    plot!(p1, model_psth_l_component.time_ms, model_psth_l_component.rate_hz,
          lw=2.0, ls=:dash, label="L-click component (Δrate)", color=:coral)

    # ==================================================================
    # Panel 2: Right clicks
    # ==================================================================

    data_psth_r = compute_psth(ourtrials_ms, :r_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    model_psth_r_full = compute_predicted_psth(
        preds.full_rate, ourtrials_ms, dm, :r_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    model_psth_r_component = compute_predicted_psth(
        delta_rclicks, ourtrials_ms, dm, :r_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    p2 = plot(title="Right clicks$title_suffix",
              xlabel="Time from click (ms)", ylabel="Rate (Hz)")
    plot!(p2, data_psth_r.time_ms, data_psth_r.rate_hz,
          lw=2.0, ribbon=data_psth_r.sem_hz, fillalpha=0.2,
          label="Data PSTH", color=:gray40)
    plot!(p2, model_psth_r_full.time_ms, model_psth_r_full.rate_hz,
          lw=2.5, label="Model (full)", color=:teal)
    plot!(p2, model_psth_r_component.time_ms, model_psth_r_component.rate_hz,
          lw=2.0, ls=:dash, label="R-click component (Δrate)", color=:coral)

    # ==================================================================
    # Panel 3: Stereoclick onset
    # ==================================================================

    data_psth_s = compute_psth(ourtrials_ms, :clicks_on;
        dt_ms=dt_ms, window_pre=0.0, window_post=500.0, smooth_ms=5.0)

    model_psth_s = compute_predicted_psth(
        preds.full_rate, ourtrials_ms, dm, :clicks_on;
        dt_ms=dt_ms, window_pre=0.0, window_post=500.0, smooth_ms=5.0)

    delta_clickson = compute_component_delta_rate(preds, :clicks_on)
    model_psth_s_comp = compute_predicted_psth(
        delta_clickson, ourtrials_ms, dm, :clicks_on;
        dt_ms=dt_ms, window_pre=0.0, window_post=500.0, smooth_ms=5.0)

    p3 = plot(title="Stereoclick onset$title_suffix",
              xlabel="Time from onset (ms)", ylabel="Rate (Hz)")
    plot!(p3, data_psth_s.time_ms, data_psth_s.rate_hz,
          lw=2.0, ribbon=data_psth_s.sem_hz, fillalpha=0.2,
          label="Data PSTH", color=:gray40)
    plot!(p3, model_psth_s.time_ms, model_psth_s.rate_hz,
          lw=2.5, label="Model (full)", color=:teal)
    plot!(p3, model_psth_s_comp.time_ms, model_psth_s_comp.rate_hz,
          lw=2.0, ls=:dash, label="Stereoclick component (Δrate)", color=:coral)

    # ==================================================================
    # Panel 4: Choice-split PSTHs
    # ==================================================================

    data_by_choice = compute_psth_by_condition(
        ourtrials_ms, :trial_start, :choice;
        dt_ms=dt_ms, window_pre=0.0, window_post=1500.0, smooth_ms=20.0)

    # Split model predictions by choice
    left_choice_trials  = findall(t -> t.condition[:choice] == 1, ourtrials_ms)
    right_choice_trials = findall(t -> t.condition[:choice] == 2, ourtrials_ms)

    model_by_choice_l = compute_predicted_psth(
        preds.full_rate, ourtrials_ms, dm, :trial_start;
        dt_ms=dt_ms, window_pre=0.0, window_post=1500.0, smooth_ms=20.0)

    # For choice-split model PSTHs, create subsets
    trials_choice1 = ourtrials_ms[left_choice_trials]
    trials_choice2 = ourtrials_ms[right_choice_trials]

    # We need to extract the predicted rates for each subset of trials
    rates_choice1 = Float64[]
    for t_idx in left_choice_trials
        append!(rates_choice1, preds.full_rate[dm.trial_boundaries[t_idx]])
    end
    rates_choice2 = Float64[]
    for t_idx in right_choice_trials
        append!(rates_choice2, preds.full_rate[dm.trial_boundaries[t_idx]])
    end

    # Build temporary DMs just for trial boundary indexing.
    # Use the fitting window length (from dm), NOT the extended PSTH spike count length.
    T_trial = length(dm.trial_boundaries[1])
    dm_choice1 = _make_dummy_dm(length(left_choice_trials), T_trial)
    dm_choice2 = _make_dummy_dm(length(right_choice_trials), T_trial)

    model_psth_choice1 = compute_predicted_psth(
        rates_choice1, trials_choice1, dm_choice1, :trial_start;
        dt_ms=dt_ms, window_pre=0.0, window_post=1500.0, smooth_ms=20.0)
    model_psth_choice2 = compute_predicted_psth(
        rates_choice2, trials_choice2, dm_choice2, :trial_start;
        dt_ms=dt_ms, window_pre=0.0, window_post=1500.0, smooth_ms=20.0)

    p4 = plot(title="Choice-split$title_suffix",
              xlabel="Trial time (ms)", ylabel="Rate (Hz)")
    # Data
    plot!(p4, data_by_choice[1].time_ms, data_by_choice[1].rate_hz,
          lw=2.0, fillalpha=0.15, ribbon=data_by_choice[1].sem_hz,
          label="Data: left choice", color=:steelblue, alpha=0.6)
    plot!(p4, data_by_choice[2].time_ms, data_by_choice[2].rate_hz,
          lw=2.0, fillalpha=0.15, ribbon=data_by_choice[2].sem_hz,
          label="Data: right choice", color=:indianred, alpha=0.6)
    # Model
    plot!(p4, model_psth_choice1.time_ms, model_psth_choice1.rate_hz,
          lw=2.5, ls=:dash, label="Model: left choice", color=:steelblue)
    plot!(p4, model_psth_choice2.time_ms, model_psth_choice2.rate_hz,
          lw=2.5, ls=:dash, label="Model: right choice", color=:indianred)

    # ==================================================================
    # Panel 5: Baseline — trial-averaged rate over time
    # ==================================================================

    n_trials = length(ourtrials_ms)
    mean_observed = zeros(T_trial)
    for trial in ourtrials_ms
        mean_observed .+= Float64.(trial.spike_counts[1:T_trial])
    end
    mean_observed ./= n_trials
    observed_rate = mean_observed ./ binsize
    trial_time = collect(0:T_trial-1) .* dt_ms

    # Model: trial-averaged predicted rate
    mean_predicted = zeros(T_trial)
    for t_idx in 1:n_trials
        rows = dm.trial_boundaries[t_idx]
        mean_predicted .+= preds.full_rate[rows]
    end
    mean_predicted ./= n_trials

    # Baseline-only prediction
    baseline_lr = preds.component_logrates[:baseline]
    baseline_only_rate = exp.(baseline_lr)
    mean_baseline_only = zeros(T_trial)
    for t_idx in 1:n_trials
        rows = dm.trial_boundaries[t_idx]
        mean_baseline_only .+= baseline_only_rate[rows]
    end
    mean_baseline_only ./= n_trials

    # Smooth observed for comparison
    smooth_win = max(1, round(Int, 20.0 / dt_ms))

    p5 = plot(title="Baseline profile$title_suffix",
              xlabel="Trial time (ms)", ylabel="Rate (Hz)")
    plot!(p5, trial_time, _moving_average(observed_rate, smooth_win),
          lw=2.0, label="Data (smoothed)", color=:gray40)
    plot!(p5, trial_time, mean_predicted,
          lw=2.5, label="Model (full)", color=:teal)
    plot!(p5, trial_time, mean_baseline_only,
          lw=2.0, ls=:dash, label="Baseline only", color=:coral)

    # ==================================================================
    # Panel 6: Residuals — where does the model fail?
    # ==================================================================

    # Trial-averaged residual rate (observed − predicted)
    mean_residual = observed_rate .- mean_predicted
    smoothed_residual = _moving_average(mean_residual, smooth_win)

    p6 = plot(title="Residuals (data − model)$title_suffix",
              xlabel="Trial time (ms)", ylabel="Δ Rate (Hz)")
    hline!(p6, [0.0], color=:gray70, lw=0.8, label=nothing)
    plot!(p6, trial_time, smoothed_residual,
          lw=2.0, label="Mean residual", color=:purple,
          fillrange=0, fillalpha=0.1)

    # ==================================================================
    # Panel 7: Choice-split PSTH aligned to cpoke_out
    # ==================================================================

    data_cpoke_choice = compute_psth_by_condition(
        ourtrials_ms, :cpoke_out, :choice;
        dt_ms=dt_ms, window_pre=1500.0, window_post=500.0, smooth_ms=20.0)

    model_cpoke_choice1 = compute_predicted_psth(
        rates_choice1, trials_choice1, dm_choice1, :cpoke_out;
        dt_ms=dt_ms, window_pre=1500.0, window_post=500.0, smooth_ms=20.0)
    model_cpoke_choice2 = compute_predicted_psth(
        rates_choice2, trials_choice2, dm_choice2, :cpoke_out;
        dt_ms=dt_ms, window_pre=1500.0, window_post=500.0, smooth_ms=20.0)

    p7 = plot(title="Choice-split @ cpoke_out$title_suffix",
              xlabel="Time from cpoke_out (ms)", ylabel="Rate (Hz)")
    vline!(p7, [0.0], color=:gray60, lw=0.8, ls=:dot, label=nothing)
    plot!(p7, data_cpoke_choice[1].time_ms, data_cpoke_choice[1].rate_hz,
          lw=2.0, fillalpha=0.15, ribbon=data_cpoke_choice[1].sem_hz,
          label="Data: left choice", color=:steelblue, alpha=0.6)
    plot!(p7, data_cpoke_choice[2].time_ms, data_cpoke_choice[2].rate_hz,
          lw=2.0, fillalpha=0.15, ribbon=data_cpoke_choice[2].sem_hz,
          label="Data: right choice", color=:indianred, alpha=0.6)
    # Model PSTH at cpoke_out requires post-event predictions outside the fitting window;
    # only plot if compute_predicted_psth found valid in-window snippets.
    if model_cpoke_choice1.n_events > 0
        plot!(p7, model_cpoke_choice1.time_ms, model_cpoke_choice1.rate_hz,
              lw=2.5, ls=:dash, label="Model: left choice", color=:steelblue)
    end
    if model_cpoke_choice2.n_events > 0
        plot!(p7, model_cpoke_choice2.time_ms, model_cpoke_choice2.rate_hz,
              lw=2.5, ls=:dash, label="Model: right choice", color=:indianred)
    end

    # ==================================================================
    # Panel 8: choice_cpoke component delta-rate aligned to cpoke_out
    # ==================================================================

    p8 = plot(title="Choice kernels @ cpoke_out$title_suffix",
              xlabel="Time from cpoke_out (ms)", ylabel="Log-rate contribution")
    vline!(p8, [0.0], color=:gray60, lw=0.8, ls=:dot, label=nothing)
    hline!(p8, [0.0], color=:gray70, lw=0.8, label=nothing)

    has_kernel = !isnothing(kernel_fit) && !isnothing(kernel_dm) &&
                 hasproperty(kernel_dm, :groups) &&
                 (haskey(kernel_dm.groups, :l_choice_cpoke) ||
                  haskey(kernel_dm.groups, :r_choice_cpoke))
    if has_kernel
        ac_basis = spec.event_bases[:l_choice_cpoke]
        lag_axis_ms = -ac_basis.timepoints
        if haskey(kernel_dm.groups, :l_choice_cpoke)
            w_l = kernel_fit.w[kernel_dm.groups[:l_choice_cpoke]]
            plot!(p8, lag_axis_ms, ac_basis.B * w_l,
                  lw=2.5, label="L choice kernel", color=:steelblue)
        end
        if haskey(kernel_dm.groups, :r_choice_cpoke)
            w_r = kernel_fit.w[kernel_dm.groups[:r_choice_cpoke]]
            plot!(p8, lag_axis_ms, ac_basis.B * w_r,
                  lw=2.5, label="R choice kernel", color=:indianred)
        end
    else
        annotate!(p8, 0.5, 0.5, text("no choice kernel in model", :center, 9, :gray50))
    end

    # ==================================================================
    # Assemble
    # ==================================================================

    combined = plot(p1, p2, p3, p4, p5, p6, p7, p8,
                    layout=(4, 2), size=(900, 1300),
                    margin=5Plots.mm, legend=:topright,
                    legendfontsize=6, titlefontsize=10)

    return combined
end


# ============================================================================
# Population matrix extraction
# ============================================================================

"""
    extract_rate_matrices(fit, dm; binsize, max_ms=1500.0, smooth_ms=20.0)

Extract per-trial data, model predictions, and residuals as `(T × n_trials)` matrices.

Spike counts are Gaussian-smoothed within each trial then converted to Hz. Model
predictions (`exp(X*w)`) are already smooth continuous rates. Residuals are smoothed
data minus predictions, so all three matrices are on the same scale and suitable for PCA.

Returns a NamedTuple:
- `data`        — (T × n_trials) smoothed observed firing rate (Hz)
- `predictions` — (T × n_trials) model predicted firing rate (Hz)
- `residuals`   — (T × n_trials) smoothed data minus predictions (Hz)
- `T_bins`      — number of time bins stored per trial
- `n_trials`    — number of trials
"""
function extract_rate_matrices(
    fit, dm;
    binsize::Float64,
    max_ms::Float64    = 1500.0,
    smooth_ms::Float64 = 20.0
)
    dt_ms   = binsize * 1000.0
    dt_s    = binsize

    n_trials = length(dm.trial_boundaries)
    T_trial  = length(dm.trial_boundaries[1])
    T_max    = min(T_trial, round(Int, max_ms / dt_ms))

    pred_all = exp.(dm.X * fit.w)
    y_all    = Float64.(dm.y)

    σ_bins = smooth_ms / dt_ms

    data_mat  = zeros(T_max, n_trials)
    pred_mat  = zeros(T_max, n_trials)
    resid_mat = zeros(T_max, n_trials)

    for t in 1:n_trials
        rows      = dm.trial_boundaries[t]
        counts_t  = y_all[rows]
        smoothed  = smooth_ms > 0 ? _psth_gaussian_smooth(counts_t, σ_bins) : counts_t
        data_hz   = smoothed ./ dt_s
        pred_hz   = pred_all[rows]

        n_store = min(length(rows), T_max)
        data_mat[1:n_store, t]  .= data_hz[1:n_store]
        pred_mat[1:n_store, t]  .= pred_hz[1:n_store]
        resid_mat[1:n_store, t] .= (data_hz .- pred_hz)[1:n_store]
    end

    return (data=data_mat, predictions=pred_mat, residuals=resid_mat,
            T_bins=T_max, n_trials=n_trials)
end


"""
    extract_rate_matrices(predicted_rates, dm; binsize, max_ms=1500.0, smooth_ms=20.0)

Overload that accepts pre-computed predicted rates (e.g. from CV pooled predictions)
instead of a `GLMFit`.  `dm` must have `trial_boundaries` and `y`.
"""
function extract_rate_matrices(
    predicted_rates::Vector{Float64}, dm;
    binsize::Float64,
    max_ms::Float64    = 1500.0,
    smooth_ms::Float64 = 20.0
)
    dt_ms    = binsize * 1000.0
    dt_s     = binsize
    n_trials = length(dm.trial_boundaries)
    T_trial  = length(dm.trial_boundaries[1])
    T_max    = min(T_trial, round(Int, max_ms / dt_ms))
    y_all    = Float64.(dm.y)
    σ_bins   = smooth_ms / dt_ms

    data_mat  = zeros(T_max, n_trials)
    pred_mat  = zeros(T_max, n_trials)
    resid_mat = zeros(T_max, n_trials)

    for t in 1:n_trials
        rows     = dm.trial_boundaries[t]
        counts_t = y_all[rows]
        smoothed = smooth_ms > 0 ? _psth_gaussian_smooth(counts_t, σ_bins) : counts_t
        data_hz  = smoothed ./ dt_s
        pred_hz  = predicted_rates[rows]
        n_store  = min(length(rows), T_max)
        data_mat[1:n_store, t]  .= data_hz[1:n_store]
        pred_mat[1:n_store, t]  .= pred_hz[1:n_store]
        resid_mat[1:n_store, t] .= (data_hz .- pred_hz)[1:n_store]
    end

    return (data=data_mat, predictions=pred_mat, residuals=resid_mat,
            T_bins=T_max, n_trials=n_trials)
end


# ============================================================================
# Helpers
# ============================================================================

"""
    _make_dummy_dm(n_trials, T_trial)

Create a minimal struct with trial_boundaries for indexing into
a concatenated rate vector, without building a full design matrix.
"""
function _make_dummy_dm(n_trials::Int, T_trial::Int)
    boundaries = Vector{UnitRange{Int}}(undef, n_trials)
    for t in 1:n_trials
        row_start = (t - 1) * T_trial + 1
        row_end = row_start + T_trial - 1
        boundaries[t] = row_start:row_end
    end
    return (trial_boundaries = boundaries,)
end


"""
    _psth_gaussian_smooth(x, σ_bins)

Gaussian smoothing (same as in psth.jl — included here for self-containment).
"""
function _psth_gaussian_smooth(x::Vector{Float64}, σ_bins::Float64)
    if σ_bins ≤ 0
        return copy(x)
    end
    n = length(x)
    half_width = ceil(Int, 3 * σ_bins)
    kernel = [exp(-0.5 * (k / σ_bins)^2) for k in -half_width:half_width]
    kernel ./= sum(kernel)
    out = zeros(n)
    for i in 1:n
        acc = 0.0
        norm = 0.0
        for (j, kval) in enumerate(kernel)
            idx = i + (j - 1 - half_width)
            if 1 ≤ idx ≤ n
                acc += x[idx] * kval
                norm += kval
            end
        end
        out[i] = acc / norm
    end
    return out
end


function _moving_average(x::Vector{Float64}, window::Int)
    n = length(x)
    out = similar(x)
    half = window ÷ 2
    for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        out[i] = sum(@view x[lo:hi]) / (hi - lo + 1)
    end
    return out
end