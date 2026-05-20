"""
    lookatkernels_and_psths(fit, ourdm, ourtrials, spec;
        binsize   = 0.002,
        click_basis = spec.event_bases[:l_clicks],
        event_basis = spec.event_bases[:clicks_on],
        label     = ""
    )

Plot a 3×2 grid comparing GLM-extracted kernels against raw PSTHs
for all major model components:

    (1) Left click kernel      (2) Right click kernel
    (3) Clicks-on kernel       (4) Choice encoding
    (5) Baseline profile       (6) [empty or summary]

Arguments:
- `fit`: GLMFit result
- `ourdm`: DesignMatrix
- `ourtrials`: Vector{TrialData} (with event times in ms)
- `spec`: DesignSpec (for accessing basis functions)
- `binsize`: time bin width in seconds (default 0.002 = 2 ms)
- `click_basis`: basis used for individual click kernels
- `event_basis`: basis used for clicks_on kernel
- `label`: string label for plot titles (e.g., "Animal1 Session3 Unit42")
"""
function lookatkernels_and_psths(
    fit, ourdm, ourtrials, spec;
    binsize    = 0.002,
    click_basis = spec.event_bases[:l_clicks],
    event_basis = spec.event_bases[:clicks_on],
    label::String = ""
)

    dt_ms = binsize * 1000
    title_suffix = isempty(label) ? "" : ", $label"

    # ---- Convert event times from seconds to ms ----
    ourtrials_ms = map(ourtrials) do trial
        new_times = copy(trial.event_times)       # already in ms — don't multiply
        new_times[:trial_start] = [0.0]
        TrialData(trial.spike_counts, trial.condition, new_times,
                trial.continuous_covariates)
    end

    # ==================================================================
    # 1. Extract all kernels from the GLM fit
    # ==================================================================

    leftclicks_kernel  = extract_event_kernel(fit, ourdm, :l_clicks, click_basis)
    rightclicks_kernel = extract_event_kernel(fit, ourdm, :r_clicks, click_basis)
    clickson_kernel    = extract_event_kernel(fit, ourdm, :clicks_on, event_basis)

    n_choice_levels = spec.factors[:choice]
    β_choice = extract_encoding_weights(fit, ourdm, :choice, n_choice_levels,
                                        spec.encoding_basis)
    choice_time = spec.encoding_basis.timepoints

    baseline_cols = ourdm.groups[:baseline]
    w_baseline = fit.w[baseline_cols]
    baseline_lograte = spec.baseline_basis.B * w_baseline
    baseline_rate_hz = exp.(baseline_lograte)
    baseline_time = spec.baseline_basis.timepoints

    # ==================================================================
    # 2. Compute PSTHs
    # ==================================================================

    # --- Individual click PSTHs ---
    psth_lclicks = compute_psth(ourtrials_ms, :l_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)
    psth_rclicks = compute_psth(ourtrials_ms, :r_clicks;
        dt_ms=dt_ms, window_pre=0.0, window_post=100.0, smooth_ms=2.0)

    comp_left  = compare_kernel_to_psth(psth_lclicks, leftclicks_kernel)
    comp_right = compare_kernel_to_psth(psth_rclicks, rightclicks_kernel)

    # --- Clicks-on (stereoclick onset) PSTH ---
    psth_clickson = compute_psth(ourtrials_ms, :clicks_on;
        dt_ms=dt_ms, window_pre=0.0, window_post=500.0, smooth_ms=10.0)
    comp_clickson = compare_kernel_to_psth(psth_clickson, clickson_kernel)

    # --- Choice: split trials by choice, align to trial start ---
    psth_by_choice = compute_psth_by_condition(
        ourtrials_ms, :trial_start, :choice;
        dt_ms=dt_ms, window_pre=0.0, window_post=1500.0, smooth_ms=5.0)

    psth_left_choice  = psth_by_choice[1]
    psth_right_choice = psth_by_choice[2]

    Δrate_choice = psth_left_choice.rate_hz .- psth_right_choice.rate_hz
    time_choice  = psth_left_choice.time_ms

    mean_rate_choice = (psth_left_choice.baseline_hz +
                        psth_right_choice.baseline_hz) / 2
    predicted_Δrate_choice = mean_rate_choice .*
        (exp.(β_choice[:, 1]) .- exp.(β_choice[:, 2]))

    # --- Baseline: average firing rate over trial time ---
    # Compute the trial-averaged firing rate as a function of time
    # within the trial, aligned to trial start.
    n_trials = length(ourtrials_ms)
    T_trial = length(ourtrials_ms[1].spike_counts)
    mean_spike_counts = zeros(T_trial)
    for trial in ourtrials_ms
        mean_spike_counts .+= Float64.(trial.spike_counts)
    end
    mean_spike_counts ./= n_trials
    observed_rate_hz = mean_spike_counts ./ (binsize)  # counts/bin ÷ seconds/bin = Hz
    trial_time_ms = collect(0:T_trial-1) .* dt_ms

    # ==================================================================
    # 3. Plot everything
    # ==================================================================

    # --- Panel 1: Left click kernel ---
    p1 = plot(title="Left click kernel$title_suffix",
              xlabel="Time from click (ms)", ylabel="Δ rate (Hz)")
    plot!(p1, comp_left.time_ms, comp_left.kernel_rate_change,
          lw=2.5, label="GLM kernel", color=:teal)
    plot!(p1, comp_left.time_ms, comp_left.psth_rate,
          lw=2.0, ribbon=comp_left.psth_sem,
          fillalpha=0.2, label="PSTH ± SEM", color=:coral)

    # --- Panel 2: Right click kernel ---
    p2 = plot(title="Right click kernel$title_suffix",
              xlabel="Time from click (ms)", ylabel="Δ rate (Hz)")
    plot!(p2, comp_right.time_ms, comp_right.kernel_rate_change,
          lw=2.5, label="GLM kernel", color=:teal)
    plot!(p2, comp_right.time_ms, comp_right.psth_rate,
          lw=2.0, ribbon=comp_right.psth_sem,
          fillalpha=0.2, label="PSTH ± SEM", color=:coral)

    # --- Panel 3: Clicks-on (stereoclick) kernel ---
    p3 = plot(title="Stereoclick kernel$title_suffix",
              xlabel="Time from clicks onset (ms)", ylabel="Δ rate (Hz)")
    plot!(p3, comp_clickson.time_ms, comp_clickson.kernel_rate_change,
          lw=2.5, label="GLM kernel", color=:teal)
    plot!(p3, comp_clickson.time_ms, comp_clickson.psth_rate,
          lw=2.0, ribbon=comp_clickson.psth_sem,
          fillalpha=0.2, label="PSTH ± SEM", color=:coral)

    # --- Panel 4: Choice encoding ---
    p4 = plot(title="Choice encoding$title_suffix",
              xlabel="Time from clicks onset (ms)",
              ylabel="Δ rate, left − right (Hz)")
    plot!(p4, choice_time, predicted_Δrate_choice,
          lw=2.5, label="GLM kernel (L − R)", color=:teal)
    plot!(p4, time_choice, Δrate_choice,
          lw=2.0, label="PSTH (L − R)", color=:coral)

    # --- Panel 5: Baseline temporal profile ---
    p5 = plot(title="Baseline profile$title_suffix",
              xlabel="Trial time (ms)", ylabel="Rate (Hz)")
    plot!(p5, baseline_time, baseline_rate_hz,
          lw=2.5, label="GLM baseline", color=:teal)
    plot!(p5, trial_time_ms, observed_rate_hz,
          lw=1.0, alpha=0.5, label="Observed mean rate", color=:gray)
    # Smooth the observed rate for a cleaner comparison
    if length(observed_rate_hz) > 50
        smooth_win = max(1, round(Int, 20.0 / dt_ms))  # ~20 ms smoothing
        smoothed = _moving_average(observed_rate_hz, smooth_win)
        plot!(p5, trial_time_ms, smoothed,
              lw=2.0, label="Observed (smoothed)", color=:coral)
    end

    # --- Panel 6: Left vs right choice PSTHs (raw, not differenced) ---
    p6 = plot(title="Choice-split PSTHs$title_suffix",
              xlabel="Time from clicks onset (ms)", ylabel="Rate (Hz)")
    plot!(p6, psth_left_choice.time_ms, psth_left_choice.rate_hz,
          lw=2.0, ribbon=psth_left_choice.sem_hz,
          fillalpha=0.15, label="Left choice", color=:steelblue)
    plot!(p6, psth_right_choice.time_ms, psth_right_choice.rate_hz,
          lw=2.0, ribbon=psth_right_choice.sem_hz,
          fillalpha=0.15, label="Right choice", color=:indianred)

    # --- Assemble grid ---
    combined = plot(p1, p2, p3, p4, p5, p6,
                    layout=(3, 2), size=(900, 1000),
                    margin=5Plots.mm, legend=:topright,
                    legendfontsize=7, titlefontsize=10)

    return combined
end


"""
    _moving_average(x::Vector{Float64}, window::Int) → Vector{Float64}

Simple centered moving average for smoothing the observed rate.
"""
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