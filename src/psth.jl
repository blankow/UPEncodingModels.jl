#=
╔══════════════════════════════════════════════════════════════════════════════╗
║  psth.jl                                                                   ║
║                                                                            ║
║  Compute peri-stimulus time histograms aligned to event occurrences        ║
║  (click trains or any other events). Designed to produce a sanity          ║
║  check against the GLM-extracted event kernels.                            ║
║                                                                            ║
║  Usage:                                                                    ║
║    include("psth.jl")                                                      ║
║    psth = compute_psth(trials, :left_click; dt_ms=1.0, ...)               ║
║    comparison = compare_kernel_to_psth(psth, kernel_result)                ║
╚══════════════════════════════════════════════════════════════════════════════╝
=#


"""
    compute_psth(
        trials::Vector{TrialData},
        event_name::Symbol;
        dt_ms        = 1.0,
        window_pre   = 50.0,    # ms before event
        window_post  = 150.0,   # ms after event
        smooth_ms    = 0.0,     # Gaussian smoothing σ (0 = no smoothing)
        baseline_win = (-50.0, 0.0)  # window for baseline subtraction (ms)
    ) → NamedTuple

Compute a peri-stimulus time histogram aligned to every occurrence of
the named event across all trials.

## How it works

For each occurrence of `event_name` in each trial, the function extracts
a snippet of the spike train in a window around the event time, from
`window_pre` ms before to `window_post` ms after. All snippets are
averaged to produce the mean firing rate as a function of peri-event
time, in Hz.

## Arguments

- `trials`: Vector of TrialData (with the new event_times format:
  Dict{Symbol, Vector{Float64}}).
- `event_name`: which event to align to (e.g. :left_click, :right_click).
- `dt_ms`: time bin width in ms (must match your design matrix).
- `window_pre`: how far before the event to include (positive number, in ms).
- `window_post`: how far after the event to include (in ms).
- `smooth_ms`: standard deviation of Gaussian smoothing kernel (in ms).
  Set to 0 for no smoothing (raw binned PSTH). A value of 10–20 ms
  gives a smooth rate estimate that's easier to compare visually to
  the GLM kernel.
- `baseline_win`: tuple (start, end) in ms relative to event for computing
  baseline firing rate. Default (-50, 0) uses the 50 ms before the event.

## Returns a NamedTuple with:

- `time_ms::Vector{Float64}` — peri-event time axis (ms), with 0 = event
- `rate_hz::Vector{Float64}` — mean firing rate (Hz) at each time bin
- `rate_baseline_subtracted::Vector{Float64}` — rate with pre-event baseline removed
- `baseline_hz::Float64` — the mean baseline firing rate
- `sem_hz::Vector{Float64}` — standard error of the mean rate (across snippets)
- `n_events::Int` — total number of event occurrences used
- `n_trials::Int` — number of trials that contained this event
- `event_name::Symbol` — echo back for labeling

## Comparison to the GLM kernel

The raw PSTH reflects the response to the aligned event PLUS everything
else happening at that time — other clicks, baseline fluctuations,
spike history effects, etc. The GLM kernel isolates the contribution
of a single event after controlling for everything else.

So you should NOT expect perfect agreement. What you should see:

  • Same sign: if the kernel shows excitation, the PSTH should show a
    rate increase after the event.
  • Similar latency: the PSTH onset and the kernel onset should be
    roughly aligned (within ~10–20 ms).
  • Roughly similar shape: peak timing and duration should be in the
    same ballpark.

The PSTH will typically be broader and noisier than the kernel, because
it includes contributions from temporally overlapping events that the
GLM has factored out.
"""
function compute_psth(
    trials::Vector,  # Vector{TrialData}
    event_name::Symbol;
    dt_ms::Float64        = 1.0,
    window_pre::Float64   = 50.0,
    window_post::Float64  = 150.0,
    smooth_ms::Float64    = 0.0,
    baseline_win::Tuple{Float64, Float64} = (-50.0, 0.0)
)
    # Peri-event time axis
    pre_bins  = round(Int, window_pre / dt_ms)
    post_bins = round(Int, window_post / dt_ms)
    n_bins = pre_bins + post_bins + 1
    time_ms = collect((-pre_bins):(post_bins)) .* dt_ms

    # Accumulate spike counts across all event occurrences
    spike_matrix = Vector{Vector{Float64}}()  # each entry = one snippet
    n_events = 0
    n_trials_with_event = 0

    for trial in trials
        if !haskey(trial.event_times, event_name)
            continue
        end
        evt_times = trial.event_times[event_name]
        if isempty(evt_times)
            continue
        end
        n_trials_with_event += 1

        T_t = length(trial.spike_counts)

        for t_evt_ms in evt_times
            t_evt_bin = round(Int, t_evt_ms / dt_ms) + 1

            # Extract snippet around this event
            snippet = zeros(n_bins)
            valid = true
            for i in 1:n_bins
                τ = t_evt_bin + (i - 1 - pre_bins)
                if 1 ≤ τ ≤ T_t
                    snippet[i] = Float64(trial.spike_counts[τ])
                else
                    # Event too close to trial edge — skip this snippet
                    valid = false
                    break
                end
            end

            if valid
                push!(spike_matrix, snippet)
                n_events += 1
            end
        end
    end

    if n_events == 0
        error("No valid event occurrences found for :$event_name. " *
              "Check that event_times contains this key with non-empty vectors, " *
              "and that events aren't too close to trial edges.")
    end

    # Stack into matrix: (n_events × n_bins)
    M = reduce(hcat, spike_matrix)'  # n_events × n_bins

    # Mean rate and SEM
    mean_counts = vec(sum(M, dims=1)) ./ n_events
    rate_hz = mean_counts ./ (dt_ms / 1000.0)

    # SEM: standard error of the mean count, converted to Hz
    if n_events > 1
        var_counts = vec(sum((M .- mean_counts').^2, dims=1)) ./ (n_events - 1)
        sem_counts = sqrt.(var_counts ./ n_events)
        sem_hz = sem_counts ./ (dt_ms / 1000.0)
    else
        sem_hz = zeros(n_bins)
    end

    # Optional Gaussian smoothing
    if smooth_ms > 0.0
        rate_hz = _gaussian_smooth(rate_hz, smooth_ms / dt_ms)
        sem_hz = _gaussian_smooth(sem_hz, smooth_ms / dt_ms)
    end

    # Baseline subtraction
    bl_start_bin = findfirst(t -> t >= baseline_win[1], time_ms)
    bl_end_bin   = findlast(t -> t <= baseline_win[2], time_ms)
    if isnothing(bl_start_bin) || isnothing(bl_end_bin) || bl_start_bin > bl_end_bin
        baseline_hz = 0.0
    else
        baseline_hz = sum(rate_hz[bl_start_bin:bl_end_bin]) /
                      (bl_end_bin - bl_start_bin + 1)
    end
    rate_baseline_subtracted = rate_hz .- baseline_hz

    return (
        time_ms       = time_ms,
        rate_hz       = rate_hz,
        rate_baseline_subtracted = rate_baseline_subtracted,
        baseline_hz   = baseline_hz,
        sem_hz        = sem_hz,
        n_events      = n_events,
        n_trials      = n_trials_with_event,
        event_name    = event_name,
    )
end


"""
    compute_psth_by_condition(
        trials::Vector,
        event_name::Symbol,
        condition_name::Symbol;
        kwargs...
    ) → Dict{Int, NamedTuple}

Compute separate PSTHs for each level of a condition factor.
Returns a Dict mapping condition level → PSTH result.

Useful for checking whether the click kernel differs by condition
(e.g., do left clicks evoke different responses on left-reward vs
right-reward trials?).
"""
function compute_psth_by_condition(
    trials::Vector,
    event_name::Symbol,
    condition_name::Symbol;
    kwargs...
)
    # Find all levels of this condition
    levels = unique([t.condition[condition_name] for t in trials
                     if haskey(t.condition, condition_name)])
    sort!(levels)

    results = Dict{Int, NamedTuple}()
    for level in levels
        subset = filter(t -> haskey(t.condition, condition_name) &&
                             t.condition[condition_name] == level, trials)
        if !isempty(subset)
            results[level] = compute_psth(subset, event_name; kwargs...)
        end
    end

    return results
end


"""
    compare_kernel_to_psth(psth, kernel_result; baseline_subtract=true)
        → NamedTuple

Align a GLM-extracted event kernel with a PSTH for visual comparison.

The kernel lives on a lag axis (0 to kernel_duration ms), while the
PSTH lives on a peri-event axis (-pre to +post ms). This function
puts both on a common time axis and returns them ready to plot.

## Arguments
- `psth`: result from compute_psth
- `kernel_result`: result from extract_event_kernel (has fields .lags, .kernel)
- `baseline_subtract`: if true, use the baseline-subtracted PSTH rate

## Returns a NamedTuple with:
- `time_ms`: common time axis (ms, 0 = event)
- `psth_rate`: PSTH firing rate on this axis
- `psth_sem`: SEM of the PSTH
- `kernel_lograte`: the GLM kernel (in log-rate units) on this axis
- `kernel_rate_change`: the kernel converted to a rate change (Hz) for
    more direct comparison:
        Δrate ≈ baseline_rate × (exp(kernel) − 1)
    This linearization is approximate but gives the kernel in the same
    units as the baseline-subtracted PSTH.
"""
function compare_kernel_to_psth(
    psth::NamedTuple,
    kernel_result::NamedTuple;
    baseline_subtract::Bool = true
)
    # PSTH on its time axis
    psth_rate = baseline_subtract ? psth.rate_baseline_subtracted : psth.rate_hz
    psth_sem  = psth.sem_hz
    psth_time = psth.time_ms

    # Kernel on its lag axis (starts at lag > 0)
    kernel_lags = kernel_result.lags
    kernel_vals = kernel_result.kernel

    # Build common time axis (union of both)
    t_min = minimum(psth_time)
    t_max = max(maximum(psth_time), maximum(kernel_lags))
    dt = psth_time[2] - psth_time[1]
    common_time = collect(t_min:dt:t_max)
    n_common = length(common_time)

    # Interpolate PSTH onto common axis
    psth_interp = zeros(n_common)
    sem_interp  = zeros(n_common)
    for i in 1:n_common
        idx = findfirst(t -> abs(t - common_time[i]) < dt / 2, psth_time)
        if !isnothing(idx)
            psth_interp[i] = psth_rate[idx]
            sem_interp[i]  = psth_sem[idx]
        end
    end

    # Interpolate kernel onto common axis (kernel lags map to positive times)
    kernel_interp = zeros(n_common)
    for i in 1:n_common
        if common_time[i] >= 0
            idx = findfirst(lag -> abs(lag - common_time[i]) < dt / 2, kernel_lags)
            if !isnothing(idx)
                kernel_interp[i] = kernel_vals[idx]
            end
        end
    end

    # Convert kernel from log-rate to rate change (Hz)
    # Δrate = baseline × (exp(kernel) − 1)
    baseline = psth.baseline_hz
    kernel_rate_change = baseline .* (exp.(kernel_interp) .- 1.0)

    return (
        time_ms            = common_time,
        psth_rate          = psth_interp,
        psth_sem           = sem_interp,
        kernel_lograte     = kernel_interp,
        kernel_rate_change = kernel_rate_change,
        baseline_hz        = baseline,
        n_events           = psth.n_events,
        event_name         = psth.event_name,
    )
end


"""
    _gaussian_smooth(x::Vector{Float64}, σ_bins::Float64) → Vector{Float64}

Smooth a vector with a Gaussian kernel of standard deviation σ_bins
(in units of array indices / bins). Uses a truncated kernel at ±3σ.
"""
function _gaussian_smooth(x::Vector{Float64}, σ_bins::Float64)
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
        out[i] = acc / norm  # edge-corrected normalization
    end

    return out
end


# ============================================================================
# PRINTING HELPER
# ============================================================================

"""
    print_psth_summary(psth; kernel_result=nothing)

Print a text summary of the PSTH and optional kernel comparison.
"""
function print_psth_summary(psth::NamedTuple; kernel_result=nothing)
    println("PSTH Summary: :$(psth.event_name)")
    println("  Events: $(psth.n_events) occurrences across $(psth.n_trials) trials")
    println("  Window: $(psth.time_ms[1]) to $(psth.time_ms[end]) ms")
    println("  Baseline rate: $(round(psth.baseline_hz, digits=2)) Hz")

    # Find peak in the post-event window
    post_idx = findall(t -> t > 0, psth.time_ms)
    if !isempty(post_idx)
        post_rate = psth.rate_hz[post_idx]
        peak_idx = argmax(post_rate)
        peak_time = psth.time_ms[post_idx[peak_idx]]
        peak_rate = post_rate[peak_idx]
        println("  Peak post-event rate: $(round(peak_rate, digits=2)) Hz " *
                "at $(round(peak_time, digits=1)) ms")
        println("  Peak rate change: $(round(peak_rate - psth.baseline_hz, digits=2)) Hz " *
                "($(round((peak_rate / psth.baseline_hz - 1) * 100, digits=1))%)")
    end

    if !isnothing(kernel_result)
        println()
        println("  GLM kernel comparison:")
        k = kernel_result.kernel
        k_lags = kernel_result.lags
        k_peak_idx = argmax(abs.(k))
        k_peak_lag = k_lags[k_peak_idx]
        k_peak_val = k[k_peak_idx]
        println("    Kernel peak: $(round(k_peak_val, digits=4)) log-rate " *
                "at $(round(k_peak_lag, digits=1)) ms lag")
        println("    → rate multiplier: ×$(round(exp(k_peak_val), digits=3))")
        println("    → predicted Δrate: $(round(psth.baseline_hz * (exp(k_peak_val) - 1), digits=2)) Hz")
        if !isempty(post_idx)
            println("    PSTH Δrate:        $(round(peak_rate - psth.baseline_hz, digits=2)) Hz")
        end
    end
end