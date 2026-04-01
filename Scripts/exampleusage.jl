

# ============================================================================
# PART 8: EXAMPLE USAGE
# ============================================================================

"""
    example_pipeline()

Demonstrates the complete pipeline with synthetic data. This function
serves as a template you can adapt for real data.
"""
function example_pipeline()
    println("="^70)
    println("NEURAL ENCODING GLM — EXAMPLE PIPELINE")
    println("="^70)

    # ---- Step 1: Define basis functions ----
    println("\n[1] Constructing basis functions...")

    history_basis = make_spike_history_basis(
        n_basis    = 8,
        max_lag_ms = 150.0,
        dt_ms      = 1.0,
        offset     = 3.0     # fine resolution at short lags
    )

    baseline_basis = make_trial_time_basis(
        n_basis      = 12,
        trial_dur_ms = 2000.0,
        dt_ms        = 1.0,
        spacing      = :linear
    )

    encoding_basis = make_trial_time_basis(
        n_basis      = 10,
        trial_dur_ms = 2000.0,
        dt_ms        = 1.0,
        spacing      = :linear
    )

    event_basis = make_trial_time_basis(
        n_basis      = 8,
        trial_dur_ms = 400.0,   # event kernels span 400 ms
        dt_ms        = 1.0,
        spacing      = :linear
    )

    println("  History basis:  $(history_basis.n_basis) functions, 0–150 ms")
    println("  Baseline basis: $(baseline_basis.n_basis) functions, 0–2000 ms")
    println("  Encoding basis: $(encoding_basis.n_basis) functions, 0–2000 ms")
    println("  Event basis:    $(event_basis.n_basis) functions, 0–400 ms")

    # ---- Step 2: Create synthetic trial data ----
    println("\n[2] Generating synthetic trial data...")

    T_bins = 2001  # 0 to 2000 ms at 1 ms resolution
    n_trials = 200

    trials = TrialData[]
    for t in 1:n_trials
        # Random condition assignment
        stim_level  = rand(1:4)
        reward_level = rand(1:2)

        # Generate a synthetic spike train (Poisson, ~20 Hz baseline)
        # In real usage, this comes from your data
        base_rate = 0.02  # probability per 1-ms bin ≈ 20 Hz
        counts = rand.(Ref(0:1), T_bins)  # crude Bernoulli approximation
        # Add a stimulus-driven rate increase at 500–1000 ms
        for τ in 500:1000
            if rand() < 0.01 * stim_level
                counts[τ] = 1
            end
        end

        push!(trials, TrialData(
            counts,
            Dict(:stimulus => stim_level, :reward => reward_level),
            Dict(:stim_on => 500.0, :go_cue => 1200.0),
            Dict{Symbol, Vector{Float64}}()  # no continuous covariates
        ))
    end

    println("  Generated $n_trials trials, $T_bins time bins each")
    println("  Factors: stimulus (4 levels), reward (2 levels)")
    println("  Events: stim_on (500 ms), go_cue (1200 ms)")

    # ---- Step 3: Build the design matrix ----
    println("\n[3] Building design matrix...")

    spec = DesignSpec(
        Dict(:stimulus => 4, :reward => 2),   # factors
        [:stim_on, :go_cue],                   # events
        Symbol[],                               # no continuous covariates
        baseline_basis,
        encoding_basis,
        history_basis,
        event_basis,
        1.0                                     # dt_ms
    )

    dm = build_design_matrix(trials, spec)

    println("  Design matrix: $(size(dm.X, 1)) rows × $(dm.n_params) columns")
    println("  Column groups:")
    for (gname, cols) in dm.groups
        println("    :$gname → columns $cols ($(length(cols)) params)")
    end

    # ---- Step 4: Fit the model ----
    println("\n[4] Fitting Poisson GLM (IRLS with ridge regularization)...")

    fit = fit_poisson_glm(dm;
        dt_ms   = 1.0,
        λ_ridge = Dict(
            :baseline => 0.01,
            :stimulus => 0.1,
            :reward   => 0.1,
            :history  => 1.0,
            :stim_on  => 0.1,
            :go_cue   => 0.1,
        ),
        max_iter = 50,
        verbose  = true
    )

    println("  Converged: $(fit.converged) in $(fit.n_iter) iterations")
    println("  Deviance: $(round(fit.deviance, sigdigits=6))")

    # ---- Step 5: Variance decomposition ----
    println("\n[5] Variance decomposition (Type III analogue)...")

    vd = variance_decomposition(dm, fit; dt_ms=1.0,
        groups_to_test = [:stimulus, :reward, :stim_on, :go_cue, :history])

    for (gname, res) in vd
        pct = round(res.frac_explained * 100, digits=3)
        println("  :$gname → ΔD = $(round(res.ΔD, digits=2)), " *
                "frac explained = $pct%, n_params = $(res.n_params)")
    end

    # ---- Step 6: Cross-validation ----
    println("\n[6] Cross-validated deviance explained (5-fold)...")

    cv = crossval_deviance(dm, trials;
        dt_ms   = 1.0,
        λ_ridge = Dict(
            :baseline => 0.01,
            :stimulus => 0.1,
            :reward   => 0.1,
            :history  => 1.0,
            :stim_on  => 0.1,
            :go_cue   => 0.1,
        ),
        n_folds = 5,
        verbose = true
    )

    println("\n  Overall CV deviance explained: $(round(cv.cv_dev_explained * 100, digits=2))%")
    println("  Per-factor CV contributions:")
    for (gname, frac) in cv.cv_factor_contributions
        println("    :$gname → $(round(frac * 100, digits=3))%")
    end

    # ---- Step 7: Extract interpretable quantities ----
    println("\n[7] Extracting encoding weights and spike history filter...")

    β_stim = extract_encoding_weights(fit, dm, :stimulus, 4, encoding_basis)
    println("  Stimulus encoding weights: $(size(β_stim)) (timepoints × levels)")

    h_filter = extract_spike_history_filter(fit, dm, history_basis)
    println("  Spike history filter: $(length(h_filter)) lag points")
    println("  Filter min = $(round(minimum(h_filter), digits=4)), " *
            "max = $(round(maximum(h_filter), digits=4))")

    println("\n" * "="^70)
    println("Pipeline complete. In practice, you would now:")
    println("  • Plot β_stim columns vs encoding_basis.timepoints")
    println("  • Plot h_filter vs history_basis.timepoints")
    println("  • Run this for each neuron and analyze the distribution")
    println("    of encoding weights across the hierarchy")
    println("="^70)

    return (fit=fit, dm=dm, vd=vd, cv=cv, β_stim=β_stim, h_filter=h_filter)
end