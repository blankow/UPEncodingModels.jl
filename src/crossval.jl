
# ============================================================================
# crossval.jl: k-fold cross-validation for the Poisson GLM
# ============================================================================

"""
    make_fold_assignments(n_trials; n_folds=5, seed=42) → Vector{Int}

Assign each of `n_trials` trials to one of `n_folds` folds (balanced,
randomly shuffled). Used to generate a single assignment that can be
reused across all model variants for paired deviance comparisons.
"""
function make_fold_assignments(n_trials::Int; n_folds::Int=5, seed::Int=42)::Vector{Int}
    rng = Random.MersenneTwister(seed)
    perm = Random.randperm(rng, n_trials)
    assignments = Vector{Int}(undef, n_trials)
    for (i, trial_idx) in enumerate(perm)
        assignments[trial_idx] = mod1(i, n_folds)
    end
    return assignments
end


"""
    kfold_crossval(trials, spec, ridge_penalties; kwargs...) → NamedTuple

Fit the Poisson GLM with k-fold cross-validation.

Builds the full design matrix once, then for each fold k slices out the
training and test rows, fits on the training set, and evaluates held-out
deviance on the test set.  The null model rate (used for cv_null_deviance)
is estimated from the training mean so evaluation is genuinely out-of-sample.

Also returns an in-sample fit on all trials so both metrics can be saved
in a single call.

## Keyword arguments
- `n_folds=5`
- `fold_assignments`: pre-computed assignments (length=n_trials, values 1:n_folds).
  If `nothing`, generated via `make_fold_assignments(n_trials; n_folds, seed)`.
  Pass the same vector to every model variant for paired comparisons.
- `seed=42`: RNG seed (only used when fold_assignments=nothing)
- `dt_ms=2.0`: time bin width in ms
- `max_iter=100`, `tol=1e-8`: IRLS settings

## Returns
NamedTuple with:
- `fold_assignments`
- `folds`: Vector of per-fold NamedTuples (see below)
- `cv_deviance`, `cv_null_deviance`, `cv_frac_explained`
- `insample_fit`, `insample_dm`: full-data fit
- `insample_devsum`: deviance_summary on all trials

Each fold NamedTuple:
- `train_indices`, `test_indices`
- `fit`: GLMFit on training data
- `test_boundaries`: per-trial row ranges in the test rate vectors
- `test_trials`: Vector{TrialData} for the test fold
- `test_predicted_rates`: exp(X_test * w), length = total test bins
- `test_component_logrates`: Dict{Symbol, Vector{Float64}} on test set
- `cv_deviance`, `cv_null_deviance`
"""
function kfold_crossval(
    trials::Vector{TrialData},
    spec::DesignSpec,
    ridge_penalties::Dict{Symbol, Float64};
    n_folds::Int = 5,
    fold_assignments::Union{Nothing, Vector{Int}} = nothing,
    seed::Int = 42,
    dt_ms::Float64 = 2.0,
    max_iter::Int = 100,
    tol::Float64 = 1e-8
)
    n_trials = length(trials)
    Δ = dt_ms / 1000.0

    if isnothing(fold_assignments)
        fold_assignments = make_fold_assignments(n_trials; n_folds=n_folds, seed=seed)
    end

    # Build the full design matrix once; reuse rows for each fold
    full_dm = build_design_matrix(trials, spec)

    # In-sample fit on all trials
    insample_fit = fit_poisson_glm(full_dm; dt_ms=dt_ms, λ_ridge=ridge_penalties,
                                    max_iter=max_iter, tol=tol)
    insample_devsum = deviance_summary(full_dm, insample_fit; dt_ms=dt_ms)

    fold_results = Vector{Any}(undef, n_folds)

    for k in 1:n_folds
        test_idx  = findall(a -> a == k, fold_assignments)
        train_idx = findall(a -> a != k, fold_assignments)

        # Row-slice the full DM into train and test subsets
        test_rows  = vcat([full_dm.trial_boundaries[t] for t in test_idx]...)
        train_rows = vcat([full_dm.trial_boundaries[t] for t in train_idx]...)

        X_test  = full_dm.X[test_rows,  :]
        y_test  = full_dm.y[test_rows]
        X_train = full_dm.X[train_rows, :]
        y_train = full_dm.y[train_rows]

        # Temporary train DM (trial_boundaries unused by fitter)
        dm_train = DesignMatrix(X_train, y_train, full_dm.groups,
                                Vector{UnitRange{Int}}(), full_dm.n_params)

        fit_k = fit_poisson_glm(dm_train; dt_ms=dt_ms, λ_ridge=ridge_penalties,
                                  max_iter=max_iter, tol=tol)

        # Held-out predictions
        test_pred_rates = exp.(X_test * fit_k.w)

        # Per-component log-rate contributions on the test set
        test_comp_lr = Dict{Symbol, Vector{Float64}}()
        for (gname, cols) in full_dm.groups
            test_comp_lr[gname] = X_test[:, cols] * fit_k.w[cols]
        end

        # Held-out deviance for this fold
        cv_dev_k = poisson_deviance(y_test, test_pred_rates, Δ)

        # Null deviance: constant rate estimated from training data, evaluated on test
        null_rate_hz = Float64(sum(y_train)) / length(y_train) / Δ
        null_pred    = fill(null_rate_hz, length(y_test))
        null_dev_k   = poisson_deviance(y_test, null_pred, Δ)

        # Per-trial boundaries within the test rate vectors (for PSTH assembly)
        test_boundaries = Vector{UnitRange{Int}}(undef, length(test_idx))
        offset = 0
        for (i, t_orig) in enumerate(test_idx)
            T_t = length(full_dm.trial_boundaries[t_orig])
            test_boundaries[i] = (offset + 1):(offset + T_t)
            offset += T_t
        end

        fold_results[k] = (
            train_indices           = train_idx,
            test_indices            = test_idx,
            fit                     = fit_k,
            test_boundaries         = test_boundaries,
            test_trials             = trials[test_idx],
            test_predicted_rates    = test_pred_rates,
            test_component_logrates = test_comp_lr,
            cv_deviance             = cv_dev_k,
            cv_null_deviance        = null_dev_k,
        )
    end

    cv_deviance      = sum(f.cv_deviance      for f in fold_results)
    cv_null_deviance = sum(f.cv_null_deviance for f in fold_results)
    cv_frac_explained = 1.0 - cv_deviance / cv_null_deviance

    return (
        fold_assignments  = fold_assignments,
        folds             = fold_results,
        cv_deviance       = cv_deviance,
        cv_null_deviance  = cv_null_deviance,
        cv_frac_explained = cv_frac_explained,
        insample_fit      = insample_fit,
        insample_dm       = full_dm,
        insample_devsum   = insample_devsum,
    )
end
