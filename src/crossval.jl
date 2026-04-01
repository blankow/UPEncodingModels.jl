

# ============================================================================
# crossval.jl: Cross-validation of GLM encoding models, with deviance decomposition by factor
# ============================================================================

"""
    crossval_deviance(dm::DesignMatrix, spec::DesignSpec, trials::Vector{TrialData};
        dt_ms     = 1.0,
        λ_ridge   = Dict{Symbol,Float64}(),
        cv_scheme = :trials,
        n_folds   = 5,
        verbose   = false
    ) → NamedTuple

Cross-validated deviance explained, with decomposition by factor.

## cv_scheme options:

- `:trials` — hold out entire trials. This is the standard approach for
  assessing how well the model predicts neural responses on new trials
  of the same experimental conditions. Trials are assigned to folds
  stratified by condition (as far as possible).

- `:timeblocks` — hold out contiguous blocks of time bins within each trial.
  Useful for assessing temporal generalization but beware of autocorrelation
  leaking across the fold boundary. A gap of ~50–100 ms between train and
  test blocks is recommended.

Returns:
- `cv_dev_explained::Float64` — cross-validated deviance explained (full model)
- `cv_factor_contributions::Dict{Symbol,Float64}` — ΔD per factor, cross-validated
- `fold_results::Vector` — per-fold results for diagnostics
"""
function crossval_deviance(
    dm::DesignMatrix,
    trials::Vector{TrialData};
    dt_ms::Float64                   = 1.0,
    λ_ridge::Dict{Symbol,Float64}   = Dict{Symbol,Float64}(),
    n_folds::Int                     = 5,
    verbose::Bool                    = false
)
    Δ = dt_ms / 1000.0
    n_trials = length(trials)

    # Assign trials to folds (simple round-robin; for stratified,
    # sort trials by condition first)
    fold_assignment = mod.(0:(n_trials-1), n_folds) .+ 1

    # Accumulators for deviance across folds
    D_full_total = 0.0
    D_null_total = 0.0
    D_reduced = Dict{Symbol, Float64}()
    for gname in keys(dm.groups)
        D_reduced[gname] = 0.0
    end
    n_test_total = 0

    fold_results = []

    for fold in 1:n_folds
        if verbose
            println("--- Fold $fold / $n_folds ---")
        end

        # Split trials
        test_idx  = findall(fold_assignment .== fold)
        train_idx = findall(fold_assignment .!= fold)

        # Gather test/train row indices
        test_rows  = vcat([dm.trial_boundaries[t] for t in test_idx]...)
        train_rows = vcat([dm.trial_boundaries[t] for t in train_idx]...)

        # Extract submatrices
        X_train = dm.X[train_rows, :]
        y_train = dm.y[train_rows]
        X_test  = dm.X[test_rows, :]
        y_test  = dm.y[test_rows]

        # Build a temporary DesignMatrix for the training set
        dm_train = DesignMatrix(
            X_train, y_train, dm.groups,
            [1:length(y_train)],  # single block (boundaries not needed for fitting)
            dm.n_params
        )

        # Fit on training data
        fit_train = fit_poisson_glm(dm_train; dt_ms=dt_ms, λ_ridge=λ_ridge,
                                     verbose=false)

        # Evaluate on test data
        λ_test = exp.(X_test * fit_train.w)
        D_fold_full = poisson_deviance(y_test, λ_test, Δ)
        D_fold_null = null_deviance(y_test, Δ)

        D_full_total += D_fold_full
        D_null_total += D_fold_null
        n_test_total += length(y_test)

        # For each group, evaluate the reduced model on test data
        for (gname, cols) in dm.groups
            w_red = copy(fit_train.w)
            w_red[cols] .= 0.0
            λ_red = exp.(X_test * w_red)
            D_reduced[gname] += poisson_deviance(y_test, λ_red, Δ)
        end

        push!(fold_results, (
            fold = fold,
            n_test = length(y_test),
            D_full = D_fold_full,
            D_null = D_fold_null,
            dev_explained = 1.0 - D_fold_full / D_fold_null
        ))

        if verbose
            de = 1.0 - D_fold_full / D_fold_null
            println("  Deviance explained: $(round(de * 100, digits=2))%")
        end
    end

    # Aggregate
    cv_de = 1.0 - D_full_total / D_null_total

    cv_factor = Dict{Symbol, Float64}()
    for (gname, D_red) in D_reduced
        # ΔD for this group = D_reduced − D_full, aggregated across folds
        cv_factor[gname] = (D_red - D_full_total) / D_null_total
    end

    return (
        cv_dev_explained = cv_de,
        cv_factor_contributions = cv_factor,
        fold_results = fold_results
    )
end