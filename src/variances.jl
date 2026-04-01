

# ============================================================================
# variances.jl: Variance decomposition for GLM encoding models
# ============================================================================

"""
    variance_decomposition(dm::DesignMatrix, fit::GLMFit;
        dt_ms = 1.0,
        groups_to_test = nothing
    ) → Dict{Symbol, NamedTuple}

Compute the deviance-based variance decomposition: for each group of
parameters (experimental factor, spike history, etc.), measure the
increase in deviance when that group is removed (coefficients set to zero).

This is the GLM analogue of Type III sums of squares — each factor is
assessed conditional on all other factors being in the model.

Returns a Dict mapping group name → (ΔD, frac_deviance_explained, n_params).
"""
function variance_decomposition(
    dm::DesignMatrix,
    fit::GLMFit;
    dt_ms::Float64              = 1.0,
    groups_to_test::Union{Nothing, Vector{Symbol}} = nothing
)
    Δ = dt_ms / 1000.0
    X = dm.X
    y = dm.y

    # Full model predictions and deviance
    λ_full = exp.(X * fit.w)
    D_full = poisson_deviance(y, λ_full, Δ)
    D_null = null_deviance(y, Δ)

    groups = isnothing(groups_to_test) ? collect(keys(dm.groups)) : groups_to_test

    results = Dict{Symbol, NamedTuple{(:ΔD, :frac_explained, :n_params), Tuple{Float64, Float64, Int}}}()

    for gname in groups
        if !haskey(dm.groups, gname)
            @warn "Group $gname not found in design matrix, skipping."
            continue
        end
        cols = dm.groups[gname]

        # Zero out this group's parameters
        w_reduced = copy(fit.w)
        w_reduced[cols] .= 0.0

        λ_reduced = exp.(X * w_reduced)
        D_reduced = poisson_deviance(y, λ_reduced, Δ)

        ΔD = D_reduced - D_full
        frac = ΔD / D_null

        results[gname] = (ΔD = ΔD, frac_explained = frac, n_params = length(cols))
    end

    return results
end