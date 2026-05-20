

# ============================================================================
# fitting.jl: Poisson GLM fitting with optional ridge regularization
# ============================================================================

"""
    GLMFit

Result of fitting the Poisson GLM.

Fields:
- `w::Vector{Float64}` — estimated parameter vector
- `log_likelihood::Float64` — log-likelihood at the optimum
- `deviance::Float64` — Poisson deviance at the optimum
- `converged::Bool` — whether the optimizer converged
- `n_iter::Int` — number of iterations
"""
struct GLMFit
    w::Vector{Float64}
    log_likelihood::Float64
    deviance::Float64
    converged::Bool
    n_iter::Int
end


"""
    poisson_loglik(w, X, y, Δ) → Float64

Compute the Poisson log-likelihood:

    ℓ(w) = Σ_i [ y_i · (Xw)_i − Δ · exp((Xw)_i) ]

where (Xw)_i = log λ_i is the log-rate, Δ is the bin width (for converting
rate to expected count), and we drop the −log(y_i!) constant.
"""
function poisson_loglik(
    w::Vector{Float64},
    X::Matrix{Float64},
    y::Vector{Int},
    Δ::Float64
)
    η = X * w                      # linear predictor (log-rate)
    ll = 0.0
    @inbounds for i in eachindex(y)
        ll += y[i] * η[i] - Δ * exp(η[i])
    end
    return ll
end


"""
    poisson_deviance(y, λ_hat, Δ) → Float64

Compute the Poisson deviance:

    D = 2 Σ_i [ y_i · log(y_i / μ̂_i) − (y_i − μ̂_i) ]

where μ̂_i = λ̂_i · Δ is the predicted expected count. Terms where y_i = 0
contribute only the −(y_i − μ̂_i) = μ̂_i part (since 0·log(0) = 0 by convention).
"""
function poisson_deviance(
    y::Vector{Int},
    λ_hat::Vector{Float64},
    Δ::Float64
)
    D = 0.0
    @inbounds for i in eachindex(y)
        μ = λ_hat[i] * Δ
        if y[i] > 0
            D += y[i] * log(y[i] / μ) - (y[i] - μ)
        else
            D += μ  # since y_i = 0
        end
    end
    return 2.0 * D
end


"""
    fit_poisson_glm(dm::DesignMatrix;
        dt_ms     = 1.0,
        λ_ridge   = Dict{Symbol,Float64}(),
        max_iter  = 100,
        tol       = 1e-8,
        method    = :IRLS,
        verbose   = false
    ) → GLMFit

Fit the Poisson GLM with optional per-group ridge regularization.

## Arguments

- `dm`: The assembled DesignMatrix.
- `dt_ms`: Time bin width in ms. Enters as Δ = dt_ms / 1000 in the Poisson
  likelihood (converting ms to seconds, assuming rates in Hz).
- `λ_ridge`: A Dict mapping group names (from dm.groups) to ridge penalty
  strengths. Groups not listed get λ = 0 (no regularization).
  Recommended starting points:
    • :baseline  → 0.01  (light; let the baseline be flexible)
    • :history   → 1.0   (moderate; prevent instability)
    • factor names → 0.1  (light; preserve sensitivity)
    • event names → 0.1

## Method

`:IRLS` — Iteratively Reweighted Least Squares (Newton-Raphson for the
canonical Poisson GLM). Each iteration solves a weighted least-squares
problem:

    w^{new} = (X'WX + Λ)^{-1} X'W z

where W = diag(μ̂) is the weight matrix (Poisson variance = mean),
z = X w^{old} + W^{-1}(y − μ̂) is the working response, and Λ is
the block-diagonal ridge penalty matrix.

This is the standard algorithm for Poisson GLMs and is guaranteed to
converge to the unique optimum when the log-likelihood is concave
(which it is for the canonical log link, and remains so with a
positive-definite ridge penalty).

`:LBFGS` — L-BFGS quasi-Newton optimization of the penalized log-likelihood.
Useful when the design matrix is very large and forming X'WX is expensive.
Requires ForwardDiff or manual gradient computation.
"""
function fit_poisson_glm(
    dm::DesignMatrix;
    dt_ms::Float64                   = 1.0,
    λ_ridge::Dict{Symbol,Float64}   = Dict{Symbol,Float64}(),
    max_iter::Int                    = 100,
    tol::Float64                     = 1e-8,
    method::Symbol                   = :IRLS,
    verbose::Bool                    = false
)
    X = dm.X
    y = dm.y
    Δ = dt_ms / 1000.0   # convert to seconds
    N, P = size(X)

    # Build the penalty matrix Λ (diagonal)
    Λ_diag = zeros(P)
    for (gname, cols) in dm.groups
        if haskey(λ_ridge, gname)
            Λ_diag[cols] .= λ_ridge[gname]
        end
    end
    Λ = Diagonal(Λ_diag)

    if method == :IRLS
        return _fit_irls(X, y, Δ, Λ, max_iter, tol, verbose)
    elseif method == :LBFGS
        return _fit_lbfgs(X, y, Δ, Λ, max_iter, tol, verbose)
    else
        error("Unknown method: $method. Use :IRLS or :LBFGS.")
    end
end


"""
    _fit_irls(X, y, Δ, Λ, max_iter, tol, verbose) → GLMFit

IRLS implementation. This is the workhorse for moderate-dimensional problems
(up to ~500 parameters, ~10^6 observations).

The per-iteration cost is dominated by forming and factorizing the
(P × P) matrix X'WX + Λ, which is O(NP² + P³). For very large P,
use L-BFGS instead.
"""
function _fit_irls(
    X::Matrix{Float64},
    y::Vector{Int},
    Δ::Float64,
    Λ::Diagonal{Float64, Vector{Float64}},
    max_iter::Int,
    tol::Float64,
    verbose::Bool
)
    N, P = size(X)
    w = zeros(P)   # initialize at zero (log-rate = 0 → rate = 1 Hz)

    converged = false
    n_iter = 0

    for iter in 1:max_iter
        n_iter = iter

        # Predicted rates and expected counts
        η = X * w
        μ = Δ .* exp.(η)       # expected counts = Δ · λ

        # Clamp μ away from zero for numerical stability
        μ = max.(μ, 1e-12)

        # Working weights: W = diag(μ)  (Poisson variance function)
        # Working response: z = η + (y − μ) ./ μ
        W = μ
        z = η .+ (Float64.(y) .- μ) ./ μ

        # Solve the weighted least squares problem:
        #   (X'WX + Λ) w_new = X'W z
        #
        # For efficiency, compute X'WX by scaling rows of X by √W
        # and using the normal equations.
        sqrtW = sqrt.(W)
        Xw = sqrtW .* X        # each row i of X scaled by √(W_i)
        zw = sqrtW .* z        # weighted response

        # Form the P×P system
        A = Xw' * Xw + Λ
        b = Xw' * zw

        # Solve via Cholesky (A is positive definite)
        w_new = A \ b

        # Check convergence
        δ = norm(w_new - w) / (norm(w) + 1e-10)
        if verbose
            ll = poisson_loglik(w_new, X, y, Δ) - 0.5 * w_new' * Λ * w_new
            # println("IRLS iter $iter: Δw = $(round(δ, sigdigits=4)), penalized LL = $(round(ll, sigdigits=6))")
        end

        w = w_new

        if δ < tol
            converged = true
            break
        end
    end

    # Compute final quantities
    η = X * w
    λ_hat = exp.(η)
    ll = poisson_loglik(w, X, y, Δ)
    dev = poisson_deviance(y, λ_hat, Δ)

    return GLMFit(w, ll, dev, converged, n_iter)
end


"""
    _fit_lbfgs(X, y, Δ, Λ, max_iter, tol, verbose) → GLMFit

L-BFGS implementation for large-scale problems where forming X'WX
is prohibitive.

This computes the gradient of the penalized log-likelihood analytically:

    ∇ℓ(w) = X'(y − μ) − Λw

where μ = Δ · exp(Xw) is the vector of expected counts. Each gradient
evaluation is O(NP) — a single matrix-vector product — with no P×P
matrices involved.

Requires the Optim.jl package.
"""
function _fit_lbfgs(
    X::Matrix{Float64},
    y::Vector{Int},
    Δ::Float64,
    Λ::Diagonal{Float64, Vector{Float64}},
    max_iter::Int,
    tol::Float64,
    verbose::Bool
)
    N, P = size(X)
    yf = Float64.(y)

    # Negative penalized log-likelihood (to minimize)
    function neg_pll(w)
        η = X * w
        val = 0.0
        @inbounds for i in 1:N
            val += -yf[i] * η[i] + Δ * exp(η[i])
        end
        val += 0.5 * dot(w, Λ, w)
        return val
    end

    # Gradient of the negative penalized log-likelihood
    function neg_pll_grad!(g, w)
        η = X * w
        μ = Δ .* exp.(η)
        residual = μ .- yf       # (N,)
        mul!(g, X', residual)    # g = X'(μ − y)
        g .+= Λ * w             # add penalty gradient
        return nothing
    end

    # Initial guess
    w0 = zeros(P)

    # This requires `using Optim`
    # Uncomment the following when Optim.jl is available:
    #=
    result = Optim.optimize(
        neg_pll,
        neg_pll_grad!,
        w0,
        Optim.LBFGS(),
        Optim.Options(
            iterations = max_iter,
            g_tol = tol,
            show_trace = verbose
        )
    )
    w = Optim.minimizer(result)
    converged = Optim.converged(result)
    n_iter = Optim.iterations(result)
    =#

    # Fallback: use the IRLS method if Optim is not loaded
    @warn "L-BFGS requires Optim.jl. Falling back to IRLS."
    return _fit_irls(X, yf, Δ, Λ, max_iter, tol, verbose)
end