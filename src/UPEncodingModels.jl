module UPEncodingModels

using LinearAlgebra
using SparseArrays
using Random

#= using LinearAlgebra, SparseArrays, Optim, ForwardDiff                   ║
║    Optional: StatsBase (for cross-validation utilities)
=#

include("rcbfs.jl")
include("builddesign.jl")
include("fitting.jl")
include("crossval.jl")
include("utils.jl")
include("predictions.jl")
include("variances.jl")
include("deviancesummary.jl")
include("psth.jl")


export RaisedCosineBasis, make_spike_history_basis, make_trial_time_basis,
       TrialData, DesignSpec, DesignMatrix, GLMFit,
       build_design_matrix, fit_poisson_glm, predict,
       poisson_deviance, null_deviance, deviance_explained,
       variance_decomposition, crossval_deviance,
       extract_encoding_weights, extract_spike_history_filter,
       effect_code, mean, example_pipeline, ilmean, deviance_summary,
       extract_event_kernel, compute_psth, compute_psth_by_condition,
       compare_kernel_to_psth, print_psth_summary

# Write your package code here.


end
