# This is some 'intermediate' code that peers into the glm kernels and generates comparisons with the empirical PSTHs. Just sanity checks. 

using UPEncodingModels, UPDataWrangler
import Random
using LinearAlgebra, Statistics
using Plots
using StatsPlots: violin, boxplot

#######################################################################################################
# Read in some data
floc = "/Volumes/SDXC128/UPData_all/"

sessid = "A324_2023_07_21"
regionlist = ["M1"]

binsize = 0.002 # time bin width (sec)
trials_dict, spikecounts_mat, rnames_vect, region_dict, time_vect, lclicksmat, rclicksmat = wrangle_data(floc, 
                                                                            sessid, 
                                                                            0.0, 
                                                                            0.0, 
                                                                            binsize, 
                                                                            regionlist = regionlist,
                                                                            click_times=true,
                                                                            smooth = false
)

#######################################################################################################



function main(unitnum, trials_dict, spikecounts_mat, rnames_vect, region_dict, time_vect, lclicksmat, rclicksmat, spec, binsize)

    ########################################################
    #    loop over data and store trials info for analysis
    ####################
        
    trials = TrialData[]
    total_spikes = 0
    max_history_lag = 150  # ms
    ntrials = trials_dict["ntrials"]
    tbins = length(time_vect) * (time_vect[2] - time_vect[1]) 
    
    choice_vect = Int.(trials_dict["wentright"]) .+ 1
    clicks_on_vect = Float64.(trials_dict["tclickson"] - trials_dict["tstarts"] + trials_dict["stereoclick"]) 
    clicks_off_vect = Float64.(trials_dict["tclicksoff"] - trials_dict["tstarts"])
    cpoke_out_vect = Float64.(trials_dict["startturn"] - trials_dict["tstarts"])
    time_vect = Float64.(time_vect)
    
    #NOTE: change this line:
    
        
    for trialnum in 1:ntrials
        counts = vec(spikecounts_mat[unitnum, :, trialnum])
        l_clicks = lclicksmat[trialnum]
        r_clicks = rclicksmat[trialnum]
        
        total_spikes += sum(counts)
    
        push!(trials, TrialData(
            counts,
            Dict(:choice => choice_vect[trialnum]), #, :previous_choice => previous_choice),
            Dict(:clicks_on => [clicks_on_vect[trialnum]*1000.0], 
                    :clicks_off => [clicks_off_vect[trialnum]*1000.0], 
                    :cpoke_out => [cpoke_out_vect[trialnum]*1000.0],
                    :l_clicks => l_clicks*1000.0,
                    :r_clicks => r_clicks*1000.0
                    ),
            Dict{Symbol, Vector{Float64}}()               # no continuous covariates
        ))
    end
    
    
    mean_rate = total_spikes / (ntrials * tbins)  # Hz
    
    println("  Total spikes: $total_spikes")
    println("  Mean firing rate: $(round(mean_rate, digits=2)) Hz")
    
    
    # ============================================================================
    # CONSTRUCT BASIS FUNCTIONS
    # ============================================================================

    #NOTE: these have been moved out of this function
    
    
    # ============================================================================
    # STEP 5: BUILD THE DESIGN MATRIX
    # ============================================================================
    #
    # The DesignSpec tells the builder what components to include and which
    # basis sets to use for each. The builder then assembles the full design
    # matrix by iterating over trials and filling in each component.
    #
    # For our model, the columns are:
    #   [15 baseline | 36 stimulus (3 indicators × 12 basis) |
    #    12 reward (1 indicator × 12 basis) | 8 spike history |
    #    10 stim_on kernel | 10 go_cue kernel | 10 reward_delivery kernel]
    #
    # Total: 15 + 36 + 12 + 8 + 10 + 10 + 10 = 101 parameters
    
    # spec = DesignSpec(
    #     Dict(), #Dict(:choice => 2), #,:previous_choice=>2 :previous_reward => 2),  # factors
    #     [:clicks_on],                         # event kernels
    #     [:l_clicks, :r_clicks],                                        # clicks are the only continuous covariates for now
    #     baseline_basis,
    #     encoding_basis,
    #     history_basis,
    #     event_basis,
    #     binsize
    # )
    
    dm = build_design_matrix(trials, spec)
    
    
    
    group_names = sort(collect(keys(dm.groups)), by=g -> dm.groups[g][1])
    for gname in group_names
        cols = dm.groups[gname]
        
    end
    
    # Sanity check: verify that the design matrix has reasonable values
    
    for gname in group_names
        cols = dm.groups[gname]
        X_sub = dm.X[:, cols]
        nnz_frac = count(x -> abs(x) > 1e-10, X_sub) / length(X_sub)
        col_norms = [norm(X_sub[:, j]) for j in 1:size(X_sub, 2)]
        
    end
    
    # ============================================================================
    # STEP 6: FIT THE POISSON GLM
    # ============================================================================
    #
    # We use IRLS with per-group ridge regularization. The penalty strengths
    # were chosen based on the guidelines discussed earlier:
    #
    #   • Baseline (0.01): very light — let it be flexible enough to capture
    #     the anticipatory ramp and any other slow temporal dynamics.
    #
    #   • Stimulus and reward encoding (0.1): light — we want to detect
    #     genuine encoding but avoid fitting noise in the trial-averaged
    #     response. With 37 trials per condition and 3001 time bins, the
    #     data are rich enough that light regularization suffices.
    #
    #   • Spike history (2.0): moderate — prevents the positive-feedback
    #     instability and smooths the filter, but still allows the
    #     refractory period and adaptation to be captured.
    #
    #   • Event kernels (0.1): light — the events are time-locked and
    #     averaged over all trials, so the signal-to-noise ratio is high.
    
    
    
    ridge_penalties = Dict(
        :baseline         => 0.01,
        :choice           => 0.1,
        :l_clicks         => 0.1,
        :r_clicks         => 0.1,
        :history          => 2.0,
        :clicks_on        => 0.1,
        :clicks_off       => 0.1,
        :cpoke_out        => 0.1,
    )
    
    
    println("I'm here")
    
    # fit_time = @elapsed begin
    fit = fit_poisson_glm(
        dm;
        dt_ms    = binsize*1000,
        λ_ridge  = ridge_penalties,
        max_iter = 100,
        tol      = 1e-8,
        method   = :IRLS,
        verbose  = true
    )

    println("I'm here again")
    # end

    #NOTE: debugging
    # Wherever you have the design matrix and fit result:
    λ_hat = exp.(dm.X * fit.w)
    Δ = binsize*1000.0 / 1000.0
    μ_hat = λ_hat .* Δ
    y = dm.y

    
    
    @show sum(μ_hat) / length(μ_hat)
    @show sum(y) / length(y)
    @show sum(λ_hat) / length(λ_hat)
    @show maximum(λ_hat)
    
    
    # # Overall deviance explained
    # Δ = binsize
    # λ_hat = exp.(dm.X * fit.w)
    # de = deviance_explained(dm.y, λ_hat, Δ)
    
    
    # # ============================================================================
    # # STEP 8: VARIANCE DECOMPOSITION
    # # ============================================================================
    
    
    #     # :baseline         => 0.01,
    #     # :choice           => 0.1,
    #     # :l_clicks         => 0.1,
    #     # :r_clicks         => 0.1,
    #     # :history          => 2.0,
    #     # :clicks_on        => 0.1,
    #     # :clicks_off       => 0.1,
    #     # :cpoke_out        => 0.1,
    # #
    # # For each model component, we measure the increase in deviance when that
    # # component is removed (its parameters set to zero). This is the GLM
    # # analogue of Type III sums of squares.
    # #
    # # The "fraction explained" is relative to the null model (constant rate),
    # # so it represents each component's share of the total predictable variance.
    
    # println("\n" * "-"^72)
    # println("STEP 8: Variance decomposition (in-sample)")
    # println("-"^72)
    
    # vd = variance_decomposition(dm, fit;
    #     dt_ms = binsize*1000,
    #     groups_to_test = [:choice, :clicks_on, :l_clicks, :r_clicks, :history, :baseline]
    # )
    
    # total_explained = sum(res.frac_explained for (_, res) in vd)
    
    # return vd, total_explained

    #NOTE: We'll now give the new deviance summary a try:

    println("type of y = ", typeof(y))
    println("type of λ_hat = ", typeof(λ_hat))
    println("type of Δ = ", typeof(Δ))

    devsum = deviance_summary(y, λ_hat, Δ)
    

    return  fit, dm, trials #devsum #deviance_summary(dm, fit)

end  # function main

time_vect = Float64.(time_vect)

history_basis = make_spike_history_basis(
    n_basis    = 8,
    max_lag_ms = 150.0,
    dt_ms      = binsize * 1000,
    offset     = 3.0
)

baseline_basis = make_trial_time_basis(
    n_basis      = 15,
    trial_dur_ms = (time_vect[end] - time_vect[1]) * 1000,
    dt_ms        = binsize * 1000,
    spacing      = :linear
)

encoding_basis = make_trial_time_basis(
    n_basis      = 15,
    trial_dur_ms = (time_vect[end] - time_vect[1]) * 1000,
    dt_ms        = binsize*1000,
    spacing      = :linear
)

event_basis = make_trial_time_basis(
    n_basis      = 12,
    trial_dur_ms = 500.0,
    dt_ms        = binsize*1000,
    spacing      = :log
)


click_basis = make_trial_time_basis(
    n_basis      = 9,
    trial_dur_ms = 100.0,    # kernel spans 100 ms post-click
    dt_ms        = binsize*1000,
    spacing      = :log
)

# Verify basis coverage: print the peak times of each basis function

peak_lags = [history_basis.timepoints[argmax(history_basis.B[:, l])]
             for l in 1:history_basis.n_basis]

peak_times = [encoding_basis.timepoints[argmax(encoding_basis.B[:, l])]
              for l in 1:encoding_basis.n_basis]


#fourth fit
spec = DesignSpec(
    Dict(:choice => 2), #,:previous_choice=>2 :previous_reward => 2),  # factors
    [:l_clicks, :r_clicks, :clicks_on],                         # event kernels
    Symbol[],                                        # clicks are the only continuous covariates for now
    baseline_basis,
    encoding_basis,
    history_basis,
    Dict(                                         # per-event bases
        :l_clicks  => click_basis,
        :r_clicks  => click_basis,              # same basis for both click types
        :clicks_on => event_basis,                 # different basis for clicks_on
    ),
    binsize*1000
)

    

ridx = 93
unitnum = region_dict["M1"][ridx]
_rname = regionlist[1]

fit, ourdm, ourtrials = main(unitnum, trials_dict, spikecounts_mat, rnames_vect, region_dict, time_vect, lclicksmat, rclicksmat, spec, binsize); 
lookatkernels_and_psths(fit, ourdm, ourtrials, spec)



