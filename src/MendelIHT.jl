__precompile__()

module MendelIHT

    import Distances: euclidean, chebyshev, sqeuclidean
    import Base.show
    import GLM: glmvar, mueta, fit, linkinv, Link, GeneralizedLinearModel, devresid, checky, canonicallink
    import SpecialFunctions: digamma, trigamma
    import Pkg
    import StatsBase: sample, aweights
    import VCFTools: convert_gt, convert_ds

    using GLM
    using SnpArrays
    using CSV, DataFrames 
    using LinearAlgebra
    using Distributions
    using Random
    using DelimitedFiles
    using ProgressMeter
    using Reexport
    using ThreadPools
    using BGEN

    @reexport using Distributions
    @reexport using SnpArrays

    export iht, cross_validate
    export fit_iht, cv_iht, cv_iht_distribute_fold, iht_run_many_models
    export loglikelihood, deviance, score!, mle_for_r
    export project_k!, project_group_sparse!
    export IHTVariable, make_snparray, standardize!, maf_weights
    export save_prev!, naive_impute, initialize_beta
    export simulate_random_snparray, simulate_correlated_snparray
    export make_bim_fam_files, simulate_random_response, adhoc_add_correlation
    export random_covariance_matrix
    export pve

    # IHT will only work on single/double precision floats!
    const Float = Union{Float64,Float32}

    include("data_structures.jl")
    include("utilities.jl")
    include("simulate_utilities.jl")
    include("fit.jl")
    include("cross_validation.jl")
    include("multivariate.jl")
    include("wrapper.jl")
    include("pve.jl")

    # test data directory
    datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

    # force Julia to precompile some common functions (only Gaussian case are handled here)
    function __init__()
        dir = normpath(MendelIHT.datadir())
        cross_validate(joinpath(dir, "normal"), Normal, verbose=false, cv_summaryfile="_tmp_init_cv_file_.txt")
        cross_validate(joinpath(dir, "multivariate"), MvNormal, phenotypes=[6, 7], verbose=false, cv_summaryfile="_tmp_init_cv_file_.txt")
        rm("_tmp_init_cv_file_.txt", force=true)
    end

end # end module
