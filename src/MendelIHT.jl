__precompile__()

    module MendelIHT

    import Distances: euclidean, chebyshev, sqeuclidean
    import Base.show
    import GLM: glmvar, mueta, fit, linkinv, Link, GeneralizedLinearModel, devresid, checky, canonicallink
    import SpecialFunctions: digamma, trigamma
    import StatsBase: sample

    using GLM
    using SnpArrays
    using DataFrames
    using Random
    using LinearAlgebra
    using Distributions
    using SparseArrays
    using Distributed

    export L0_reg, cv_iht, cv_iht_distribute_fold, iht_run_many_models
    export loglikelihood, deviance, score!, mle_for_r
    export project_k!, project_group_sparse!
    export IHTVariables, make_snparray, standardize!, maf_weights
    export std_reciprocal, save_prev!, naive_impute
    export simulate_random_snparray, simulate_correlated_snparray
    export make_bim_fam_files, simulate_random_response, adhoc_add_correlation

    # IHT will only work on single/double precision floats!
    const Float = Union{Float64,Float32}

    include("data_structures.jl")
    include("utilities.jl")
    include("simulate_utilities.jl")
    include("iht.jl")
    include("cross_validation.jl")
    include("negbinfit_nuisance.jl")

end # end module
