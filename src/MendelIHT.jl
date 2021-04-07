__precompile__()

module MendelIHT

    import Distances: euclidean, chebyshev, sqeuclidean
    import Base.show
    import GLM: glmvar, mueta, fit, linkinv, Link, GeneralizedLinearModel, devresid, checky, canonicallink
    import SpecialFunctions: digamma, trigamma
    import Pkg
    import StatsBase: sample, aweights

    using GLM
    using SnpArrays
    using DataFrames
    using LinearAlgebra
    using Distributions
    using Distributed
    using Random
    using DelimitedFiles
    using ProgressMeter
    using Reexport

    export iht, cross_validate
    export fit_iht, cv_iht, iht_run_many_models
    export simulate_random_snparray, simulate_correlated_snparray
    export make_bim_fam_files, simulate_random_response, adhoc_add_correlation
    export random_covariance_matrix
    export pve
    @reexport using Distributions

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

end # end module
