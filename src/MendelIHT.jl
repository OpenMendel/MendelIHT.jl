__precompile__()

module MendelIHT

	import Distances: euclidean, chebyshev, sqeuclidean
	import Base.show
	import GLM: glmvar, mueta, fit, linkinv, Link, GeneralizedLinearModel, devresid, checky, canonicallink
	import SpecialFunctions: digamma, trigamma

	using GLM
	using SnpArrays
	using DataFrames
	using Random
	using LinearAlgebra
	using Distributions
	using SparseArrays
	using Distributed

	export loglikelihood, deviance, score!, L0_reg, iht_run_many_models, update_r!
	export iht_path, simulate_random_snparray, make_bim_fam_files, project_k!
	export IHTVariables, use_A2_as_minor_allele, make_snparray, standardize!
	export std_reciprocal, project_group_sparse!, save_prev!, maf_weights
	export simulate_random_response, adhoc_add_correlation, cv_iht
	export simulate_correlated_snparray, naive_impute, cv_iht_distribute_fold, mle_for_θ

	# IHT will only work on single/double precision floats!
	const Float = Union{Float64,Float32}

	include("data_structures.jl")
	include("utilities.jl")
	include("simulate_utilities.jl")
	include("iht.jl")
	include("cross_validation.jl")
	include("negbinfit_nuisance.jl")

end # end module
