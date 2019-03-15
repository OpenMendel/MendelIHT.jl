__precompile__()

module MendelIHT

	import Distances: euclidean, chebyshev, sqeuclidean
	import StatsFuns: logistic
	import SpecialFunctions: lfactorial
	import Base.show
	import GLM: glmvar, fit, linkinv, Link, GeneralizedLinearModel, devresid, checky, canonicallink

	using SnpArrays
	using MendelBase
	using DataFrames
	using Random
	using LinearAlgebra
	using Distributions
	using SparseArrays
	using DelimitedFiles
	using Distributed

	export loglikelihood, deviance, score!, L0_reg, iht_run_many_models
	export iht_path, cv_iht_distributed, MendelIHT, simulate_random_snparray
	export IHTVariables, use_A2_as_minor_allele, make_snparray
	export std_reciprocal, project_group_sparse!, save_prev!, calculate_snp_weights
	export update_mean!, At_mul_B!, A_mul_B!, IHT, update_mean!

	# IHT will only work on single/double precision floats!
	const Float = Union{Float64,Float32}

	include("IHT_wrapper.jl")
	include("data_structures.jl")
	include("utilities.jl")
	include("iht.jl")
	include("cross_validation.jl")
	include("cross_validation_distributed.jl")

end # end module
