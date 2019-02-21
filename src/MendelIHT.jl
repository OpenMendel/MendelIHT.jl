__precompile__()

module MendelIHT

	import Distances: euclidean, chebyshev, sqeuclidean
	import StatsFuns: logistic
	import SpecialFunctions: lfactorial
	import Base.show
	import Gadfly.plot
	using SnpArrays
	using MendelBase
	using DataFrames
	using Gadfly
	using StatsBase
	using Random
	using LinearAlgebra
	using Distributions
	using SparseArrays
	using DelimitedFiles
	using GLM

	export L0_normal_reg, L0_logistic_reg, L0_poisson_reg
	export iht_path, cv_iht, MendelIHT, simulate_random_snparray
	export IHTVariables, use_A2_as_minor_allele, make_snparray, regress
	export project_k!, std_reciprocal, project_group_sparse!
	export update_df!, At_mul_B!, A_mul_B!, check_y_content, IHT
	export save_prev!, update_mean!, normalize!

	export L0_normal_reg2, L0_poisson_reg2, L0_logistic_reg2

	# IHT will only work on single/double precision floats!
	const Float = Union{Float64,Float32}

	include("IHT_wrapper.jl")
	include("data_structures.jl")
	include("utilities.jl")
	include("gwas_normal.jl")
	include("gwas_logistic.jl")
	include("gwas_poisson.jl")
	include("cross_validation.jl")

end # end module
