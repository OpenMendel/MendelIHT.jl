module IHT

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

export L0_reg, iht_path, cv_iht, MendelIHT, L0_logistic_reg, L0_poisson_reg
export IHTVariables, use_A2_as_minor_allele, make_snparray, regress
export project_k!, std_reciprocal, project_group_sparse!
export update_df!, At_mul_B!, A_mul_B!, check_y_content
export save_prev!, update_mean!, normalize!, simulate_random_snparray

# IHT will only work on single/double precision floats!
const Float = Union{Float64,Float32}

include("data_structures.jl")
include("MendelIHT_utilities.jl")
include("MendelIHT.jl")
include("gwas_normal.jl")
include("gwas_logistic.jl")
include("gwas_poisson.jl")
include("cross_validation.jl")

end # end module IHT
