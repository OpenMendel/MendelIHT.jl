using Distributed
addprocs(4)

@everywhere using MendelIHT
@everywhere using SnpArrays
@everywhere using DataFrames
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
@everywhere using GLM
@everywhere using Test
@everywhere using Random

# write your own tests here
include("MendelIHT_utilities_test.jl")
include("L0_reg_test.jl")
include("multivariate_test.jl")
include("cv_iht_test.jl")
include("wrapper_test.jl")
