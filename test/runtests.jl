using Distributed
addprocs(4)

@everywhere using MendelIHT
@everywhere using SnpArrays
@everywhere using DataFrames
@everywhere using Distributions
@everywhere using Random
@everywhere using LinearAlgebra
@everywhere using GLM
@everywhere using Test

# write your own tests here
@everywhere include("MendelIHT_utilities_test.jl")
@everywhere include("L0_reg_test.jl")
@everywhere include("cv_iht_test.jl")
