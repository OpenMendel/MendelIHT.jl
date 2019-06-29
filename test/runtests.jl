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
include("MendelIHT_utilities_test.jl")
include("L0_reg_test.jl")
include("cv_iht_test.jl")
