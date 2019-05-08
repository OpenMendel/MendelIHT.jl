using Distributed
addprocs(4)

using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using GLM
using Test

# write your own tests here
include("MendelIHT_utilities_test.jl")
include("L0_reg_test.jl")
include("cv_iht_test.jl")
