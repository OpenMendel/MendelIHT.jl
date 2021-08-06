using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsBase
using LinearAlgebra
using GLM
using Test
using Random
using BenchmarkTools
using DelimitedFiles
using BGEN
using VCFTools

# write your own tests here
include("utilities_test.jl")
include("L0_reg_test.jl")
include("multivariate_test.jl")
include("cv_iht_test.jl")
include("wrapper_test.jl")
