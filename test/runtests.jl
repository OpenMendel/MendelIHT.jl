using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using GLM
using Test
using DelimitedFiles
using CSV

# write your own tests here
include("MendelIHT_utilities_test.jl")

# To see code coverage, cd to the MendelIHT folder and run the following
# julia -e 'using Pkg; Pkg.test("MendelIHT",coverage=true)'
# @show get_summary(process_file("src/MendelIHT_utilities_test.jl"))