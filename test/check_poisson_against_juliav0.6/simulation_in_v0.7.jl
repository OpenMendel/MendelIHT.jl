# BELOW ARE POISSON SIMULATIONS in 0.7
using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

#set random seed
Random.seed!(1111)

#read in x, y, and β from file
x_temp = readdlm("poisson_matrix", Int64)
y      = readdlm("poisson_response")
true_b = readdlm("poisson_true_model")
y = reshape(y, 1000)

# size of problem and model size
n, p = size(x_temp)
k = count(!iszero, true_b)

#construct our snp matrix
x   = make_snparray(x_temp)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z   = ones(n, 1)                   # non-genetic covariates, just the intercept

#compute poisson IHT result
v = IHTVariables(x, z, y, 1, k)
result = L0_poisson_reg(v, x, z, y, 1, k, glm = "poisson")

#check result
estimated_models = zeros(k)
correct_position = find(true_b)
estimated_models .= result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))