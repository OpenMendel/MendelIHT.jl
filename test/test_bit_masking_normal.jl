# FIRST WE USE A FULL SnpArray WITH mask_n TO COMPUTE A MODEL

using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

#dimension of problem
n = 2000
p = 10000
k = 10 # number of true predictors

#set random seed
Random.seed!(1111)
rng = MersenneTwister(1234);
mask_n = bitrand(rng, n)

#simulat data
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
noise = rand(Normal(0, 0.1), n)
y = xbm * true_b + noise

#compute full logistic IHT result using mask_n
v = IHTVariables(x, z, y, 1, k)
mask_n = bitrand(n)
result = L0_reg(v, x, z, y, 1, k)





##### BELOW USES THE SUBSET OF SnpArrays TO COMPUTE THE MODEL

#subset of snpmatrix and response
x_mask = make_snparray(x[mask_n, :])
y_mask = y[mask_n]
z_mask = ones(sum(mask_n), 1)

#compute normal IHT result using subsetted x and y
v_mask = IHTVariables(x_mask, z_mask, y_mask, 1, k)
result_mask = L0_reg(v_mask, x_mask, z_mask, y_mask, 1, k)






###### CHECK RESULTS AGREE
estimated_models = result.beta[correct_position]
estimated_models_mask = result_mask.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    true_β           = true_model, 
    estimated_β      = estimated_models,
    estimated_β_mask = estimated_models_mask)
result.loss
result_mask.loss