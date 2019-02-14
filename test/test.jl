using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using Random
using LinearAlgebra

function run_poisson(n :: Int64, p :: Int64)
    #set random seed
    Random.seed!(1111)

    #simulate data
    k = 10 # number of true predictors
    bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
    x = simulate_random_snparray(n, p, bernoulli_rates)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

    #construct snpmatrix, covariate files, and true model b
    z           = ones(n, 1)                   # non-genetic covariates, just the intercept
    true_b      = zeros(p)                     # model vector
    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
    shuffle!(true_b)                           # Shuffle the entries
    correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

    #check maf
    bernoulli_rates[correct_position]

    #simulate phenotypes under different noises by: y = Xb + noise
    y_temp = xbm * true_b

    # Simulate poisson data
    λ = exp.(y_temp) #inverse log link
    y = [rand(Poisson(x)) for x in λ]
    y = Float64.(y)

    #compute poisson IHT result
    result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false)

    #check result
    estimated_models = result.beta[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        correct_position = correct_position, 
        true_β           = true_model, 
        estimated_β      = estimated_models)
    
    #display results
    @show compare_model
    println("Total iteration number was " * string(result.iter))
    println("Total time was " * string(result.time))
end

Random.seed!(2019)
for i = 1:25
    @info("running the $i th model")
    n = rand(500:2000) 
    p = rand(1:10)n
    println("n, p = " * string(n) * ", " * string(p))
    run_poisson(n, p)
end



#some function that runs poisson regression on same SNP matrices, using different model sizes
function run_poisson()

	n, p = 2000, 20000

    #set random seed
    Random.seed!(1111)

    #simulate data
    bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
    x = simulate_random_snparray(n, p, bernoulli_rates)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1)                   # non-genetic covariates, just the intercept

    for k in 1:30
	    true_b      = zeros(p)                     # model vector
	    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
	    shuffle!(true_b)                           # Shuffle the entries
	    correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

	    #simulate phenotypes under different noises by: y = Xb + noise
	    y_temp = xbm * true_b

	    # Simulate poisson data
	    λ = exp.(y_temp) #inverse log link
	    y = [rand(Poisson(x)) for x in λ]
	    y = Float64.(y)

	    #compute poisson IHT result
	    result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false)

	    #check result
	    estimated_models = result.beta[correct_position]
	    true_model = true_b[correct_position]
	    compare_model = DataFrame(
	        correct_position = correct_position, 
	        true_β           = true_model, 
	        estimated_β      = estimated_models)
    
	    #display results
	    @show compare_model
	    println("Total iteration number was " * string(result.iter))
	    println("Total time was " * string(result.time))
	end
end









function run_logistic(n :: Int64, p :: Int64, debias :: Bool)
    #set random seed
    Random.seed!(1111)

    #simulate data
    k = 10 # number of true predictors
    bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
    x = simulate_random_snparray(n, p, bernoulli_rates)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

    #construct snpmatrix, covariate files, and true model b
    z           = ones(n, 1)                   # non-genetic covariates, just the intercept
    true_b      = zeros(p)                     # model vector
    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
    shuffle!(true_b)                           # Shuffle the entries
    correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

    #simulate phenotypes under different noises by: y = Xb + noise
    y_temp = xbm * true_b

    # Apply inverse logit link and sample from the vector of distributions
    prob = logistic.(y_temp) #inverse logit link
    y = [rand(Bernoulli(x)) for x in prob]
    y = Float64.(y)

    #compute logistic IHT result
    result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=debias, show_info=false)

    #check result
    estimated_models = result.beta[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        correct_position = correct_position, 
        true_β           = true_model, 
        estimated_β      = estimated_models)

    #display results
    @show compare_model
    println("n = " * string(n) * ", and p = " * string(p))
    println("Total iteration number was " * string(result.iter))
    println("Total time was " * string(result.time))
end

for i = 1:100
    println("running the $i th model")
    n = rand(500:2000) 
    p = rand(1:10)n
    debias = true
    run_logistic(n, p, debias)
end










using Revise
using Random
using LinearAlgebra
using GLMNet
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using Random
using LinearAlgebra

function iht_lasso_poisson(n :: Int64, p :: Int64)
    #set random seed
    Random.seed!(1111)

    #define maf and true model size
    bernoulli_rates = 0.5rand(p)
    k = 10

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(n, p, bernoulli_rates)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z           = ones(n, 1)                   # non-genetic covariates, just the intercept
    true_b      = zeros(p)                     # model vector
    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
    shuffle!(true_b)                           # Shuffle the entries
    correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

    # Simulate poisson data
    y_temp = xbm * true_b
    λ = exp.(y_temp) #inverse log link
    y = [rand(Poisson(x)) for x in λ]
    y = Float64.(y)

    #compute poisson IHT result
    path = collect(1:15)
    num_folds = 5
    folds = rand(1:num_folds, size(x, 1))
    k_est_iht = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf=false, glm="poisson", debias=false)
    iht_result = L0_poisson_reg(x, z, y, 1, k_est, glm = "poisson", debias=false, convg=false, show_info=false, true_beta=true_b)

    #compute poisson lasso result
    x_float = [convert(Matrix{Float64}, x, center=true, scale=true) z]
    cv = glmnetcv(x_float, y, Poisson(), dfmax=15, nfolds=5, folds=folds)
    best = argmin(cv.meanloss)
    lasso_result = cv.path.betas[:, best]
    k_est_lasso = length(findall(!iszero, lasso_result))

    #check result
    IHT_model = iht_result.beta[correct_position]
    lasso_model = lasso_result[correct_position]
    true_model = true_b[correct_position]
    compare_model = DataFrame(
        correct_position = correct_position, 
        true_β           = true_model, 
        iht_β            = IHT_model,
        lasso_β          = lasso_model)
    @show compare_model

    #compute summary statistics
    lasso_num_correct_predictors = findall(!iszero, lasso_model)
    lasso_false_positives = k_est_lasso - lasso_num_correct_predictors
    lasso_false_negatives = k - lasso_num_correct_predictors
    iht_num_correct_predictors = findall(!iszero, IHT_model)
    iht_false_positives = k_est_iht - iht_num_correct_predictors
    iht_false_negatives = iht_num_correct_predictors
    println("IHT: cv found " * "$k_est_iht predictors, with " * "$iht_false_positives false positives " * "and $iht_false_negatives false negatives")
    println("lasso: cv found " * "$k_est_lasso predictors, with " * "$lasso_false_positives false positives and " * "$lasso_false_negatives false negatives \n\n")

    return lasso_false_positives, lasso_false_negatives, iht_false_positives, iht_false_negatives
end


Random.seed!(2019)
lasso_false_positives = 0
lasso_false_negatives = 0 
iht_false_positives = 0
iht_false_negatives = 0
for i = 1:25
    @info("running the $i th model")
    n = rand(100:1000) 
    p = rand(1:10)n
    println("n, p = " * string(n) * ", " * string(p))
    lfp, lfn, ifp, ifn = iht_lasso_poisson(n, p)

    lasso_false_positives += lfp
    lasso_false_negatives += lfn
    iht_false_positives += ifp
    iht_false_negatives += ifn
end
println("Lasso: Total number of false positives = $lasso_false_positives")
println("Lasso: Total number of false negatives = $lasso_false_negatives")
println("IHT  : Total number of false positives = $iht_false_positives")
println("IHT  : Total number of false negatives = $iht_false_negatives")








using Revise
using GLMNet #julia wrapper for GLMNet package in R, which calls fortran
using GLM
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using Random
using LinearAlgebra
using MendelBase

n, p = 223, 2411

#set random seed
Random.seed!(1234)

#define maf and true model size
bernoulli_rates = 0.5rand(p)
k = 5

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, bernoulli_rates)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

# Simulate poisson data
y_temp = xbm * true_b
λ = exp.(y_temp) #inverse log link
y = [rand(Poisson(x)) for x in λ]
y = Float64.(y)

x_float = convert(Matrix{Float64}, x, center=true, scale=true)
x_true = [x_float[:, correct_position] z]
regular_result = glm(x_true, y, Poisson(), LogLink())
regular_result = regular_result.pp.beta0
true_model = true_b[correct_position]
compare_model = DataFrame(
    true_β  = true_model, 
    regular_β = regular_result[1:end-1])









#problem diagnosis
using DelimitedFiles
using Statistics
using Plots

#doesn't work at all
sim = 28

#works well
sim = 25
sim = 1

y = readdlm("/Users/biona001/.julia/dev/MendelIHT/docs/IHT_GLM_simulations_mean/data_simulation_$sim" * ".txt")
mean(y)
var(y)
histogram(y)






