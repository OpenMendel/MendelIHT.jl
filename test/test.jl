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
    iht_result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false, true_beta=true_b)

    #compute poisson lasso result
    x_float = [convert(Matrix{Float64}, x, center=true, scale=true) z]
    path = glmnet(x_float, y, Poisson())
    cv = glmnetcv(x_float, y, Poisson())
    best = argmin(cv.meanloss)
    lasso_result = cv.path.betas[:, best]

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
end


Random.seed!(2019)
for i = 1:25
    @info("running the $i th model")
    n = rand(100:1000) 
    p = rand(1:10)n
    println("n, p = " * string(n) * ", " * string(p))
    iht_lasso_poisson(n, p)
end
