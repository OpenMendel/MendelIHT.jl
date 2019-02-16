#BELOW ARE NORMAL SIMUATIONS
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#simulat data
n = 2000
p = 10000
bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
x = simulate_random_snparray(n, p, bernoulli_rates)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 40              # number of true predictors
s = 0.1            # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#compute IHT result for less noisy data
result = L0_reg(x, z, y, 1, k, debias=false)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))

#how to get predicted response?
xb = zeros(y_temp)
SnpArrays.A_mul_B!(xb, x, result.beta, mean_vec, std_vec)
[y xb]



# path = collect(1:20)
# num_folds = 5
# folds = rand(1:num_folds, size(x, 1))
# cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal")





#BELOW ARE LOGISTIC SIMUATIONS
#load packages
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

#simulat data
n = 2000
p = 10000
k = 10 # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # non-genetic covariates, just the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate bernoulli data
y_temp = xbm * true_b
prob = logistic.(y_temp) #inverse logit link
y = [rand(Bernoulli(x)) for x in prob]
y = Float64.(y)

#compute logistic IHT result
result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=true, show_info=false, convg=true,init=false)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))


#how to get predicted response?
xb = zeros(y_temp)
SnpArrays.A_mul_B!(xb, x, result.beta, mean_vec, std_vec)
xb = logistic.(xb) #apply inverse link: E(Y) = g^-1(Xβ)
# [y round(xb)]
[y xb]






# BELOW ARE POISSON SIMULATIONS
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using Random
using LinearAlgebra

#simulat data
n = 2000
p = 20000
k = 30 # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # non-genetic covariates, just the intercept
true_b = zeros(p)
true_b[1:k] = rand(Normal(0, 0.4), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

# Simulate poisson data
y_temp = xbm * true_b
λ = exp.(y_temp) #inverse log link
y = [rand(Poisson(x)) for x in λ]
y = Float64.(y)

#compute poisson IHT result
result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=true, convg=false, show_info=false, true_beta=true_b, scale=false, init=false)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))




############## NORMAL CROSS VALIDATION SIMULATION
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#simulat data
n = 3000
p = 20000
bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
x = simulate_random_snparray(n, p, bernoulli_rates)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 12    # number of true predictors
s = 0.1  # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#specify path and folds
path = collect(1:20)
num_folds = 5
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
k_est = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=false)

#compute l0 result using best estimate for k
l0_result = L0_reg(x, z, y, 1, k_est, debias=false)

#check result
estimated_models = l0_result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(l0_result.iter))
println("Total time was " * string(l0_result.time))





########### LOGISTIC CROSS VALIDATION SIMULATION CODE##############
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools
using Random
using LinearAlgebra


#simulat data
n = 2000
p = 20000
k = 12    # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = rand(Normal(0, 0.25), k)     # k true response
shuffle!(true_b)                           # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate bernoulli data
y_temp = xbm * true_b
prob = logistic.(y_temp) #inverse logit link
y = [rand(Bernoulli(x)) for x in prob]
y = Float64.(y)

#specify path and folds
path = collect(1:20)
num_folds = 5
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
k_est = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "logistic", debias=true)

#compute l0 result using best estimate for k
l0_result = L0_logistic_reg(x, z, y, 1, k_est, glm = "logistic", debias=true, show_info=false)

#check result
estimated_models = l0_result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(l0_result.iter))
println("Total time was " * string(l0_result.time))








############## POISSON CROSS VALIDATION SIMULATION
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#sizes that does not work (well):
# n, p = 999, 10000
# n, p = 2999, 10000
# n, p = 2000, 20001

#simulat data
n = 2000
p = 20000 #20001 does not work!
k = 10 # number of true predictors
bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)

#prevent rare alleles from entering model
# clamp!(bernoulli_rates, 0.1, 1.0)
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

#specify path and folds
path = collect(5:15)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
k_est = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf=false, glm="poisson", debias=false)



#compute poisson IHT result
result = L0_poisson_reg(x, z, y, 1, k_est, glm = "poisson", debias=false, show_info=true)

#check result
estimated_models = zeros(k)
estimated_models .= result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))









#Below attempts to tests lasso against models that does not work with poisson

using Revise
using Lasso
using Distributions
using DataFrames
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

#models that work well
n, p = 1770, 10620

#models that doesn't work (well)
n, p = 1154, 5770 
n, p = 1847, 3694
n, p = 570, 2280

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
    iht_β      		 = IHT_model,
    lasso_β			 = lasso_model)













#below are normal simulation for lasso
n = 100
p = 1000
x = rand(n, p)
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)
noise = rand(Normal(0, 0.1), n)
y = x * true_b + noise

#tried using Lasso.jl -> does not work due to POOR documentation
# result = fit(LassoPath, x, y, Normal()) 

#try using GLMNet now
path = glmnet(x, y)
cv = glmnetcv(x, y)
best = argmin(cv.meanloss)
result = cv.path.betas[:, best]

estimated_models = result[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)








using Revise
using Lasso
using Distributions
using DataFrames
using Random
using LinearAlgebra
using GLMNet

#below are poisson simulation for lasso
Random.seed!(2019)
n = 500
p = 1030
k = 10
x = rand(n, p)
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

y_temp = x * true_b
λ = exp.(y_temp) #inverse log link
y = [rand(Poisson(x)) for x in λ]
y = Float64.(y)

path = glmnet(x, y, Poisson(), dfmax=20)
cv = glmnetcv(x, y, Poisson(), dfmax=20, nfolds=5)
best = argmin(cv.meanloss)
result = cv.path.betas[:, best]

estimated_models = result[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("best model has " * string(length(findall(!iszero, result))) * " predictors")

