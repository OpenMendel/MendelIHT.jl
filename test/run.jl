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
p = 10100

#set random seed
Random.seed!(1111)

k = 10 # number of true predictors
bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
x = simulate_random_snparray(n, p, bernoulli_rates)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes (e.g. vector y) via: y = Xb
y_temp = xbm * true_b

# Apply inverse logit link and sample from the vector of distributions
prob = logistic.(y_temp) #inverse logit link
y = [rand(Bernoulli(x)) for x in prob]
y = Float64.(y)

#compute logistic IHT result
# result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic")
result = L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=true, show_info=true)

# @benchmark L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic") seconds = 30

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


#sizes that does not work (well):
# n, p = 999, 10000
# n, p = 2999, 10000
# n, p = 2000, 20001
# n, p = 140, 420

#simulat data
n = 2000
p = 10001 #20001 does not work!

#set random seed
Random.seed!(1111)

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

#compute poisson IHT result
result = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false, true_beta=true_b)

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
b = zeros(p)
b[366] = 1.4
b[1323] = -0.8
b[1447] = -0.1
b[1686] = -2.7
b[2531] = 0.3
b[3293] = 1.4 
b[4951] = 0.4
b[5078] = 0.1
b[6180] = 0.5
b[7048] = 0.2
xb = xbm * b
xb = exp.(xb) #apply inverse link: E(Y) = g^-1(Xβ)

result = xbm * result.beta
result = exp.(result)
[y result xb]
[y-result y-xb]
sum(abs2, y-result), sum(abs2, y-xb) 
abs.(y-result) .> abs.(y-xb)





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

#set random seed
Random.seed!(1111)

#simulat data
n = 2000
p = 20000
k = 12    # number of true predictors
bernoulli_rates = 0.5rand(p) #minor allele frequencies are drawn from uniform (0, 0.5)
x = simulate_random_snparray(n, p, bernoulli_rates)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes: y = Xb
y_temp = xbm * true_b

# Apply inverse logit link and sample from the vector of distributions
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
path = collect(1:20)
num_folds = 5
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








#below are poisson simulation for lasso
Random.seed!(2019)
n = 120
p = 1030
x = rand(n, p)
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

y_temp = x * true_b
λ = exp.(y_temp) #inverse log link
y = [rand(Poisson(x)) for x in λ]
y = Float64.(y)

path = glmnet(x, y, Poisson())
cv = glmnetcv(x, y, Poisson())
best = argmin(cv.meanloss)
result = cv.path.betas[:, best]

estimated_models = result[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)


