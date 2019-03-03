#BELOW ARE NORMAL SIMUATIONS
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#simulat data
n = 5000
p = 30000
k = 10 # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # non-genetic covariates, just the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)
noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#compute IHT result for less noisy data
result = L0_normal_reg(x, xbm, z, y, 1, k, debias=false)
# @benchmark L0_normal_reg(x, xbm, z, y, 1, k, debias=false)


#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))

rm("tmp.bed", force=true)


#this code backtracks when loglikelihood decreases
# result = L0_normal_reg2(x, z, y, 1, k, debias=true)

# #check result
# estimated_models = result.beta[correct_position]
# true_model = true_b[correct_position]
# compare_model = DataFrame(
#     correct_position = correct_position, 
#     true_β           = true_model, 
#     estimated_β      = estimated_models)
# println("Total iteration number was " * string(result.iter))
# println("Total time was " * string(result.time))








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
n = 1200
p = 20000
k = 10 # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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
result = L0_logistic_reg(x, xbm, z, y, 1, k, glm = "logistic", debias=false, show_info=false, convg=true, init=false)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))

rm("tmp.bed", force=true)



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
n = 1000
p = 10000
k = 10 # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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
result = L0_poisson_reg(x, xbm, z, y, 1, k, glm = "poisson", debias=false, convg=false, show_info=false, true_beta=true_b, scale=false, init=false)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))

rm("tmp.bed", force=true)


############## NORMAL CROSS VALIDATION SIMULATION
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using Distributed

#add workers
addprocs(4)
nprocs()

#simulat data
n = 1000
p = 10000
k = 10 # number of true predictors

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # non-genetic covariates, just the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)
noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#specify path and folds
path = collect(1:20)
num_folds = 4
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
# mses = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=false)
mses = cv_iht_distributed(x, z, y, 1, path, folds, num_folds, "normal", use_maf = false, debias=false, showinfo=false, parallel=false)
# @benchmark cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=false) seconds=60
# @benchmark cv_iht_distributed($x, $z, $y, 1, $path, $folds, $num_folds, "normal", use_maf = false, debias=false, showinfo=false, parallel=false) seconds=60

rm("tmp.bed", force=true)


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
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using Distributed

#add workers
# addprocs(4)
# nprocs()

#simulat data
n = 1000
p = 10000
k = 10    # number of true predictors
glm = "logistic"

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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

#specify path and folds
path = collect(1:20)
num_folds = 5
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
mses = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "logistic", debias=true)
# mses = cv_iht_distributed(x, z, y, 1, path, folds, num_folds, glm, use_maf = false, debias=true)

rm("tmp.bed", force=true)


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
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using Distributed

# add workers
addprocs(4)
nprocs()

#simulat data
n = 5000
p = 10000
k = 10 # number of true predictors
glm="poisson"

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # non-genetic covariates, just the intercept
true_b = zeros(p)
true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

# Simulate poisson data
y_temp = xbm * true_b
λ = exp.(y_temp) #inverse log link
y = [rand(Poisson(x)) for x in λ]
y = Float64.(y)

#specify path and folds
path = collect(1:20)
num_folds = 5
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
# mses = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf=false, glm="poisson", debias=false)
mses = cv_iht_distributed(x, z, y, 1, path, folds, num_folds, glm, use_maf=false, debias=true)

rm("tmp.bed", force=true)



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

