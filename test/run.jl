using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using DelimitedFiles
using BenchmarkTools
using Random
using LinearAlgebra
using GLM
using Plots

#simulat data with k true predictors, from distribution d and with link l.
n = 5000
p = 30000
k = 10
d = Normal
l = canonicallink(d())
# l = LogLink()

#set random seed
Random.seed!(1111)

#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
x = simulate_random_snparray(n, p, "test1.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1)

# simulate response, true model b, and the correct non-0 positions of b
true_b = zeros(p)
# true_b[1:5] = [0.01; 0.05; 0.1; 0.25; 0.5]
true_b[1:k] .= collect(0.1:0.1:1.0)
true_c = [4.0]
# true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
if d == Normal || d == Bernoulli || d == Poisson
    prob = GLM.linkinv.(l, xbm * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
    # prob = linkinv.(l, xbm * true_b + z * true_c)
    # clamp!(prob, -20, 20)
    # y = [rand(d(i)) for i in prob]
    # k = k + 1
elseif d == NegativeBinomial
    nn = 1
    μ = GLM.linkinv.(l, xbm * true_b)
    # μ = linkinv.(l, xbm * true_b + z * true_c)
    # k = k + 1
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ nn)
    y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
elseif d == Gamma
    μ = GLM.linkinv.(l, xbm * true_b)
    β = 1 ./ μ # here β is the rate parameter for gamma distribution
    y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
end
y = Float64.(y)
# histogram(y, bins=30)
# mean(y)
# var(y)

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false)
@benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false)

#check result
compare_model = DataFrame(
    position    = correct_position,
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("test1.bed", force=true)



# SIMULATION FOR FLOATING POINT MATRICES
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using DelimitedFiles
using BenchmarkTools
using Random
using LinearAlgebra
using GLM
using Plots

#simulat data with k true predictors, from distribution d and with link l.
n = 5000
p = 30000
k = 10
d = Normal
l = canonicallink(d())
# l = LogLink()

#set random seed
Random.seed!(1111)

#construct x matrix and non genetic covariate (intercept)
T = Float32
x = randn(T, n, p)
z = ones(T, n, 1)

# simulate response, true model b, and the correct non-0 positions of b
true_b = zeros(T, p)
true_b[1:k] .= collect(0.1:0.1:1.0)
true_c = [4.0]
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
if d == Normal || d == Bernoulli || d == Poisson
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
    # prob = linkinv.(l, x * true_b + z * true_c)
    # clamp!(prob, -20, 20)
    # y = [rand(d(i)) for i in prob]
    # k = k + 1
elseif d == NegativeBinomial
    nn = 10
    μ = GLM.linkinv.(l, x * true_b)
    # μ = linkinv.(l, x * true_b + z * true_c)
    # k = k + 1
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ nn)
    y = [rand(d(nn, Float64(i))) for i in prob] #number of failtures before nn success occurs
elseif d == Gamma
    μ = GLM.linkinv.(l, x * true_b)
    β = 1 ./ μ # here β is the rate parameter for gamma distribution
    y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
end
y = T.(y)
# histogram(y, bins=30)
mean(y)
var(y)

#run IHT
result = L0_reg(x, z, y, 1, k, d(), l, debias=false)
@benchmark result = L0_reg(x, z, y, 1, k, d(), l, debias=false)

#check result
compare_model = DataFrame(
    position    = correct_position,
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))




#BELOW ARE SIMULATION FOR binomial
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors, from distribution d and with link l.
n = 1000
p = 10000
k = 10
d = Binomial
l = CloglogLink()
nn = 10 #number of tries for binomial/negative-binomial

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
prob = μ ./ nn
y = [rand(d(nn, i)) for i in prob]
y = Float64.(y)

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true) seconds = 60

#check result
compare_model = DataFrame(
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("tmp.bed", force=true)





#BELOW ARE SIMULATION FOR gamma
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors, from distribution d and with link l.
n = 1000
p = 10000
k = 10
d = Gamma
# l = canonicallink(d())
l = LogLink()
θ = 3 #scale parameter for gamma

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
# true_b[1:k] = randn(k)
true_b[1:k] = rand(Normal(0, 1.0), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
clamp!(μ, -20, 20)
y = [rand(d(i, θ)) for i in μ]
histogram(y)

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true) seconds = 60

#check result
compare_model = DataFrame(
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("tmp.bed", force=true)






#BELOW ARE SIMULATION FOR inverse Gaussian
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors, from distribution d and with link l.
n = 1000
p = 10000
k = 10
d = InverseGaussian
l = LogLink()
λ = 1 # shape parameter for inverse gaussian

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
# true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
clamp!(μ, -20, 20)
mean_parameter = 1 ./ μ #mean parameter for inverse gaussian distribution
y = [rand(d(i, λ)) for i in mean_parameter]
clamp!(y, 0, 20)

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=true, init=false, show_info=false)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true) seconds = 60

#check result
compare_model = DataFrame(
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("tmp.bed", force=true)








############## CROSS VALIDATION SIMULATION for normal, bernoulli, and Poisson

#first add workers
using Distributed
addprocs(4)
nprocs()

#load packages into all worker
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using GLM

#simulat data
n = 1000
p = 10000
k = 10 # number of true predictors
d = Normal
l = canonicallink(d())

#set random seed
Random.seed!(2018)

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
d == Poisson ? true_b[1:k] = rand(Normal(0, 0.3), k) : true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
y_temp = xbm * true_b
prob = linkinv.(l, y_temp)
y = [rand(d(i)) for i in prob]
y = Float64.(y)

#specify path and folds
path = collect(1:20)
num_folds = 4
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, use_maf=false, debias=true, parallel=true);

#compute l0 result using best estimate for k
k_est = argmin(mses)
result = L0_reg(x, xbm, z, y, 1, k_est, d(), l, debias=false, init=false, show_info=false)

#check result
compare_model = DataFrame(
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("tmp.bed", force=true)


############## RUNNING A BUNCH OF MODELS for normal, bernoulli, and Poisson

#first add workers
using Distributed
addprocs(4)
nprocs()

#load packages into all worker
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using GLM

#simulat data
n = 2000
p = 10000
k = 10 # number of true predictors
d = Normal
l = canonicallink(d())

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
d == Poisson ? true_b[1:k] = rand(Normal(0, 1.0), k) : true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
y_temp = xbm * true_b
prob = linkinv.(l, y_temp)
y = [rand(d(i)) for i in prob]
y = Float64.(y)

#specify path (i.e. all models `k` you want to test)
path = collect(1:20)

#run results
result = iht_run_many_models(d(), l, x, z, y, 1, path, parallel=true, debias=true);
@benchmark iht_run_many_models(d(), l, x, z, y, 1, path, parallel=true, debias=true) seconds=60


#clean up
rm("tmp.bed", force=true)



#RUN IHT with all CPU available
using Distributed
addprocs(8)
nprocs()

using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors, from distribution d and with link l.
n = 1000
p = 10000
k = 10
d = NegativeBinomial
# l = canonicallink(d())
l = LogLink()

#set random seed
Random.seed!(33)

#construct snpmatrix, covariate files, and true model b
T = Float32
x = simulate_random_snparray(n, p, undef)
xbm = SnpBitMatrix{T}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(T, n, 1) # the intercept

# simulate response, true model b, and the correct non-0 positions of b
true_b = zeros(T, p)
# true_b[1:4] .= [0.1; 0.25; 0.5; 0.8]
true_b[1:10] .= collect(0.1:0.1:1.0)
# true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
if d == Normal || d == Poisson || d == Bernoulli
    prob = GLM.linkinv.(l, xbm * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
elseif d == NegativeBinomial
    nn = 10
    μ = GLM.linkinv.(l, xbm * true_b)
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ nn)
    y = [rand(d(nn, Float64(i))) for i in prob] #number of failtures before nn success occurs
elseif d == Gamma
    μ = GLM.linkinv.(l, xbm * true_b)
    β = 1 ./ μ # here β is the rate parameter for gamma distribution
    y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
end
y = T.(y)
# histogram(y)
# var(y) / mean(y)

#specify path and folds
path = collect(1:20)
num_folds = 8
folds = rand(1:num_folds, size(x, 1))

# run threaded IHT
# result = iht_run_many_models(d(), l, x, z, y, 1, path);
destin = "/Users/biona001/Desktop/"
mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, destin=destin, folds=folds, init=false, use_maf=false, debias=false, parallel=true)
mses = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, destin=destin, folds=folds, init=false, use_maf=false, debias=false, parallel=true)

#benchmarking
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=false, parallel=false) seconds=30 #33.800s, 135.13MiB
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=false, parallel=true) seconds=30  #6.055 s, 34.23 MiB

#run IHT
result = L0_reg(x, xbm, z, y, 1, argmin(mses), d(), l, debias=true, init=false, use_maf=false)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false) seconds=60
# @code_warntype L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false)

#check result
compare_model = DataFrame(
    position    = correct_position,
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))

#clean up
rm("tmp.bed", force=true)


# CROSS VALIDATION ON GENERAL MATRIX
using Distributed
addprocs(8)
nprocs()

# @everywhere using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors, from distribution d and with link l.
n = 2000
p = 10000
k = 10
d = Normal
l = canonicallink(d())
# l = LogLink()

#set random seed
Random.seed!(2019)

#construct x matrix and non genetic covariate (intercept)
T = Float32
x = randn(T, n, p)
z = ones(T, n, 1)

# simulate response, true model b, and the correct non-0 positions of b
true_b = zeros(T, p)
true_b[1:k] .= collect(0.1:0.1:1.0)
true_c = [T.(4.0)]
shuffle!(true_b)
correct_position = findall(!iszero, true_b)

#simulate phenotypes (e.g. vector y)
if d == Normal || d == Bernoulli || d == Poisson
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
    # prob = linkinv.(l, x * true_b + z * true_c)
    # clamp!(prob, -20, 20)
    # y = [rand(d(i)) for i in prob]
    # k = k + 1
elseif d == NegativeBinomial
    nn = 10
    μ = GLM.linkinv.(l, x * true_b)
    # μ = linkinv.(l, x * true_b + z * true_c)
    # k = k + 1
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ nn)
    y = [rand(d(nn, Float64(i))) for i in prob] #number of failtures before nn success occurs
elseif d == Gamma
    μ = GLM.linkinv.(l, x * true_b)
    β = 1 ./ μ # here β is the rate parameter for gamma distribution
    y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
end
y = T.(y)
mean(y)
var(y)

#specify path and folds
path = collect(1:50)
num_folds = 8
folds = rand(1:num_folds, size(x, 1))

# run threaded IHT
# result = iht_run_many_models(d(), l, x, z, y, 1, path);
mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true)
mses = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true)

# default number of threads in BLAS
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30#14.146 s, 1.20 GiB
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30 #17.019 s, 619.66 MiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30 #14.475 s, 1.21 GiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30  #12.904 s, 295.80 KiB

BLAS.set_num_threads(1)
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30#17.346 s, 1.20 GiB
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30 #16.970 s, 619.66 MiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30 #17.014 s, 1.21 GiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30  #12.686 s, 295.80 KiB

#export OPENBLAS_NUM_THREADS=1
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30#17.485 s, 1.20 GiB
@benchmark cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30 #17.867 s, 619.66 MiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=false) seconds=30 #18.230 s, 1.21 GiB
@benchmark cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, parallel=true) seconds=30  #12.991 s, 295.80 KiB


#run IHT
result = L0_reg(x, z, y, 1, argmin(mses), d(), l, debias=true, init=false, use_maf=false)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false) seconds=60
# @code_warntype L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false)

#check result
compare_model = DataFrame(
    position    = correct_position,
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])
println("Total iteration number was " * string(result.iter))
println("Total time was " * string(result.time))
println("Total found predictors = " * string(length(findall(!iszero, result.beta[correct_position]))))






######## Esitmate negative binomial nuisance parameters

# using Distributed
#addprocs(4)

#load necessary packages for running all examples below
using Revise
# using Debugger
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using SpecialFunctions
using GLM
using DelimitedFiles
using Statistics
using BenchmarkTools


n = 1000
p = 10000
k = 10
d = NegativeBinomial
l = LogLink()
est_r = :MM     # Update r using MM algorithm

# set random seed for reproducibility
Random.seed!(728) 

# simulate SNP matrix, store the result in a file called tmp.bed
x = simulate_correlated_snparray(n, p, "tmp.bed")

#construct the SnpBitMatrix type (needed for L0_reg() and simulate_random_response() below)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

# intercept is the only nongenetic covariate
z = ones(n, 1) 

# simulate response y, true model b, and the correct non-0 positions of b
y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l);

# test L0_reg
result = L0_reg(x, xbm, z, y, 1, k, d(), l, est_r)

mu = rand(1000)
@code_warntype mle_for_r(y, mu, 1.0, :MM)

#cross validation
path = collect(1:20)
num_folds = 3
mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, parallel=true, est_r=est_r);
k_est = argmin(mses)
result = L0_reg(x, xbm, z, y, 1, k_est, d(), l, est_r=est_r)


# not using k_est from cross validation
#result = L0_reg(x, xbm, z, y, 1, k, d(), l, est_r=est_r)
#@benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, est_r=est_r)

compare_model = DataFrame(
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position],
    diff = true_b[correct_position] - result.beta[correct_position])

sum(compare_model.diff)

#@show compare_model

#clean up
rm("tmp.bed", force=true)




using Revise
using MendelIHT
using Random

J, k, n = 2, 0.9, 20
y = ones(n)
y[5] = 3
y_copy = copy(y)
group = repeat(1:5, inner=4)
project_group_sparse!(y, group, J, k)
for i = 1:length(y)
    println(i,"  ",group[i],"  ",y[i],"  ",y_copy[i])
end


Random.seed!(2019)
J, k, n = 2, 0.5, 9
y = 5rand(n)
y_copy = copy(y)
group = repeat(1:3, inner=3)
project_group_sparse!(y, group, J, k)
for i = 1:length(y)
    println(i,"  ",group[i],"  ",y[i],"  ",y_copy[i])
end




