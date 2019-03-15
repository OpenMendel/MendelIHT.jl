#BELOW ARE SIMULATION FOR normal, bernoulli, and poisson
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
d = Poisson
l = canonicallink(d())
# g = Int[]
# w = Float64[]

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, = simulate_random_snparray(n, p, "tmp.bed")
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

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)
# @benchmark L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false)
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



#BELOW ARE SIMULATION for negative binomial
using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using GLM

#simulat data with k true predictors
n = 1000
p = 10000
k = 10
d = NegativeBinomial
l = LogLink()
nn = 10 #number of successes until stopping

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
prob = 1 ./ (1 .+ μ ./ nn)
y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
y = Float64.(y)

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=false)
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
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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
l = LogLink()
α = 1 #shape parameter for gamma

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
# true_b[1:k] = randn(k)
true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
β = 1 ./ μ #here β is the rate parameter for gamma distribution
y = [rand(d(α, i)) for i in β]

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=true, init=false, show_info=false, convg=true)
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
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 1) # the intercept
true_b = zeros(p)
true_b[1:k] = randn(k)
# true_b[1:k] = rand(Normal(0, 0.3), k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)

#simulate phenotypes (e.g. vector y) 
μ = linkinv.(l, xbm * true_b)
mean_parameter = 1 ./ μ #mean parameter for inverse gaussian distribution
y = [rand(d(i, λ)) for i in mean_parameter]

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=true, init=false, show_info=false, convg=true)
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
d = Poisson
l = canonicallink(d())

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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
mses = cv_iht_distributed(d(), l, x, z, y, 1, path, folds, num_folds, use_maf=false, debias=true, parallel=true);

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
n = 1000
p = 12000
k = 10 # number of true predictors
d = Poisson
l = canonicallink(d())

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
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

#clean up
rm("tmp.bed", force=true)
