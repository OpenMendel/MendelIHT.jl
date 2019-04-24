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







function simulate_random_snparray(
    n :: Int64,
    p :: Int64,
)
    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    A1 = BitArray(undef, n, p)
    A2 = BitArray(undef, n, p)
    mafs = zeros(Float64, p)
    for j in 1:p
        minor_alleles = 0
        maf = 0
        while minor_alleles <= 5
            maf = 0.5rand()
            for i in 1:n
                A1[i, j] = rand(Bernoulli(maf))
                A2[i, j] = rand(Bernoulli(maf))
            end
            minor_alleles = sum(view(A1, :, j)) + sum(view(A2, :, j))
        end
        mafs[j] = maf
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(A1, A2), mafs
end

function _make_snparray(A1 :: BitArray, A2 :: BitArray)
    n, p = size(A1)
    x = SnpArray(undef, n, p)
    for i in 1:(n*p)
        c = A1[i] + A2[i]
        if c == 0
            x[i] = 0x00
        elseif c == 1
            x[i] = 0x02
        elseif c == 2
            x[i] = 0x03
        else
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end









using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#run directly
function should_be_slow(n :: Int64, p :: Int64)
    #set random seed
    Random.seed!(1111)
    simulate_random_snparray(n, p)
end
@time should_be_slow(10000, 100000)




using Revise
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#run once with super small data
function should_be_fast(n :: Int64, p :: Int64)
    #set random seed
    Random.seed!(1111)
    simulate_random_snparray(n, p)
end
should_be_fast(10, 10)
@time should_be_fast(10000, 100000)







function sum_random(x :: Vector{Float64})
   s = 0.0
   for i in x
       s += i
   end
   return s
end

#lets say I really want to sum 10 million random numbers
x = rand(3 * 10^8)
@time sum_random(x) # 2.419125 seconds (22.48 k allocations: 1.161 MiB)


#Can I first run the code on a small problem, and then my larger problem?
x = rand(3 * 10^8)
y = rand(10)
@time sum_random(y) # 0.010076 seconds (22.48 k allocations: 1.161 MiB)
@time sum_random(x) # 0.324422 seconds (5 allocations: 176 bytes)









using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

function time_normal_response(
    n :: Int64,  # number of cases
    p :: Int64,  # number of predictors (SNPs)
    k :: Int64,  # number of true predictors per group
    debias :: Bool # whether to debias
)
    Random.seed!(1111)

    #construct snpmatrix, covariate files, and true model b
    x, maf = simulate_random_snparray(n, p)
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # non-genetic covariates, just the intercept
    true_b = zeros(p)
    true_b[1:k] = randn(k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)
    noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

    #simulate phenotypes (e.g. vector y) via: y = Xb + noise
    y = xbm * true_b + noise

    #free memory from xbm.... does this work like I think it works?
    xbm = nothing
    GC.gc()

    #time the result and return
    result = @timed L0_normal_reg(x, z, y, 1, k, debias=debias)
    iter = result[1].iter
    runtime = result[2]      # seconds
    memory = result[3] / 1e6 # MB

    return iter, runtime, memory
end


#time directly:
n = 10000
p = 30000
k = 10
debias = false
iter, runtime, memory = time_normal_response(n, p, k, debias)
#(7, 17.276706309, 119.911048)

#run small example first:
n = 10000
p = 30000
k = 10
debias = false
iter, runtime, memory = time_normal_response(100, 100, 10, debias)
iter, runtime, memory = time_normal_response(n, p, k, debias)
#(14, 2.47283539, 37.778872)
#(7, 12.801054742, 82.478688)

#run same example twice:
n = 10000
p = 30000
k = 10
debias = false
iter, runtime, memory = time_normal_response(n, p, k, debias)
iter, runtime, memory = time_normal_response(n, p, k, debias)
#(7, 15.82780056, 119.911048)
#(7, 13.610244645, 82.478688)

#no GC.gc():
n = 10000
p = 30000
k = 10
debias = false
iter, runtime, memory = time_normal_response(100, 100, 10, debias)
iter, runtime, memory = time_normal_response(n, p, k, debias)
#(14, 2.661537431, 37.778872)
#(7, 12.977118564, 82.478688)





#NORMAL CROSS VALIDATION USING MEMORY MAPPED SNPARRAY - does it fix my problem?
using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using BenchmarkTools
using Distributed

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
# @time cv_iht_distributed(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=false, showinfo=false, parallel=true);

@benchmark cv_iht_distributed(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal", debias=false, showinfo=false, parallel=true) seconds=60

rm("tmp.bed", force=true)


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


#simulat data
n = 1000
p = 10000
k = 10    # number of true predictors

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # k true response
shuffle!(true_b)                           # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate bernoulli data
y_temp = xbm * true_b
prob = logistic.(y_temp) #inverse logit link
y = [rand(Bernoulli(x)) for x in prob]
y = Float64.(y)

#specify path and folds
path = collect(1:20)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
mses = cv_iht_test(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "logistic", debias=true)


K = collect(1:2:10)
errors = map(K) do k
    return k
end




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
true_b[1:k] = randn(k)
shuffle!(true_b)
correct_position = findall(x -> x != 0, true_b)
noise = rand(Normal(0, 0.1), n) # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#run k = 1,2,....,20
path = collect(1:20)

#compute IHT result for each k in path
iht_run_many_models(x, z, y, 1, path, "normal", use_maf = false, debias=false, showinfo=false, parallel=true)







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
using DelimitedFiles

#simulat data
n = 1000
p = 10000
k = 10 # number of true predictors

#set random seed
Random.seed!(2019)

#construct snpmatrix, covariate files, and true model b
x, maf = simulate_random_snparray(n, p, "normal.bed")
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
num_folds = 5
folds = rand(1:num_folds, size(x, 1))

# convert and save floating point version of the genotype matrix for glmnet
x_float = [convert(Matrix{Float64}, x, center=true, scale=true) z]
writedlm("x_float", x_float)
writedlm("y", y)
writedlm("folds", folds)




# trying GLM's fitting function
Random.seed!(2019)
n = 1000000
p = 10
x = randn(n, p)
b = randn(p)
L = LogitLink()

#simulate bernoulli data
y_temp = x * b
prob = linkinv.(L, y_temp) #inverse logit link
y = [rand(Bernoulli(i)) for i in prob]
y = Float64.(y)

glm_result = fit(GeneralizedLinearModel, x, y, Bernoulli(), L)
hi = glm_result.pp.beta0

glm_result_old = regress(x, y, "logistic")
hii = glm_result_old[1]

[b hi hii]






#deviance residual vs y - xb

function run_once()
    n = 1000
    p = 10000
    k = 10
    d = Poisson
    l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    x, maf = simulate_random_snparray(n, p, "tmp.bed")
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept
    true_b = zeros(p)
    d == Poisson ? true_b[1:k] = rand(Normal(0, 0.3), k) : true_b[1:k] = randn(k)
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate phenotypes (e.g. vector y) via: y = Xb + noise
    y_temp = xbm * true_b
    prob = linkinv.(l, y_temp)
    y = [rand(d(i)) for i in prob]
    y = Float64.(y)

    #run IHT
    result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true)

    #clean up
    rm("tmp.bed", force=true)

    found = length(findall(!iszero, result.beta[correct_position]))
    runtime = result.time

    return found, runtime
end

#set random seed
Random.seed!(1111)
function run()
    total_found = 0
    total_time = 0
    for i in 1:30
        f, t = run_once()
        total_found += f
        total_time += t
        println("finished $i run")
    end
    avg_found = total_found / 30
    avg_time = total_time / 30
    println(avg_found)
    println(avg_time)
end
run()





function loglik_obs(::Normal, y, μ, wt, ϕ) #this is wrong
    return wt*logpdf(Normal(μ, sqrt(ϕ)), y)
end

function test()
    y, mu = rand(1000), rand(1000)
    ϕ = MendelIHT.deviance(Normal(), y, mu)/length(y)
    ϕ = 1.0
    logl = 0.0
    for i in eachindex(y, mu)
        logl += loglik_obs(Normal(), y[i], mu[i], 1, ϕ)
    end
    println(logl)

    println(loglikelihood(Normal(), y, mu))
end
test() 






b = result.beta
μ = linkinv.(l, xbm*b)
# sum(logpdf.(Poisson.(μ), y))
loglikelihood(Poisson(), y, xbm*b)
loglikelihood_test(Poisson(), y, μ)









#debugging malloc: Incorrect checksum for freed object
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
d = Normal
l = canonicallink(d())

#set random seed
Random.seed!(1111)

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

#run IHT
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, show_info=false, convg=true)

train_idx = bitrand(n)
path = collect(1:20)

p, q = size(x, 2), size(z, 2)
nmodels = length(path)
betas = zeros(p, nmodels)
cs = zeros(q, nmodels)

x_train = SnpArray(undef, sum(train_idx), p)
copyto!(x_train, @view x[train_idx, :])
y_train = @view(y[train_idx])
z_train = @view(z[train_idx, :])
x_trainbm = SnpBitMatrix{Float64}(x_train, model=ADDITIVE_MODEL, center=true, scale=true); 

k = path[1]
debias = false
showinfo = false
d = d()
result = L0_reg(x_train, x_trainbm, z_train, y_train, 1, k, d, l, debias=debias, init=false, show_info=showinfo, convg=true)





#simulate correlated columns


n = 1000
p = 10000
k = 10
d = Bernoulli
l = canonicallink(d())

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, = simulate_random_snparray(n, p, "tmp.bed")

#ad hoc method to construct correlated columns
c = 0.9
for i in 1:size(x, 1)
    prob = rand(Bernoulli(c))
    prob == 1 && (x[i, 2] = x[i, 1]) #make 2nd column the same as first column 90% of the time
end

tmp = convert(Matrix{Float64}, @view(x[:, 1:2]), center=true, scale=true)
cor(tmp[:, 1], tmp[:, 2])








#testing if IHT finds intercept and/or nongenetic covariates, normal response
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
d = Normal
l = canonicallink(d())

#set random seed
Random.seed!(1111)

#construct snpmatrix, covariate files, and true model b
x, = simulate_random_snparray(n, p, "tmp")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
z = ones(n, 2) # the intercept
z[:, 2] .= randn(n)

#define true_b and true_c
true_b = zeros(p)
true_b[1:k-2] = randn(k-2)
shuffle!(true_b)
correct_position = findall(!iszero, true_b)
true_c = [3.0; 3.5]

#simulate phenotype
prob = linkinv.(l, xbm * true_b .+ z * true_c)
y = [rand(d(i)) for i in prob]

#run result
result = L0_reg(x, xbm, z, y, 1, k, d(), l, debias=false, init=false, use_maf=false)

#compare with correct answer
compare_model = DataFrame(
    position    = correct_position,
    true_β      = true_b[correct_position], 
    estimated_β = result.beta[correct_position])

compare_model = DataFrame(
    true_c      = true_c[1:2], 
    estimated_c = result.c[1:2])
