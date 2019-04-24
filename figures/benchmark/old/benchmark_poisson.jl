using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

function _benchmark_normal_response(
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

	#benchmark the result and return
    b = @benchmarkable L0_normal_reg($x, $z, $y, 1, $k, debias=$debias) seconds=85000 samples=10
    
    #should also save the iteration number for this particular SnpBitMatrix
    iter = L0_normal_reg(x, z, y, 1, k).iter

    return run(b), iter
end

function _benchmark_logistic_response(
    n :: Int64,        # number of cases
    p :: Int64,        # number of predictors (SNPs)
    k :: Int64,        # number of true predictors per group
    debias :: Bool
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

	#simulate bernoulli data
	y_temp = xbm * true_b
	prob = logistic.(y_temp) #inverse logit link
	y = [rand(Bernoulli(x)) for x in prob]
	y = Float64.(y)

	#benchmark the result and return
    b = @benchmarkable L0_logistic_reg($x, $z, $y, 1, $k, glm = "logistic", debias=$debias) seconds=85000 samples=10

    #should also save the iteration number for this particular SnpBitMatrix
    iter = L0_logistic_reg(x, z, y, 1, k, glm = "logistic").iter

    return run(b), iter
end

function _benchmark_poisson_response(
    n :: Int64,        # number of cases
    p :: Int64,        # number of predictors (SNPs)
    k :: Int64,        # number of true predictors per group
    debias :: Bool
)
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

	#benchmark the result and return
    b = @benchmarkable L0_poisson_reg($x, $z, $y, 1, $k, glm = "poisson", debias=$debias, convg=false, show_info=false, scale=false, init=false) seconds=85000 samples=10

    #should also save the iteration number for this particular SnpBitMatrix
    iter = L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=debias, convg=false, show_info=false, scale=false, init=false).iter

    return run(b), iter
end

# arguments passed from command line
n = parse(Int64, ARGS[1])
p = 1000000
k = 10
debias = true
response = "poisson"

#do benchmarking
if response == "normal"
	b, iter = _benchmark_normal_response(n, p, k, debias)
elseif response == "logistic"
	b, iter = _benchmark_logistic_response(n, p, k, debias)
elseif response == "poisson"
	b, iter = _benchmark_poisson_response(n, p, k, debias)
else
	error("shouldn't have reached here")
end

#write result to corresponding file
if debias
	dir = response * "_results/"
else
	dir = response * "_results_nodebias/"
end
file = string(n) * "_by_" * string(p)
open(dir * file, "w") do f
	write(f, "time (seconds), memory (MB), iteration, number of samples\n")
	write(f, string(median(b.times)/1e9) * ", " * string(b.memory/1e6) * ", " * string(iter) * ", " * string(b.params.samples))
end


