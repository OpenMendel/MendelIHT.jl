using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

function time_poisson_response(
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

    #free memory from xbm.... does this work like I think it works?
	xbm = nothing
	GC.gc()

	#time the result and return
    result = @timed L0_poisson_reg(x, z, y, 1, k, glm = "poisson", debias=debias, convg=true, show_info=false, true_beta=true_b, scale=false, init=false)
    iter = result[1].iter
    runtime = result[2]      # seconds
    memory = result[3] / 1e6 # MB

    return iter, runtime, memory
end

function main(n :: Int64, i :: Int64)
	p = 1000
	k = 10
	debias = false
	response = "poisson"

	#run result: first run on super small data to force compile, then run result
	iter, runtime, memory = time_poisson_response(100, 100, k, debias)
	iter, runtime, memory = time_poisson_response(n, p, k, debias)

	#write result to corresponding file
	if debias
		dir = response * "_results/"
	else
		dir = response * "_results_nodebias/"
	end
	file = string(n) * "_by_" * string(p) * "_run$i"
	open(dir * file, "w") do f
		write(f, "time (seconds), memory (MB), iteration\n")
		write(f, string(runtime) * ", " * string(memory) * ", " * string(iter))
	end
end

# arguments passed from command line
n = parse(Int64, ARGS[1])
i = parse(Int64, ARGS[2]) #i'th run: time result 5 times so we can average (didn't use benchmarktools cuz takes too long)

main(n, i)

