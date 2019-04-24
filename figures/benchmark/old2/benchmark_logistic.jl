using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

function time_logistic_response(
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

    #free memory from xbm.... does this work like I think it works?
	xbm = nothing
	GC.gc()

	#time the result and return
    result = @timed L0_logistic_reg(x, z, y, 1, k, glm = "logistic", debias=debias, show_info=false, convg=true, init=false)
    iter = result[1].iter
    runtime = result[2]      # seconds
    memory = result[3] / 1e6 # MB

    return iter, runtime, memory
end

function main(n :: Int64, i :: Int64)
	p = 1000
	k = 10
	debias = true
	response = "logistic"

	#run result: first run on super small data to force compile, then run result
	iter, runtime, memory = time_logistic_response(100, 100, k, debias)
	iter, runtime, memory = time_logistic_response(n, p, k, debias)

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


