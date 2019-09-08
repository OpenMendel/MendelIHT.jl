using Distributed
addprocs(16)

using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using GLM
using CSV

# function for converting nongenetic covariates 
function make_nongenetic_covariates()
	z = CSV.read("Covariate_polrGWAS.csv")
	z = [z[:, 3:6] z[:, 8] z[:, 12:end]]
	rename!(z, :x1 => :bmi)
	CSV.write("Covariate_Final.csv", z)
end

# runs a small example so cross validation code compiles 
function run_iht_once_to_force_compile(d, l)
	n = 1000
	p = 10000
	k = 10

	#set random seed
	Random.seed!(2019)

	#construct x matrix and non genetic covariate (intercept)
	T = Float64
	x = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{T}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(T, n, 1)

	# simulate response, true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)

	#simulate phenotypes (e.g. vector y)
	if d == Normal || d == Bernoulli || d == Poisson
	    prob = linkinv.(l, xbm * true_b)
	    clamp!(prob, -20, 20)
	    y = [rand(d(i)) for i in prob]
	    # prob = linkinv.(l, x * true_b + z * true_c)
	    # clamp!(prob, -20, 20)
	    # y = [rand(d(i)) for i in prob]
	    # k = k + 1
	elseif d == NegativeBinomial
	    nn = 10
	    μ = linkinv.(l, xbm * true_b)
	    # μ = linkinv.(l, x * true_b + z * true_c)
	    # k = k + 1
	    clamp!(μ, -20, 20)
	    prob = 1 ./ (1 .+ μ ./ nn)
	    y = [rand(d(nn, Float64(i))) for i in prob] #number of failtures before nn success occurs
	elseif d == Gamma
	    μ = linkinv.(l, xbm * true_b)
	    β = 1 ./ μ # here β is the rate parameter for gamma distribution
	    y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
	end
	y .= T.(y)

	#specify path and folds
	path = collect(1:10)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run IHT
	cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, debias=false, parallel=true)
end

function run()
	d = Normal
	l = canonicallink(d())
	run_iht_once_to_force_compile(d, l)

	#Import covariates and phenotype. (data is in $SCRATCH)
	filedir = "/u/scratch/b/biona001/"
	x = SnpArray(filedir * "ukb.plink.filtered.bed")
	z = CSV.read(filedir * "Covariate_Final.csv")
	z = convert(Matrix{Float64}, z)
	standardize!(z)
	z = [ones(size(z, 1)) z]
	y = CSV.read(filedir * "Covariate_polrGWAS.csv")[:AveSBP] # average systolic blood pressure
	y = convert(Vector{Float64}, y)

	#set random seed
	Random.seed!(2019)

	#specify path and folds
	path = collect(1:5)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run cross validation
	mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, debias=false, parallel=true)
	writedlm("mses_$(first(path))_to_$(last(path))", mses)
end
