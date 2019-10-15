using Distributed
addprocs(5)

using MendelIHT
using SnpArrays
using DataFrames
using Distributions
using Random
using LinearAlgebra
using GLM
using CSV
using DelimitedFiles

# function for reading in result after cv
function read_mse_results()
	filename = "mse_model_"
	mses = zeros(50)
	for i in 1:50
		cur_file = filename * string(i)
		mses[i] = (isfile(cur_file) ? readdlm(cur_file)[1] : Inf)
	end
	return mses
end
mses = read_mse_results()
findmin(mses)

# using Plots
# plot(mses)

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
	y = T.(y)

	#specify path and folds
	path = collect(1:10)
	num_folds = 5
	folds = rand(1:num_folds, size(x, 1))

	# run IHT
	scratch_folder = "/u/scratch/b/biona001/"
	cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, destin=scratch_folder, showinfo=false, folds=folds, debias=false, parallel=true)
end

#main function 
function run(cur_path::Int64)
	# d = Normal
	d = Bernoulli
	l = canonicallink(d())
	run_iht_once_to_force_compile(d, l)

	#Import covariates (data is in $SCRATCH on hoffman2)
	filedir = "/u/scratch/b/biona001/ukbiobank/"
	x = SnpArray(filedir * "ukb.plink.filtered.imputed.bed")
	z = CSV.read(filedir * "Covariate_Final.csv")
	z = convert(Matrix{Float64}, z)
	standardize!(z)
	z = [ones(size(z, 1)) z]

	# quantitative phenotype
	# y = CSV.read(filedir * "Covariate_polrGWAS.csv")[:AveSBP] # average systolic blood pressure 
	# y = convert(Vector{Float64}, y)

	# binary phenotype
	sbp = CSV.read(filedir * "Covariate_polrGWAS.csv")[:AveSBP] # average systolic blood pressure 
	dbp = CSV.read(filedir * "Covariate_polrGWAS.csv")[:AveDBP] # average diastolic blood pressure 
	y = zeros(Float64, length(sbp))
	y[sbp .>= 140] .= 1.0
	y[dbp .>= 90]  .= 1.0

	#set random seed
	Random.seed!(2019)

	#specify path and folds
	path = [cur_path]
	num_folds = 5
	folds = rand(1:num_folds, size(x, 1))

	# run cross validation
	scratch_folder = "/u/scratch/b/biona001/"
	mses = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, destin=scratch_folder, folds=folds, debias=false, parallel=true)
	writedlm("mse_model_$(path[1])", mses)
end

# arguments passed from command line on hoffman cluster
const cur_path = parse(Int64, ARGS[1])

run(cur_path)


