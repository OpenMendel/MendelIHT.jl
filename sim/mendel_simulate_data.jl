using IHT
using SnpArrays
using DataFrames
using Distributions
srand(1111)


# function test_cv()
# 	#specify dimension and noise of data
# 	n = 1000    # number of cases
# 	p = 8000    # number of predictors
# 	k = 10      # number of true predictors per group
# 	J = 1       # number of active groups
# 	s = 0.01    # variance deviation of noise

# 	#simulate the data
# 	x, z, y, b, b_true_position, mean_vec, std_vec, v = simulate_data(n, p, k, J, s)

# 	#compute IHT result and compare it side by side with the true model
# 	path      = collect(1:20) 
# 	num_folds = 5
# 	folds     = rand(1:num_folds, n)
# 	cv_iht(x, z, y, J, path, folds, num_folds)
# end



function simulate_data(
	n :: Int64,   # number of cases
	p :: Int64,   # number of predictors
	k :: Int64,   # number of true predictors per group
	J :: Int64,   # number of active groups
	s :: Float64  # variance deviation of noise
)
	#construct snpmatrix, covariate files, and true model b
	x      		= SnpArray(rand(0:2, n, p))    # a random snpmatrix (~37MB)
	z      		= ones(n, 1)                   # non-gentic covariates (only including grand mean)
	true_b      = zeros(p)				       # model vector
	c     		= randn() 					   # intercept
	true_b[1:k] = randn(k)			  	       # Initialize k non-zero entries in the true model
	shuffle!(true_b)					       # Shuffle the entries
	correct_position = find(true_b)		       # keep track of what the true entries are
	noise = rand(Normal(0, s), n)			   # noise

	#simulate the phenotype
	#y = x*true_b + c*z[:, 1] + noise
	y = x*true_b + c*z[:, 1] + noise

	return x, z, y, true_b, correct_position, c
end

function test_L0_reg()
	#specify dimension and noise of data
	n = 5000       # number of cases
	p = 30000      # number of predictors
	k = 10         # number of true predictors per group
	J = 1          # number of active groups
	s = 0.01       # variance of noise

	#simulate the data
	x, z, y, true_b, correct_position, c = simulate_data(n, p, k, J, s)

	# Construct IHTVariable needed to run L0_reg
    v = IHTVariables(x, z, y, J, k)

	#compute IHT result and compare it side by side with the true model
	result = L0_reg(v, x, z, y, J, k)
	estimated_position = find(result.beta)
	estimated_β = result.beta[correct_position]
	println("\n True intercept      = " * string(c))
	println(" Estimated intercept = " * string(result.c[1]) * "\n")
	compare = DataFrame(true_position = correct_position, estimated_position = estimated_position,
		true_β = true_b[correct_position], estimated_β = estimated_β)

	return compare
end
test_L0_reg()







