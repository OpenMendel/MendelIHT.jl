using IHT
using SnpArrays
using DataFrames
using Distributions
srand(1111)

#specify dimension and noise of data
n = 1000       # number of cases
p = 9999       # number of predictors
k = 10         # number of true predictors per group
J = 1          # number of active groups
s = 0.01       # variance of noise

#construct snpmatrix, covariate files, and true model b
x_temp  	= SnpArray(rand(0:2, n, p)) # a random snpmatrix
z           = ones(n, 1)                # non-genetic covariates, just the intercept
true_b      = zeros(p)				    # model vector
true_b[1:k] = randn(k)			  	    # Initialize k non-zero entries in the true model
shuffle!(true_b)					    # Shuffle the entries
correct_position = find(true_b)		    # keep track of what the true entries are
noise = rand(Normal(0, s), n)			# noise

#Next we configure a regression problem. In this case, we need a data matrix with a grand mean included:
x = SnpArray(n, p+1)
x[:, 1:p] .= x_temp
x[:, end] .= SnpArray(ones(n))

#simulate the phenotype (no intercept!)
y = x_temp*true_b + noise

# Construct IHTVariable needed to run L0_reg
v = IHTVariables(x, z, y, J, k)

#compute IHT result and compare it
result = L0_reg(v, x, z, y, J, k)

#compare and contrast
true_model = true_b[correct_position]
estimated_model = result.beta[correct_position]
compare_model = DataFrame(true_β = true_model, estimated_β = estimated_model)

# println("\n True intercept      = " * string(c))
# println(" Estimated intercept = " * string(result.c[1]) * "\n")
# compare_position = DataFrame(true_position = correct_position, estimated_position = estimated_position)
# compare_model = DataFrame(true_β = true_b[correct_position], estimated_β = estimated_model)
