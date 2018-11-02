using IHT
using SnpArrays
using DataFrames
using Distributions
srand(1111)

#specify dimension and noise of data
n = 1000       # number of cases
p = 10000      # number of predictors
k = 10         # number of true predictors per group
J = 1          # number of active groups
s = 0.01       # variance of noise

#construct snpmatrix, covariate files, and true model b
x       	= SnpArray(rand(0:2, n, p)) # a random snpmatrix
z           = ones(n, 1)                # non-genetic covariates, just the intercept
true_b      = zeros(p)				    # model vector
true_b[1:k] = randn(k)			  	    # Initialize k non-zero entries in the true model
shuffle!(true_b)					    # Shuffle the entries
correct_position = find(true_b)		    # keep track of what the true entries are
noise = rand(Normal(0, s), n)			# noise

#compute mean and std for the simulated data
mean_vec, minor_allele = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate the phenotype 
y = zeros(n)
SnpArrays.A_mul_B!(y, x, true_b, mean_vec, std_vec)
y .+= noise #add N(0, 0.01) noise

# Construct IHTVariable needed to run L0_reg
v = IHTVariables(x, z, y, J, k)

#compute IHT result and compare it
result = L0_reg(v, x, z, y, J, k)

#compare and contrast
true_model = true_b[correct_position]
estimated_model = result.beta[correct_position]
compare_model = DataFrame(true_β = true_model, estimated_β = estimated_model)
