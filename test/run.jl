#BELOW ARE NORMAL SIMUATIONS
using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#import snp data and construct SnpBitMatrix
const gwas1 = SnpArray("gwas 1 data.bed")
const gwas1bm = SnpBitMatrix{Float64}(gwas1, model=ADDITIVE_MODEL, center=true, scale=true); #construct SnpBitMatrix based on mouse SnpArrays

#compute dimension and specify noise of data
n, p = size(gwas1) # number of cases/predictors
k = 10             # number of true predictors
s = 0.1            # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = gwas1bm * true_b + noise

#compute IHT result for less noisy data
v = IHTVariables(gwas1, z, y, 1, k)
result = L0_reg(v, gwas1, z, y, 1, k)

#check result
estimated_models = result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))

#how to get predicted response?
xb = zeros(y_temp)
SnpArrays.A_mul_B!(xb, x, result.beta, mean_vec, std_vec)
[y xb]



# path = collect(1:20)
# num_folds = 5
# folds = rand(1:num_folds, size(x, 1))
# cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal")






#BELOW ARE LOGISTIC SIMUATIONS
#load packages
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools

#set random seed
srand(123) 

#specify dimension and noise of data
n = 5000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
# y_temp .+= noise #add some noise

# # Apply inverse logit link to map y to {0, 1} 
# y_old = 1 ./ (1 .+ exp.(-y_temp)) #inverse logit link
# y_old .= round.(y_old)                     #map y to 0, 1

# Apply inverse logit link to map y to {0, 1} 
y = logistic.(y_temp)               #inverse logit link
hi = rand(length(y))
y = Float64.(hi .< y)  #map y to 0, 1

#compute logistic IHT result
estimated_models = zeros(k)
v = IHTVariables(x, z, y, 1, k)
result = L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic")

# @benchmark L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic") seconds = 30

#check result
estimated_models .= result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))


#how to get predicted response?
xb = zeros(y_temp)
SnpArrays.A_mul_B!(xb, x, result.beta, mean_vec, std_vec)
xb = logistic.(xb) #apply inverse link: E(Y) = g^-1(Xβ)
# [y round(xb)]
[y xb]





# BELOW ARE POISSON SIMULATIONS
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic

#set random seed
srand(1111) 

#specify dimension and noise of data
n = 5000                        # number of cases
p = 100000                       # number of predictors
k = 10                          # number of true predictors per group
# s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
# noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
# y_temp .+= noise #add some noise

# Apply inverse logit link to map y to {0, 1} 
# poisson = 
y = zeros(n)
y_temp = exp.(y_temp)                  #inverse log link
for i in 1:n
	dist = Poisson(y_temp[i])
	y[i] = rand(dist)
end

#compute logistic IHT result
scale = sum(y) / n
estimated_models = zeros(k)
v = IHTVariables(x, z, y, 1, k)
result = L0_poisson_reg(v, x, z, y, 1, k, glm = "poisson")

#check result
estimated_models .= result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))


#how to get predicted response?
xb = zeros(y_temp)
SnpArrays.A_mul_B!(xb, x, result.beta, mean_vec, std_vec)
xb = exp.(xb) #apply inverse link: E(Y) = g^-1(Xβ)
[y xb]



########### LOGISTIC CROSS VALIDATION SIMULATION CODE##############
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools

#set random seed
srand(123) 

#specify dimension and noise of data
n = 5000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
# s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
# noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)

# Apply inverse logit link to map y to {0, 1} 
y = logistic.(y_temp)               #inverse logit link
hi = rand(length(y))
y = Float64.(hi .< y)  #map y to 0, 1

#specify path and folds
path = collect(1:5)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "logistic")





############## POISSON CROSS VALIDATION SIMULATION
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic

#set random seed
srand(1111) 

#specify dimension and noise of data
n = 2000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
# s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
# noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)

# Apply inverse link
y = zeros(n)
y_temp = exp.(y_temp)                  #inverse log link
for i in 1:n
	dist = Poisson(y_temp[i])
	y[i] = rand(dist)
end

#specify path and folds
path = collect(1:5)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "poisson")





############## NORMAL CROSS VALIDATION SIMULATION
using IHT
using SnpArrays
using DataFrames
using Distributions

#set random seed
srand(123) 

#specify dimension and noise of data
n = 2000                        # number of cases
p = 10000                       # number of predictors
k = 10                          # number of true predictors per group
s = 0.1                         # noise vector, from very little noise to a lot of noise

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
noise = rand(Normal(0, s), n)              # noise vectors from N(0, s) where s ∈ S = {0.01, 0.1, 1, 10}s

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
y = y_temp += noise

#specify path and folds
path = collect(1:5)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
result = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal")


































function test(
	p :: AbstractVector{Float64}, 
	x :: SnpLike{2},
	z :: AbstractMatrix{Float64},
	b :: AbstractVector{Float64},
	c :: AbstractVector{Float64}
)
	@inbounds @simd for i in eachindex(p)
        xβ = dot(view(x.A1, i, :), b) + dot(view(x.A2, i, :), b) + dot(view(z, i, :), c)
        p[i] = e^xβ / (1 + e^xβ)
    end
end

p = rand(100)
x = SnpArray(rand(0:2, 100, 100))
z = rand(100, 100)
b = rand(100)
c = rand(100)
test(p, x, z, b, c)

function test()
	p = 1000000
	k = 1000
	x = rand(0:2, p)
	β = zeros(p)
	β[1:100] = randn(100)
	shuffle!(β)
	e^dot(x, β) / (1 + e^dot(x, β))
end

using BenchmarkTools
srand(111)
function test()
	n = 100
	M = zeros(n, n)
	V = [rand(n, n) for _ in 1:40]
	for i in 1:40
		M = M + V[i]
	end
end
@benchmark test()

n = 100
M = zeros(n, n)
V = [rand(n, n) for _ in 1:40]
function test(hi::Matrix{Float64}, hii::Vector{Matrix{Float64}})
	for i in 1:40
		hi = hi + hii[i]
	end
end
test(M, V)

n = 100
M = zeros(n, n)
V = [rand(n, n) for _ in 1:40]
function test2(hi::Matrix{Float64}, hii::Vector{Matrix{Float64}})
	for i in 1:40
		hi .+= hii[i]
	end
end
test2(M, V)


using BenchmarkTools
srand(111)
function test2()
	n = 100
	M = zeros(n, n)
	V = [rand(n, n) for _ in 1:40]
	for i in 1:size(V, 1)
		M .+= V[i]
	end
end
@benchmark test2()



using StatsFuns: logistic
X = randn(1000, 2)
Y = X * [2, 3] .+ 1
p = logistic.(Y)
y = Float64.(rand(length(p)) .< p)





function inverse_link!{T <: Float64}(
    p :: Vector{T},
    xb :: Vector{T}
)
    @inbounds @simd for i in eachindex(p)
        p[i] = 1.0 / (1.0 + e^(-xb[i]))
    end
end

using StatsFuns: logistic
p = zeros(1000000)
xb = randn(1000000)
inverse_link!(p, xb)
p2 = logistic.(xb)

all(p2 .== p)

@benchmark inverse_link!(p, xb)
@benchmark logistic.(xb)


function test(x::Union{Vector{Int}, Vector{Float64}})
	return sum(x)
end






#load packages
using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
p = 100000
k = 10
s = 0.1
n = 1000

#construct snpmatrix, covariate files, and true model b
x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
z           = ones(n, 1)                   # non-genetic covariates, just the intercept
true_b      = zeros(p)                     # model vector
true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
shuffle!(true_b)                           # Shuffle the entries
correct_position = find(true_b)            # keep track of what the true entries are
noise = rand(Normal(0, s), n)              # noise

#compute mean and std used to standardize data to mean 0 variance 1
mean_vec, minor_allele, = summarize(x)
update_mean!(mean_vec, minor_allele, p)
std_vec = std_reciprocal(x, mean_vec)

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
y = y_temp + noise

#compute IHT result for less noisy data
v = IHTVariables(x, z, y, 1, k)
hi = @benchmark L0_reg(v, x, z, y, 1, k)

