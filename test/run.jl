#BELOW BENCHMARK PERFORMANCE OF L0_reg 
using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#simulat data
n = 2000
p = 10000
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 5              # number of true predictors
s = 0.1            # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#compute IHT result for less noisy data
k = 3
v = IHTVariables(x, z, y, 1, k)
result = L0_reg(v, x, z, y, 1, k)

@benchmark L0_reg(v, x, z, y, 1, k) seconds = 30






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

#simulat data
n = 2000
p = 10000
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 10              # number of true predictors
s = 0.1            # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#compute IHT result for less noisy data
v = IHTVariables(x, z, y, 1, k)
result = L0_reg(v, x, z, y, 1, k)

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
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

#set random seed
Random.seed!(1111)

#simulat data
n = 6000
p = 5000
k = 100 # number of true predictors
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes (e.g. vector y) via: y = Xb
y_temp = xbm * true_b

# Apply inverse logit link and sample from the vector of distributions
y = zeros(n)
prob = logistic.(y_temp) #inverse log link
for i in 1:n
    dist = Bernoulli(prob[i])
    y[i] = rand(dist)
end

#compute logistic IHT result
v = IHTVariables(x, z, y, 1, k)
result = L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic")

# @benchmark L0_logistic_reg(v, x, z, y, 1, k, glm = "logistic") seconds = 30

#check result
estimated_models = zeros(k)
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
using BenchmarkTools
using Random
using LinearAlgebra
using StatsFuns: logistic

#set random seed
Random.seed!(1111)

#simulat data
n = 1000
p = 10000
k = 10 # number of true predictors
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes (e.g. vector y) via: y = Xb
y_temp = xbm * true_b

# Apply inverse log link
y = zeros(n)
y_temp = exp.(y_temp)                  #inverse log link
for i in 1:n
	dist = Poisson(y_temp[i])
	y[i] = rand(dist)
end

#compute poisson IHT result
v = IHTVariables(x, z, y, 1, k)
result = L0_poisson_reg(v, x, z, y, 1, k, glm = "poisson")

#check result
estimated_models = zeros(k)
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







############## NORMAL CROSS VALIDATION SIMULATION
using IHT
using SnpArrays
using DataFrames
using Distributions
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#simulat data
n = 2000
p = 10000
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 5    # number of true predictors
s = 0.1  # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are
noise = rand(Normal(0, s), n)                   # noise vectors from N(0, s) 

#simulate phenotypes (e.g. vector y) via: y = Xb + noise
y = xbm * true_b + noise

#specify path and folds
path = collect(1:10)
num_folds = 3
folds = rand(1:num_folds, size(x, 1))

#compute cross validation
result = cv_iht(x, z, y, 1, path, folds, num_folds, use_maf = false, glm = "normal")





########### LOGISTIC CROSS VALIDATION SIMULATION CODE##############
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools
using Random
using LinearAlgebra

#set random seed
Random.seed!(1111)

#simulat data
n = 2000
p = 10000
x = simulate_random_snparray(n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#specify true model size and noise of data
k = 5    # number of true predictors
s = 0.1  # noise vector

#construct covariates (intercept) and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes under different noises by: y = Xb + noise
y_temp = zeros(n)
mul!(y_temp, xbm, true_b)

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

z = zeros(1000)
zz = zeros(1000)
y = rand(1000)
p = rand(1000)
@benchmark z .= y - p
@benchmark zz .= y .- p





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




using LinearAlgebra
using BenchmarkTools
function old_logl(x :: Vector{Float64}, y :: Vector{Float64})
	return dot(x, y) - sum(log.(1.0 .+ exp.(y))) 
end
function new_logl(x :: Vector{Float64}, y :: Vector{Float64})
	logl = 0.0
	for i in eachindex(x)
		logl += x[i]*y[i] - log(1.0 + exp(y[i]))
	end
	return logl
end
x = rand(1000)
y = rand(1000)
old_logl(x, y) ≈ new_logl(x, y)
@benchmark old_logl(x, y) #median = 33.845 μs
@benchmark new_logl(x, y) #median = 34.986 μs



using LinearAlgebra, SnpArrays, BenchmarkTools
x = SnpArray(undef, 10000, 10000)
xbm = SnpBitMatrix{Float64}(x, center=true, scale=true)
Base.summarysize(x)   # 25640152
Base.summarysize(xbm) # 25320360
z = zeros(10000)
y = rand(10000)
@benchmark mul!(z, xbm, y) # 187.767 ms

hi = @view x[1:1000, 1:1000]
hibm = SnpBitMatrix{Float64}(hi) #this should work
hibm = SnpBitMatrix{Float64}(hi, center=true) 

x_mask = @view x[1:9999, 1:9999]
xbm_mask = SnpBitMatrix{Float64}(x_mask, center=true, scale=true)
Base.summarysize(x_mask)   # 25640152
Base.summarysize(xbm_mask) # 25320360
z = zeros(9999)
y = rand(9999)
@benchmark mul!(z, x_mask, y) # 187.767 ms



using LinearAlgebra, SnpArrays, BenchmarkTools
x = SnpArray(undef, 10000, 10000)
x_subset = x[1:1000, 1:1000]
x_subsetbm = SnpBitMatrix{Float64}(x_subset, center=true, scale=true)



xbm = SnpBitMatrix{Float64}(x, center=true, scale=true);



