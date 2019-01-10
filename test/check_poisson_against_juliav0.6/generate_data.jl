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
k = 10 

#simulate random snp matrix without missing value
x_tmp = rand(0:2, n, p)
x = SnpArray("poisson.bed", n, p)
for i in 1:(n*p)
    if x_tmp[i] == 0
        x[i] = 0x00
    elseif x_tmp[i] == 1
        x[i] = 0x02
    else
        x[i] = 0x03
    end
end
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

#simulate non-genetic covariates and true model b
z = ones(n, 1)          # non-genetic covariates, just the intercept
true_b = zeros(p)       # model vector
true_b[1:k] = randn(k)  # Initialize k non-zero entries in the true model
shuffle!(true_b)        # Shuffle the entries
correct_position = findall(x -> x != 0, true_b) # keep track of what the true entries are

#simulate phenotypes (e.g. vector y) via: y = Xb
y_temp = xbm * true_b

# Apply inverse link
y = zeros(n)
y_temp = exp.(y_temp)                  #inverse log link
for i in 1:n
	dist = Poisson(y_temp[i])
	y[i] = rand(dist)
end

# write y, b to file
writedlm("poisson_matrix", x_tmp)
writedlm("poisson_response", y)
writedlm("poisson_true_model", true_b)

#to read in the data, use:
# x = readlm("poisson_matrix", Int64)
# y = readdlm("poisson_response", Int64)
# true_b = readdlm("poisson_true_model")

