# BELOW ARE POISSON SIMULATIONS in 0.6
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic

#set random seed
srand(1111) 

#import snpmatrix, covariate files, and true model b
x_temp = readdlm("poisson_matrix", Int64)
y      = readdlm("poisson_response")
true_b = readdlm("poisson_true_model")
y = reshape(y, 1000)

# size of problem and true model size
n, p = size(x_temp)
k = countnz(true_b)

#construct our snp matrix
x = SnpArray(x_temp)
z = ones(n, 1)                   # non-genetic covariates, just the intercept

#compute mean and std_reci
mean_vec, minor_allele, = summarize(x)
for i in 1:p
    minor_allele[i] ? mean_vec[i] = 2.0 - 2.0mean_vec[i] : mean_vec[i] = 2.0mean_vec[i]
end
std_vec = std_reciprocal(x, mean_vec)

#compute poisson IHT result
v = IHTVariables(x, z, y, 1, k)
result = L0_poisson_reg(v, x, z, y, 1, k, glm = "poisson")

#check result
estimated_models = zeros(k)
correct_position = find(true_b)
estimated_models .= result.beta[correct_position]
true_model = true_b[correct_position]
compare_model = DataFrame(
    correct_position = correct_position, 
    true_β           = true_model, 
    estimated_β      = estimated_models)
println("Total iteration number was " * string(result.iter))
