#### This script simulates example data in the /data/ folder.
#### Phenotypes are stored in 6th column of the `.fam` files and also 
#### stored separatedly in `phenotypes.txt`. Non-genetic covariates include
#### an intercept and sex, which is stored separately in `covariates.txt`.
#### The true β is stored in `normal_true_beta.txt`.

using Revise
using MendelIHT
using SnpArrays
using Random
using GLM
using DelimitedFiles

################################
######## Gaussian data #########
################################
n = 1000            # number of samples
p = 10000           # number of SNPs
k = 10              # 8 causal SNPs and 2 causal covariates (intercept + sex)
d = Normal          # Gaussian (continuous) phenotypes
l = IdentityLink()  # canonical link function

# set random seed
Random.seed!(1111)

# simulate `.bed` file with no missing data
x = simulate_random_snparray("normal.bed", n, p)
xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 

# nongenetic covariate: first column is the intercept, second column is sex: 0 = male 1 = female
z = ones(n, 2) 
z[:, 2] .= rand(0:1, n)
writedlm("covariates.txt", z, ',') # save covariates
standardize!(@view(z[:, 2:end])) # standardize covariates

# randomly set genetic predictors where causal βᵢ ~ N(0, 1)
true_b = zeros(p) 
true_b[1:k-2] = randn(k-2)
shuffle!(true_b)

# find correct position of genetic predictors
correct_position = findall(!iszero, true_b)

# save true SNP's position and effect size
open("normal_true_beta.txt", "w") do io
    println(io, "snpID,effectsize")
    for pos in correct_position
        println(io, "snp$pos,", true_b[pos])
    end
end

# define effect size of non-genetic predictors: intercept & sex
true_c = [1.0; 1.5] 

# simulate phenotype using genetic and nongenetic predictors
# note for d = Normal, l = IdentityLink(), `prob = xla * true_b .+ z * true_c`
prob = GLM.linkinv.(l, xla * true_b .+ z * true_c)
y = [rand(d(i)) for i in prob]
y = Float64.(y); # turn y into floating point numbers
writedlm("phenotypes.txt", y) # save phenotypes

# create `.bim` and `.bam` files using phenotype
make_bim_fam_files(x, y, "normal")






####### IHT TEST RUNS #########

# without sex as covariate (but intercept automatically included)
result = iht("normal", 9) 
[true_b[correct_position] result.beta[correct_position]] # compare IHT's result with answer

# include sex as covariate
result = iht("normal", "covariates.txt", 10)
[true_b[correct_position] result.beta[correct_position]] # compare IHT's result with answer

# phenotypes inputted as separate file
result = iht("phenotypes.txt", "normal", "covariates.txt", 10)
[true_b[correct_position] result.beta[correct_position]] # compare IHT's result with answer





##### CROSS VALIDATION TESTS ######

# without sex as covariate (but intercept automatically included)
mses = cross_validate("normal", 1:20) 
argmin(mses)

# include sex as covariate
mses = cross_validate("normal", "covariates.txt", 1:20)
argmin(mses)

# phenotypes inputted as separate file
mses = cross_validate("phenotypes.txt", "normal", "covariates.txt", 1:20)
argmin(mses)
