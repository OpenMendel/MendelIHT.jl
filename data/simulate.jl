#### This script simulates example data in the /data/ folder.
using Revise
using MendelIHT
using SnpArrays
using Random
using GLM

################################
######## Gaussian data #########
################################
n = 1000            # number of samples
p = 10000           # number of SNPs
k = 10              # number of causal SNPs
d = Normal          # Gaussian (continuous) phenotypes
l = IdentityLink()  # canonical link function

# set random seed
Random.seed!(1111)

# simulate `.bed` file with no missing data
x = simulate_random_snparray("normal.bed", n, p)
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 

# simulate response with βⱼ ~ N(0, 1) for 10 j's
y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

# create `.bim` and `.bam` files using phenotype
make_bim_fam_files(x, y, "normal")

# save true SNP's position and effect size
open("normal_true_beta.txt", "w") do io
    println(io, "snpID,effectsize")
    for pos in correct_position
        println(io, "snp$pos,", true_b[pos])
    end
end

# test run
result = iht("normal", 10)
