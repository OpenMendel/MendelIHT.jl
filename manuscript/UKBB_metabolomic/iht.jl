##
## Phenotype file (not standardized, with header and family-ID): phenotypes.reordered.csv 
## Final phenotype file (standardized, without header/family-ID): phenotypes.reordered.standardized.csv
## Final covariates file (with family ID, not standardized): covariates.reordered.csv
## Final covariates file (without family ID, standardized, extra intercept column): covariates.reordered.standardized.csv
## Final genotype file: ukb.merged.metabolic.subset.european.400K.QC.bed
##

#
# mIHT test run (K = 5000, no cross validation)
#
using DelimitedFiles, MendelIHT, SnpArrays, LinearAlgebra
BLAS.set_num_threads(1)
y = readdlm("phenotypes.reordered.standardized.csv", ',')
z = readdlm("covariates.reordered.standardized.csv", ',')
standardize!(@view(z[:, 2:end]))
x = SnpArray("ukb.merged.metabolic.subset.european.400K.QC.bed")
xla = SnpLinAlg{Float64}(x, center=true, scale=true, impute=true)
Yt = Matrix(Transpose(y))
Zt = Matrix(Transpose(z))
fit_iht(Yt, Transpose(xla), Zt; verbose=true,
    k=5000, d = MvNormal(), max_iter=10, max_step=3, init_beta=false)

#
# running mIHT
#
using MendelIHT, Random, LinearAlgebra
BLAS.set_num_threads(1)
Random.seed!(2022)
plinkfile = "ukb.merged.metabolic.subset.european.400K.QC"
phenotypes = "phenotypes.reordered.standardized.csv"
covariates = "covariates.reordered.standardized.csv"

path = 1000:1000:10000
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3,
    covariates=covariates, phenotypes=phenotypes, min_iter=10,
    cv_summaryfile="cviht.summary.roughpath1.txt")

k_rough_guess = path[argmin(mses)]
path = (k_rough_guess - 900):100:(k_rough_guess + 900)
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3,
    covariates=covariates, phenotypes=phenotypes, min_iter=10,
    cv_summaryfile="cviht.summary.roughpath2.txt")

k_rough_guess = path[argmin(mses)]
path = (k_rough_guess - 90):10:(k_rough_guess + 90)
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3,
    covariates=covariates, phenotypes=phenotypes, min_iter=10,
    cv_summaryfile="cviht.summary.roughpath3.txt")

k_rough_guess = path[argmin(mses)]
path = (k_rough_guess - 9):(k_rough_guess + 9)
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3,
    covariates=covariates, phenotypes=phenotypes, min_iter=10,
    cv_summaryfile="cviht.summary.final.txt")

K = path[argmin(mses)]
@time iht_result = iht(plinkfile, K, MvNormal, 
    summaryfile = "iht.final.summary.txt",
    betafile = "iht.final.beta.txt",
    covariancefile = "iht.final.cov.txt",
    covariates=covariates, phenotypes=phenotypes, max_iter=2000)
