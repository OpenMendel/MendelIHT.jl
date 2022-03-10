#
# Parameter explanations
# MvNormal: Distribution of phenotypes is multivariate normal
# q: number of cross validation folds
# init_beta: get good initial estimates for genetic and nongenetic regression coefficients
# min_iter: iterate at least 10 times before checking for convergence
# 
using MendelIHT, Random
Random.seed!(2021)                        # seed for reproducibility
plinkfile = "ukb.plink.filtered"          # plink files without .bed/bim/fam 
phenotypes = "normalized.bmi.sbp.dbp.txt" # comma-seprated phenotype file (no header line)
covariates = "iht.covariates.txt"         # comma-separated covariates file (no header line)

# cross validate k = 100, 200, ..., 1000
path = 100:100:1000
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3, 
    covariates=covariates, phenotypes=phenotypes, init_beta=true, min_iter=10,
    cv_summaryfile="cviht.summary.roughpath1.txt")

# if best k was 200 in previous step, now cross validate k = 110, 120, ..., 290
k_rough_guess = path[argmin(mses)]
path = (k_rough_guess - 90):10:(k_rough_guess + 90)
@time mses = cross_validate(plinkfile, MvNormal, path=path, q=3, 
    covariates=covariates, phenotypes=phenotypes, init_beta=true, min_iter=10,
    cv_summaryfile="cviht.summary.roughpath2.txt")

# if best k was 190 in previous step, now cross validate k = 181, 182, ..., 209
k_rough_guess = path[argmin(mses)]
dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
@time mses_new = cross_validate(plinkfile, MvNormal, path=dense_path,
    covariates=covariates, q=3, phenotypes=phenotypes, min_iter=10, init_beta=true)

# now run IHT on best k
@time iht_result = iht(plinkfile, dense_path[argmin(mses_new)], MvNormal,
    covariates=covariates, phenotypes=phenotypes, min_iter=10, init_beta=true)