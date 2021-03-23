using Distributed
addprocs(4)

@everywhere begin
    using MendelIHT
    using SnpArrays
    using Random
    using GLM
    using DelimitedFiles
    using Distributions
    using LinearAlgebra
    using CSV
    using DataFrames
    using StatsBase
end

"""
k = Number of causal SNPs
p = Total number of SNPs
traits = Number of traits (phenotypes)
overlap = number of causal SNPs shared in each trait
"""
function simulate_random_beta(k::Int, p::Int, traits::Int; overlap::Int=0)
    true_b = zeros(p, traits)
    if overlap == 0
        causal_snps = sample(1:(traits * p), k, replace=false)
        true_b[causal_snps] = randn(k)
    else
        shared_snps = sample(1:p, overlap, replace=false)
        weight_vector = aweights(1 / (traits * (p - overlap)) * ones(traits * p))
        for i in 1:traits
            weight_vector[i*shared_snps] .= 0.0 # avoid sampling from shared snps
        end
        @assert sum(weight_vector) ≈ 1.0
        # simulate β for shared predictors
        for i in 1:traits
            true_b[shared_snps, i] = randn(overlap)
        end
        # simulate β for none shared predictors
        nonshared_snps = sample(1:(traits * p), weight_vector, k - traits * overlap, replace=false)
        true_b[nonshared_snps] = randn(k - traits * overlap)
    end

    return true_b
end

"""
# Arguments
plinkname = NFBC file PLINK name (excluding .bim/.bed/.fam suffix)
k = number of causal SNPs
r = number of traits

# Optional arguments
seed = random seed for reproducibility
σ2 = contribution of GRM
σe = random environmental effect
βoverlap = number of causal SNPs shared in all traits
"""
function simulate_multivariate_polygenic(
    plinkname::String, k::Int, r::Int;
    seed::Int=2021, σg=0.6, σe=0.4, βoverlap=2, 
    )
    
    # set seed
    Random.seed!(seed)

    # simulate `.bed` file with no missing data
    x = SnpArray(plinkname * ".bed")
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
    n, p = size(x)
    
    # intercept is the only nongenetic covariate
    Z = ones(n, 1)
    intercepts = randn(r)' # each trait have different intercept

    # simulate β
    B = simulate_random_beta(k, p, r, overlap=βoverlap)

    # between trait covariance matrix
    Σ = random_covariance_matrix(r)

    # between sample covariance is identity + centered GRM (estimated by GEMMA)
    Φ = readdlm(plinkname * ".cXX.txt")
    V = σg * Φ + σe * I

    # simulate y using TraitSimulations.jl
    # VCM_model = VCMTrait(Z, intercepts, x, B, [Σ], [V]) #https://github.com/OpenMendel/TraitSimulation.jl/blob/6d1f09c7332471a74b4dd6c8ef2d2b95a96c585c/src/modelframework.jl#L159
    # Y = simulate(VCM_model)

    # simulate using naive model
    μ = Z * intercepts + xla * B
    Y = rand(MatrixNormal(μ', Σ, V))
    
    return xla, Matrix(Z'), B, Σ, Y
end

function make_bim_file(x::SnpLinAlg, y::AbstractVecOrMat, name::String)
    ly = size(y, 1)
    n, p = size(x)

    #create .bim file structure: https://www.cog-genomics.org/plink2/formats#bim
    open(name * ".bim", "w") do f
        for i in 1:p
            write(f, "1\tsnp$i\t0\t1\t1\t2\n")
        end
    end
end

function make_GEMMA_fam_file(x::SnpLinAlg, y::AbstractVecOrMat, name::String)
    ly = size(y, 1)
    n, p = size(x)

    # put 1st phenotypes in 6th column, 2nd phenotype in 7th column ... etc
    traits = size(y, 2)
    open(name * ".fam", "w") do f
        for i in 1:n
            write(f, "$i\t1\t0\t0\t1")
            for j in 1:traits
                write(f, "\t$(y[i, j])")
            end
            write(f, "\n")
        end
    end
end

function make_MVPLINK_fam_and_phen_file(x::SnpLinAlg, y::AbstractVecOrMat, name::String)
    ly = size(y, 1)
    n, p = size(x)

    # only put 1 phenotype in fam file
    traits = size(y, 2)
    open(name * ".fam", "w") do f
        for i in 1:n
            write(f, "$i\t1\t0\t0\t1")
            for j in 1:traits
                write(f, "\t$(y[i, j])")
            end
            write(f, "\n")
        end
    end

    # save phenotypes are saved in separate `.phen` file
    open(name * ".phen", "w") do io
        println(io, "FID\tIID\tT1\tT2")
        for i in 1:n
            println(io, "$i\t1\t", Yt[1, i], "\t", Yt[2, i])
        end
    end
end

"""
Computes power and false positive rates
- p: total number of SNPs
- correct_snps: Indices of the true causal SNPs
- detected_snps: Indices of SNPs that are significant after testing

returns: power, number of false positives, and false positive rate
"""
function power_and_fpr(p::Int, correct_snps::Vector{Int}, signif_snps::Vector{Int})
    power = length(signif_snps ∩ correct_snps) / length(correct_snps)
    FP = length(signif_snps) - length(signif_snps ∩ correct_snps) # number of false positives
    TN = p - length(signif_snps) # number of true negatives
    FPR = FP / (FP + TN)
    return power, FP, FPR
end

"""
- filename: gemma's output file name
- correct_snps: indices for real causal SNPs

returns: power, number of false positives, and false positive rate
"""
function process_gemma_result(filename, correct_snps)
    # read GEMMA result
    gemma_df = CSV.read(filename, DataFrame)
    snps = size(gemma_df, 1)

    # pvalues
    pval_wald = gemma_df[!, :p_wald]
#     pval_lrt = gemma_df[!, :p_lrt]
#     pval_score = gemma_df[!, :p_score]

    # estimated beta
    estim_β1 = gemma_df[!, :beta_1]
    estim_β2 = gemma_df[!, :beta_2]

    # estimated covariance matrix
    estim_σ11 = gemma_df[!, :Vbeta_1_1]
    estim_σ12 = gemma_df[!, :Vbeta_1_2]
    estim_σ22 = gemma_df[!, :Vbeta_2_2];

    # check how many real SNPs were recovered
    signif_snps = findall(x -> x ≤ 0.05 / snps, pval_wald) # gemma's selected snps
    @show signif_snps ∩ correct_snps

    # compute power, false positives, and false positive rate
    power_and_fpr(snps, correct_snps, signif_snps)
end

"""
- filename: mvPLINK's output file name
- correct_snps: indices for real causal SNPs

returns: power, number of false positives, and false positive rate
"""
function process_mvPLINK(filename, correct_snps)
    # read mvPLINK result
    mvplink_df = CSV.read(filename, DataFrame, delim=' ', ignorerepeated=true)
    snps = size(mvplink_df, 1)

    # pvalues
    pval = mvplink_df[!, :P]

    # SNPs passing threshold
    signif_snps = findall(x -> x ≤ 0.05 / snps, pval)
    @show signif_snps ∩ correct_snps

    # compute power, false positives, and false positive rate
    power_and_fpr(snps, correct_snps, signif_snps)
end

"""
# Arguments
n = number of samples
p = number of SNPs
k = number of causal SNPs
r = number of traits

# Optional arguments
seed = random seed for reproducibility
σ2 = contribution of GRM
σe = random environmental effect
βoverlap = number of causal SNPs shared in all traits
"""
function one_simulation(
    k::Int, r::Int;
    seed::Int=2021, σg=0.6, σe=0.4, βoverlap=2
    )
    plinkname = "multivariate_$(r)traits" # test on local
#     plinkname = "NFBC_imputed_with_0"   # run this on Hoffman
    isdir("sim$seed") || mkdir("sim$seed")

    # simulate data
    xla, Z, B, Σ, Y = simulate_multivariate_polygenic(plinkname, k, r,
        seed=seed, σg=σg, σe=σe, βoverlap=βoverlap)
    correct_position = findall(!iszero, B)
    correct_snps = [x[1] for x in correct_position]
    writedlm("sim$(seed)/trueb.txt", B)

    # run IHT
#     Random.seed!(seed)
#     mses = cv_iht(Y, Transpose(xla), Z, parallel=true)
#     iht_result = fit_iht(Y, Transpose(xla), Z, k=argmin(mses))
#     β1, β2 = iht_result.beta[1, :], iht_result.beta[2, :]
#     detected_snps = findall(!iszero, β1) ∪ findall(!iszero, β2)
#     iht_power, iht_FP, iht_FPR = power_and_fpr(size(B, 1), correct_snps, detected_snps)
#     writedlm("sim$(seed)/iht_beta1.txt", β1)
#     writedlm("sim$(seed)/iht_beta2.txt", β2)
#     println("IHT power = $iht_power, FP = $iht_FP, FPR = $iht_FPR")

    # run MVPLINK
    phenofile = plinkname * ".phen"
    outfile = "sim$(seed)/mvPLINK_sim$seed.mqfam.total"
    make_MVPLINK_fam_and_phen_file(xla, Y, plinkname)
    run(`plink.multivariate --bfile $plinkname --noweb --mult-pheno $phenofile --mqfam`)
    mv("plink.mqfam.total", outfile, force=true)
    mvPLINK_power, mvPLINK_FP, mvPLINK_FPR = process_mvPLINK(outfile, correct_snps)
    println("mvPLINK power = $mvPLINK_power, FP = $mvPLINK_FP, FPR = $mvPLINK_FPR")

#     # run GEMMA
#     run(`gemma -bfile $plinkname -gk 1 -o gemma.grm`)
#     run(`gemma -bfile $plinkname -k output/gemma.grm.cXX.txt -maf 0.0000001 -lmm 1 -n 1 2 -o gemma.sim$seed`)
#     gemma_power, gemma_FP, gemma_FPR = process_gemma_result("output/gemma.sim$seed.assoc.txt", correct_snps)
        
    return iht_power, iht_FP, iht_FPR
end

# function NFBC1966_simulations(repeats::Int)
#     iht_power, iht_FP, iht_FPR = zeros(repeats), zeros(repeats), zeros(repeats)
#     gemma_power, gemma_FP, gemma_FPR = zeros(repeats), zeros(repeats), zeros(repeats)
#     mvPLINK_power, mvPLINK_FP, mvPLINK_FPR = zeros(repeats), zeros(repeats), zeros(repeats)
    
#     # run simulations
#     for i in 1:repeats
#         ihtp, ihtfp, ihtfpr = one_simulation()
#         iht_power[i], iht_FP[i], iht_FPR[i] = 
#     end
# end

k = 10
r = 2
seed = 2021
one_simulation(k, r, seed = seed)
