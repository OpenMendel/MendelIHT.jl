# ml julia/1.7.2
# export JULIA_NUM_THREADS=16

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
using TraitSimulation
BLAS.set_num_threads(1)

global gemma_exe = "/scratch/users/bbchu/NFBC_sim/gemma"
global plink_exe = "/scratch/users/bbchu/NFBC_sim/plink"
global mvplink_exe = "/scratch/users/bbchu/NFBC_sim/plink.multivariate"

"""
    βi ~ N(0, 0.1) chosen randomly across genome

k = Number of causal SNPs
p = Total number of SNPs
traits = Number of traits (phenotypes)
overlap = number of causal SNPs shared in each trait
"""
# function simulate_random_beta(k::Int, p::Int, traits::Int; overlap::Int=0, βσ=1.0)
#     d = Normal(0, βσ)
#     true_b = zeros(p, traits)
#     if overlap == 0
#         causal_snps = sample(1:(traits * p), k, replace=false)
#         true_b[causal_snps] = rand(d, k)
#     else
#         shared_snps = sample(1:p, overlap, replace=false)
#         weight_vector = aweights(1 / (traits * (p - overlap)) * ones(traits * p))
#         for i in 1:traits
#             weight_vector[i*shared_snps] .= 0.0 # avoid sampling from shared snps
#         end
#         @assert sum(weight_vector) ≈ 1.0 "sum(weight_vector) = $(sum(weight_vector)) != 1.0"
#         # simulate β for shared predictors
#         for i in 1:traits
#             true_b[shared_snps, i] = rand(d, overlap)
#         end
#         # simulate β for none shared predictors
#         nonshared_snps = sample(1:(traits * p), weight_vector, k - (traits * overlap), replace=false)
#         true_b[nonshared_snps] = rand(d, k - (traits * overlap))
#     end

#     return true_b
# end

"""
    βi ~ Uniform([-0.5, -0.45, ..., -0.05, 0.05, ..., 0.5]) chosen uniformly across genome

k = Number of causal SNPs
p = Total number of SNPs
traits = Number of traits (phenotypes)
overlap = number of pleiotropic SNPs (affects each trait with effect size βi / r)
"""
function simulate_fixed_beta(k::Int, p::Int, traits::Int; overlap::Int=0)
    true_b = zeros(p, traits)
    effect_sizes = collect(0.05:0.05:0.5)
    # num_causal_snps = (k - (traits - 1) * overlap) :: Int
    # @assert num_causal_snps > 0 "number of causal SNPs should be positive but was $num_causal_snps"
    # idx_causal_snps = sample(1:p, num_causal_snps, replace=false)
    # @assert length(idx_causal_snps) == num_causal_snps "length(idx_causal_snps) = $(length(idx_causal_snps)) != num_causal_snps"
    # shuffle!(idx_causal_snps)
    k_indep = k - 2overlap # pleiotropic SNPs affect 2 phenotypes
    num_causal_snps = k_indep + overlap
    @assert num_causal_snps > 0 "number of causal SNPs should be positive but was $num_causal_snps"
    idx_causal_snps = sample(1:p, num_causal_snps, replace=false)
    @assert length(idx_causal_snps) == num_causal_snps "length(idx_causal_snps) = $(length(idx_causal_snps)) != num_causal_snps"
    shuffle!(idx_causal_snps)

    # pleiotropic SNPs affect 2 phenotypes
    for i in 1:overlap
        j = idx_causal_snps[i]
        rs = sample(1:traits, 2, replace=false)
        for r in rs
            true_b[j, r] = rand(-1:2:1) * effect_sizes[rand(1:10)]
        end
    end
    # non pleiotropic SNPs affect only 1 phenotype
    for i in (overlap+1):length(idx_causal_snps)
        idx = idx_causal_snps[i]
        true_b[idx, rand(1:traits)] = rand(-1:2:1) * effect_sizes[rand(1:10)]
    end

    @assert count(!iszero, true_b) == k "count(!iszero, true_b) = $(count(!iszero, true_b)) != k = $k"

    return true_b
end

"""
    every causal SNP affects every phenotype, chosen fixed distance with each other across genome
    Their total effect size sums up to 0.5

k = Number of causal SNPs (each SNP affects every trait)
p = Total number of SNPs
traits = Number of traits (phenotypes)
"""
# function simulate_pleiotropic_beta(k::Int, p::Int, traits::Int)
#     true_b = zeros(p, traits)
#     effect_size = 0.5 / traits
#     idx_causal_snps = collect(div(p, k):div(p, k):p)
#     shuffle!(idx_causal_snps)
    
#     # pleiotropic SNPs affect every phenotype
#     for i in idx_causal_snps
#         for r in 1:traits
#             true_b[i, r] = effect_size
#         end
#     end
    
#     @assert count(!iszero, true_b) == k*traits "count(!iszero, true_b) should be equal to $(k*traits)"

#     return true_b
# end

"""
# Arguments
xla = simulated genotype matrix (converted to a SnpLinAlg)
k = number of causal SNPs
r = number of traits
Φ = estimated GRM (using GEMMA)

# Optional arguments
seed = random seed for reproducibility
σ2 = contribution of GRM
σe = random environmental effect
βoverlap = number of causal SNPs shared in all traits
"""
# function simulate_multivariate_polygenic(
#     plinkname::String, n::Int, p::Int, k::Int, r::Int;
#     seed::Int=2021, σg=0.6, σe=0.4, βoverlap=2
#     )
#     # set seed
#     Random.seed!(seed)
    
#     # simulate `.bed` file with no missing data
#     x = simulate_random_snparray("sim$seed/" * plinkname * ".bed", n, p)
#     xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
#     make_bim_fam(xla, "sim$seed/" * plinkname)

#     # intercept is the only nongenetic covariate
#     Z = ones(n, 1)
#     intercepts = zeros(r)' # each trait have 0 intercept

#     # simulate β
#     # B = simulate_pleiotropic_beta(k, p, r)
#     B = simulate_fixed_beta(k, p, r, overlap=βoverlap)
#     writedlm("sim$(seed)/trueb.txt", B)

#     # between trait covariance matrix
#     Σ = random_covariance_matrix(r)

#     # between sample covariance is identity + GRM (2x because OpenMendel always uses half the GRM)
#     run(`./gemma -bfile sim$(seed)/$plinkname -gk 1 -o $plinkname`)
#     run(`mv ./output/$(plinkname).cXX.txt sim$(seed)`)
#     Φ = readdlm("sim$(seed)/" * plinkname * ".cXX.txt")
#     V = σg * Φ + σe * I

#     # simulate y using TraitSimulations.jl (https://github.com/OpenMendel/TraitSimulation.jl/blob/master/src/modelframework.jl#L137)
#     vc = @vc Σ ⊗ V
#     μ = zeros(n, r)
#     μ_null = zeros(n, r)
#     LinearAlgebra.mul!(μ_null, Z, intercepts)
#     mul!(μ, xla, B)
#     BLAS.axpby!(1.0, μ_null, 1.0, μ)
#     VCM_model = VCMTrait(Z, intercepts, xla, B, vc, μ)
#     Y = Matrix(Transpose(simulate(VCM_model)))

#     # simulate using Distributions.jl
#     # μ = z * intercepts + xla * B
#     # Y = rand(MatrixNormal(μ', Σ, V))
    
#     return xla, Matrix(Z'), B, Σ, Y
# end

# function simulate_multivariate_sparse(
#     plinkname::String, n::Int, p::Int, k::Int, r::Int;
#     seed::Int=2021, σg=0.6, σe=0.4, βoverlap=2
#     )
#     # set seed
#     Random.seed!(seed)
    
#     # simulate `.bed` file with no missing data
#     x = simulate_random_snparray("sim$seed/" * plinkname * ".bed", n, p)
#     xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
#     make_bim_fam(xla, "sim$seed/" * plinkname)

#     # intercept is the only nongenetic covariate
#     Z = ones(n, 1)
#     intercepts = zeros(r)' # each trait have 0 intercept

#     # simulate β
#     # B = simulate_pleiotropic_beta(k, p, r)
#     B = simulate_fixed_beta(k, p, r, overlap=βoverlap)
#     writedlm("sim$(seed)/trueb.txt", B)

#     # between trait covariance matrix (but GEMMA GRM still needed)
#     run(`./gemma -bfile sim$(seed)/$plinkname -gk 1 -o $plinkname`)
#     run(`mv ./output/$(plinkname).cXX.txt sim$(seed)`)
#     Σ = random_covariance_matrix(r)
#     writedlm("sim$(seed)/trueCovariance.txt", Σ)

#     # simulate multivariate normal phenotype for each sample
#     μ = xla * B + Z*intercepts

#     # simulate response
#     Y = zeros(n, r)
#     for i in 1:n
#         μi = @view(μ[i, :])
#         Y[i, :] = rand(MvNormal(μi, Σ))
#     end
    
#     return xla, Matrix(Z'), B, Σ, Matrix(Y')
# end

"""
simulate under IHT's model
"""
# function simulate_NFBC1966_sparse(
#     plinkname::String, k::Int, r::Int;
#     seed::Int=2021, σg=0.6, σe=0.4, βoverlap=2
#     )
#     # set seed
#     Random.seed!(seed)

#     # simulate `.bed` file with no missing data
#     x = SnpArray(plinkname * ".bed")
#     xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
#     n, p = size(xla)

#     # intercept is the only nongenetic covariate
#     Z = ones(n, 1)
#     intercepts = zeros(r)' # each trait have 0 intercept

#     # simulate β
#     # B = simulate_pleiotropic_beta(k, p, r)
#     B = simulate_fixed_beta(k, p, r, overlap=βoverlap)
#     writedlm("sim$(seed)/trueb.txt", B)

#     # between trait covariance matrix
#     Σ = random_covariance_matrix(r)
#     writedlm("sim$(seed)/trueCovariance.txt", Σ)

#     # simulate multivariate normal phenotype for each sample
#     μ = xla * B + Z*intercepts

#     # simulate response
#     Y = zeros(n, r)
#     for i in 1:n
#         μi = @view(μ[i, :])
#         Y[i, :] = rand(MvNormal(μi, Σ))
#     end
    
#     return xla, Matrix(Z'), B, Σ, Matrix(Y')
# end

"""
Trait covariance matrix is σg * Φ + σe * I where Φ is the GRM. 
"""
function simulate_NFBC1966_polygenic(
    plinkname::String, k::Int, r::Int, outdir::String;
    seed::Int=2021, σg=0.1, σe=0.9, βoverlap=2,
    )
    # set seed
    Random.seed!(seed)

    # simulate `.bed` file with no missing data
    x = SnpArray(plinkname * ".bed")
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
    n, p = size(x)

    # intercept is the only nongenetic covariate
    Z = ones(n, 1)
    intercepts = zeros(r)' # each trait have 0 intercept

    # simulate β
    # B = simulate_pleiotropic_beta(k, p, r)
    B = simulate_fixed_beta(k, p, r, overlap=βoverlap)
    writedlm(joinpath(outdir, "trueb.txt"), B)

    # between trait covariance matrix
    Σ = random_covariance_matrix(r)
    writedlm(joinpath(outdir, "true_cov.txt"), Σ)

    # between sample covariance is identity + GRM
    Φ = readdlm(plinkname * ".cXX.txt")
    V = σg * Φ + σe * I

    # simulate y using TraitSimulations.jl (https://github.com/OpenMendel/TraitSimulation.jl/blob/master/src/modelframework.jl#L137)
    vc = @vc Σ ⊗ V
    μ = zeros(n, r)
    μ_null = zeros(n, r)
    LinearAlgebra.mul!(μ_null, Z, intercepts)
    mul!(μ, xla, B)
    BLAS.axpby!(1.0, μ_null, 1.0, μ)
    VCM_model = VCMTrait(Z, intercepts, xla, B, vc, μ)
    Y = Matrix(Transpose(simulate(VCM_model)))
    writedlm(joinpath(outdir, "Y.txt"), Y)

    # simulate using Distributions.jl
    # μ = z * intercepts + xla * B
    # Y = rand(MatrixNormal(μ', Σ, V))
    
    return xla, Matrix(Z'), B, Σ, Y
end

function make_bim_fam(x::AbstractMatrix, name::String)
    n, p = size(x)

    #create .bim file structure: https://www.cog-genomics.org/plink2/formats#bim
    open(name * ".bim", "w") do f
        for i in 1:p
            write(f, "1\tsnp$i\t0\t$(100i)\t1\t2\n")
        end
    end
    
    open(name * ".fam", "w") do f
        for i in 1:n
            write(f, "$i\t1\t0\t0\t1\t1.0\n")
        end
    end
end

function make_GEMMA_fam_file(x::AbstractMatrix, y::AbstractVecOrMat, outfilename::String)
    ly = size(y, 1)
    n, p = size(x)

    # put 1st phenotypes in 6th column, 2nd phenotype in 7th column ... etc
    traits = size(y, 1)
    open(outfilename, "w") do f
        for i in 1:n
            write(f, "$i\t1\t0\t0\t1")
            for j in 1:traits
                write(f, "\t$(y[j, i])")
            end
            write(f, "\n")
        end
    end
end

function make_MVPLINK_fam_and_phen_file(x::AbstractMatrix, y::AbstractVecOrMat, name::String)
    ly = size(y, 1)
    n, p = size(x)

    # put a random phenotype in fam file
    traits = size(y, 1)
    open(name * ".fam", "w") do f
        for i in 1:n
            println(f, "$i\t1\t0\t0\t1\t1")
        end
    end

    # save phenotypes in separate `.phen` file
    open(name * ".phen", "w") do io
        print(io, "FID\tIID")
        for j in 1:traits
            print(io, "\tT$j")
        end
        print(io, "\n")
        for i in 1:n
            print(io, "$i\t1")
            for j in 1:traits
                write(io, "\t$(y[j, i])")
            end
            print(io, "\n")
        end
    end
end

# function make_MVPLINK_fam_and_phen_file(x::AbstractMatrix, y::AbstractVecOrMat, name::String)
#     ly = size(y, 1)
#     n, p = size(x)

#     # leave only first phenotype in fam file
#     # this makes it a valid plink file
#     # while the actual phenotypes will be read from .phen file
#     fam_df = CSV.read(name * ".fam", DataFrame, header=false)
#     CSV.write(name * ".fam", fam_df[!, 1:6], header=false)

#     # save phenotypes in separate `.phen` file
#     ytmp = Matrix{Any}(undef, n+1, 4)
#     ytmp[1, :] .= vcat(["FID", "IID"], ["T$i" for i in 1:traits])
#     ytmp[2:end, 1] .= 1:n
#     ytmp[2:end, 2] .= 1
#     ytmp[2:end, 3:4] .= y'
#     writedlm(name * ".phen", ytmp)
# end

"""
Computes power and false positive rates
- p: total number of SNPs
- pleiotropic_snps: Indices (or ID) of the true causal SNPs that affect >1 phenotype
- independent_snps: Indices (or ID) of the true causal SNPs that affect exactly 1 phenotype
- signif_snps: Indices (or ID) of SNPs that are significant after testing

returns: pleiotropic SNP's power, independent SNP's power, number of false positives, and false positive rate
"""
function power_and_fpr(p::Int, pleiotropic_snps, independent_snps, signif_snps)
    pleiotropic_power = length(signif_snps ∩ pleiotropic_snps) / length(pleiotropic_snps)
    independent_power = length(signif_snps ∩ independent_snps) / length(independent_snps)
    correct_snps = pleiotropic_snps ∪ independent_snps
    FP = length(signif_snps) - length(signif_snps ∩ correct_snps) # number of false positives
    TN = p - length(signif_snps) # number of true negatives
    FPR = FP / (FP + TN)
    return pleiotropic_power, independent_power, FP, FPR
end
    
# https://github.com/OpenMendel/MendelPlots.jl/blob/master/src/gwasplots.jl#L108
function genomic_inflation(pvalues::AbstractVecOrMat)
    λ = median(quantile.(Chisq(1), 1 .- pvalues) ./ quantile(Chisq(1), 0.5))
end

"""
- filename: gemma's output file name
- pleiotropic_snpid: ID of causal SNPs that affect > 1 phenotypes
- independent_snpid: ID of causal SNPs that affect 1 phenotypes

returns: power, number of false positives, and false positive rate
"""
function process_gemma_result(filename, pleiotropic_snpid, independent_snpid)
    # read GEMMA result
    gemma_df = CSV.read(filename, DataFrame)
    snps = size(gemma_df, 1)

    # pvalues
    pval_wald = gemma_df[!, :p_wald]
#    pval_lrt = gemma_df[!, :p_lrt]
#     pval_score = gemma_df[!, :p_score]

    # estimated beta
#     estim_β1 = gemma_df[!, :beta_1]
#     estim_β2 = gemma_df[!, :beta_2]

#     # estimated covariance matrix
#     estim_σ11 = gemma_df[!, :Vbeta_1_1]
#     estim_σ12 = gemma_df[!, :Vbeta_1_2]
#     estim_σ22 = gemma_df[!, :Vbeta_2_2];

    # check how many real SNPs were recovered
    signif_snps = findall(x -> x ≤ 0.05 / snps, pval_wald) # gemma's selected snps
    signif_snpid = gemma_df[signif_snps, :rs]

    # return power, false positives, false positive rate, and genomic inflation
    pleiotropic_power, independent_power, FP, FPR = power_and_fpr(snps,
        pleiotropic_snpid, independent_snpid, signif_snpid)
    λ = genomic_inflation(pval_wald)
    return pleiotropic_power, independent_power, FP, FPR, λ
end

"""
- filename: mvPLINK's output file name
- pleiotropic_snps: indices for causal SNPs that affect > 1 phenotypes
- independent_snps: indices for causal SNPs that affect 1 phenotypes

returns: power, number of false positives, and false positive rate
"""
function process_mvPLINK(filename, pleiotropic_snps, independent_snps)
    # read mvPLINK result
    mvplink_df = CSV.read(filename, DataFrame, delim=' ', ignorerepeated=true)
    snps = size(mvplink_df, 1)

    # get pvalues, possibly accounting for "NA"s
    if eltype(mvplink_df[!, :P]) == Float64
        pval = mvplink_df[!, :P]
    else
        mvplink_df[findall(x -> x == "NA", mvplink_df[!, :P]), :P] .= "1.0"
        pval = parse.(Float64, mvplink_df[!, :P])
    end

    # SNPs passing threshold
    signif_snps = findall(x -> x ≤ 0.05 / snps, pval)

    # compute power, false positives, and false positive rate
    pleiotropic_power, independent_power, FP, FPR = power_and_fpr(snps,
        pleiotropic_snps, independent_snps, signif_snps)
    λ = genomic_inflation(pval)
    return pleiotropic_power, independent_power, FP, FPR, λ
end

function one_NFBC_simulation(
    set::Int,
    ld::Float64,
    k::Int, 
    r::Int,
    run_gemma::Bool, run_mvPLINK::Bool, run_mIHT::Bool, run_uIHT::Bool;
    seed::Int=2021, σg=0.1, σe=0.9, βoverlap=2,
    path=5:5:50, init_beta=false, model=:polygenic, debias=false
    )
    cur_dir = pwd()
    plinkname = ld == 1 ? "NFBC.qc.imputeBy0.chr.1" : "NFBC.qc.imputeBy0.chr.1.LD$ld"
    simdir = joinpath(pwd(), "set$(set)_LD$(ld)", "sim$seed")
    isdir(simdir) ? (return nothing) : mkpath(simdir)
    plinkfile = "/scratch/users/bbchu/NFBC_sim/data/$plinkname"

    # simulate data
    if model == :polygenic
        xla, Z, B, Σ, Y = simulate_NFBC1966_polygenic(plinkfile, k, r, simdir,
            seed=seed, σg=σg, σe=σe, βoverlap=βoverlap)
        # copy data to simulation dir (mvPLINK needs another fam file, which will be created later)
        cp("$(plinkfile).bed", joinpath(simdir, "$(plinkname).bed"))
        cp("$(plinkfile).bim", joinpath(simdir, "$(plinkname).bim"))
        cp("$(plinkfile).cXX.txt", joinpath(simdir, "$(plinkname).cXX.txt"))
        make_GEMMA_fam_file(xla, Y, joinpath(simdir, "$(plinkname).fam"))
    else
        error("simulation model can only be :polygenic")
    end

    # record pleiotropic and independent SNPs
    correct_snps = unique([x[1] for x in findall(!iszero, B)])
    pleiotropic_snps, independent_snps = Int[], Int[]
    for snp in correct_snps
        count(x -> abs(x) > 0, @view(B[snp, :])) > 1 ? 
            push!(pleiotropic_snps, snp) : push!(independent_snps, snp)
    end
    snpdata = SnpData(plinkfile)
    pleiotropic_snp_rsid = snpdata.snp_info[pleiotropic_snps, :snpid]
    independent_snp_rsid = snpdata.snp_info[independent_snps, :snpid]

    # run GEMMA
    #     note: GEMMA needs its own plink file to read phenotypes from, 
    #     so we copy the data to the simulation directory 
    #     (extra files are deleted at the very end)
    if run_gemma
        pheno_columns = [string(ri) for ri in 1:r]
        gemma_time = @elapsed begin
            run(`$gemma_exe -bfile $(joinpath(simdir, plinkname)) 
                            -k $(joinpath(simdir, plinkname)).cXX.txt 
                            -notsnp 
                            -lmm 1 
                            -n $pheno_columns 
                            -o gemma.ld$(ld).set$(set).sim$(seed)`)
        end
        gemma_outdir = joinpath(pwd(), "output")
        gemma_pleiotropic_power, gemma_independent_power, gemma_FP, gemma_FPR, gemma_λ = 
            process_gemma_result("$gemma_outdir/gemma.ld$(ld).set$(set).sim$seed.assoc.txt", pleiotropic_snp_rsid, independent_snp_rsid)
        println("GEMMA time = $gemma_time, pleiotropic power = $gemma_pleiotropic_power, independent power = $gemma_independent_power, FP = $gemma_FP, FPR = $gemma_FPR, gemma_λ=$gemma_λ")
    end

    # run multivariate IHT
    if run_mIHT
        mIHT_time = @elapsed begin
            mses = cross_validate(joinpath(simdir, plinkname), MvNormal, path=path, phenotypes=collect(1:r).+5;
                init_beta=init_beta, debias=debias, cv_summaryfile=joinpath(simdir, "miht.cviht.summary1.txt"))
            k_rough_guess = path[argmin(mses)]
            dense_path = (k_rough_guess - 4):(k_rough_guess + 4)
            mses_new = cross_validate(joinpath(simdir, plinkname), MvNormal, path=dense_path, phenotypes=collect(1:r).+5;
                init_beta=init_beta, debias=debias, cv_summaryfile=joinpath(simdir, "miht.cviht.summary2.txt"))
            iht_result = iht(joinpath(simdir, plinkname), dense_path[argmin(mses_new)], MvNormal, phenotypes=collect(1:r).+5;
                init_beta=init_beta, debias=debias, 
                summaryfile=joinpath(simdir, "miht.summary.txt"),
                betafile = joinpath(simdir, "miht.beta.txt"),
                covariancefile = joinpath(simdir, "miht.cov.txt"))
        end
        detected_snps = Int[]
        for i in 1:r # save each beta separately
            β = iht_result.beta[i, :]
            detected_snps = detected_snps ∪ findall(!iszero, β)
            writedlm(joinpath(simdir, "multivariate_iht_beta$i.txt"), β)
        end
        mIHT_pleiotropic_power, mIHT_independent_power, mIHT_FP, mIHT_FPR = power_and_fpr(size(B, 1), pleiotropic_snps, independent_snps, detected_snps)
        println("multivariate IHT time = $mIHT_time, pleiotropic power = $mIHT_pleiotropic_power, independent power = $mIHT_independent_power, FP = $mIHT_FP, FPR = $mIHT_FPR")
    end

    # run multiple univariate IHT
    if run_uIHT
        detected_snps = Int[]
        uIHT_time = @elapsed begin
            for trait in 1:r
                mses = cross_validate(joinpath(simdir, plinkname), Normal, path=path, phenotypes=trait+5;
                    init_beta=init_beta, debias=debias, 
                    cv_summaryfile=joinpath(simdir, "uiht.cviht1.summary$trait.txt"))
                k_rough_guess = path[argmin(mses)]
                dense_path = (k_rough_guess == 5) ? (0:5) : ((k_rough_guess - 4):(k_rough_guess + 4))
                mses_new = cross_validate(joinpath(simdir, plinkname), Normal, path=dense_path, phenotypes=trait+5;
                    init_beta=init_beta, debias=debias, 
                    cv_summaryfile=joinpath(simdir, "uiht.cviht2.summary$trait.txt"))
                best_k = dense_path[argmin(mses_new)]
                if best_k > 0
                    iht_result = iht(joinpath(simdir, plinkname), best_k, Normal, phenotypes=trait+5;
                        init_beta=init_beta, debias=debias, 
                        summaryfile=joinpath(simdir, "uiht.summary$trait.txt"),
                        betafile = joinpath(simdir, "uiht.beta.txt"))
                    β = iht_result.beta
                else
                    β = zeros(size(B, 2))
                end

                # save results
                detected_snps = detected_snps ∪ findall(!iszero, β)
                writedlm(joinpath(simdir, "univariate_iht_beta$trait.txt"), β)
            end
        end
        uIHT_pleiotropic_power, uIHT_independent_power, uIHT_FP, uIHT_FPR = power_and_fpr(size(B, 1), pleiotropic_snps, independent_snps, detected_snps)
        println("univariate IHT time = $uIHT_time, pleiotropic power = $uIHT_pleiotropic_power, independent power = $uIHT_independent_power, FP = $uIHT_FP, FPR = $uIHT_FPR")    
    end

    # run mvPLINK
    #     mvPLINK always save output as "plink.mqfam.total" in the current directory
    #     Thus, we cd to the current dir, run analysis, then delete extra files, 
    #     before finally cd back
    if run_mvPLINK
        cd(simdir)
        make_MVPLINK_fam_and_phen_file(xla, Y, joinpath(simdir, plinkname))
        phenofile = joinpath(simdir, plinkname * ".phen")
        mvplink_time = @elapsed begin
            run(`$mvplink_exe --bfile $(joinpath(simdir, plinkname)) --noweb --mult-pheno $phenofile --mqfam`)
        end
        mvPLINK_pleitropic_power, mvPLINK_independent_power, mvPLINK_FP, mvPLINK_FPR, mvPLINK_λ = 
            process_mvPLINK("plink.mqfam.total", pleiotropic_snps, independent_snps)
        println("mvPLINK time = $mvplink_time, pleiotropic power = $mvPLINK_pleitropic_power, 
            independent power = $mvPLINK_independent_power, FP = $mvPLINK_FP, 
            FPR = $mvPLINK_FPR, mvPLINK_λ=$mvPLINK_λ")
        cd(cur_dir)
    end

    # clean up
    rm(joinpath(simdir, "$(plinkname).fam"), force=true)
    rm(joinpath(simdir, "$(plinkname).bed"), force=true)
    rm(joinpath(simdir, "$(plinkname).bim"), force=true)
    rm(joinpath(simdir, "$(plinkname).cXX.txt"), force=true)

    # save summary stats
    df = DataFrame(method = String[], time = Float64[], 
        plei_power = Float64[], indep_power = Float64[], 
        FP = Int[], FRP = Float64[], λ = Float64[]
    )
    if @isdefined mIHT_time
        push!(df, ["mIHT", mIHT_time, mIHT_pleiotropic_power, 
            mIHT_independent_power, mIHT_FP, mIHT_FPR, NaN])
    end
    if @isdefined uIHT_time
        push!(df, ["uIHT", uIHT_time, uIHT_pleiotropic_power, 
            uIHT_independent_power, uIHT_FP, uIHT_FPR, NaN])
    end
    if @isdefined mvplink_time 
        push!(df, ["mvPLINK", mvplink_time, mvPLINK_pleitropic_power, 
            mvPLINK_independent_power, mvPLINK_FP, mvPLINK_FPR, mvPLINK_λ])
    end
    if @isdefined gemma_time
        push!(df, ["gemma", gemma_time, gemma_pleiotropic_power, 
            gemma_independent_power, gemma_FP, gemma_FPR, gemma_λ])
    end
    CSV.write(joinpath(simdir, "summary.txt"), df)

    return nothing
end

#
# polygenic/independent beta = sign * Uniform{0.05, …, 0.25}
# max condition number = 10
# q = 5, iterates ≥5 times, init_beta=true, debias=false)
# sim set 1 are for k = 10, r = 2, βoverlap = 3 each affects 2 traits, path = 5:5:50 (then search around best k)
# sim set 2 are for k = 20, r = 3, βoverlap = 5 each affects 2 traits, path = 5:5:50 (then search around best k)
# sim set 3 are for k = 30, r = 5, βoverlap = 7 each affects 2 traits, path = 5:5:50 (then search around best k)
# sim set 4 are for k = 10, r = 10, βoverlap = 3 each affects 2 traits, path = 5:5:50 (then search around best k)
# sim set 5 are for k = 20, r = 50, βoverlap = 5 each affects 2 traits, path = 5:5:50 (then search around best k)
# sim set 6 are for k = 30, r = 100, βoverlap = 7 each affects 2 traits, path = 5:5:50 (then search around best k)
#

function run_simulation(set::Int, ld::Float64)
    model = :polygenic # model=:sparse doesn't work anymore 
    σg = 0.1
    σe = 0.9
    init_beta = true
    debias = false
    path = 5:5:50
    βoverlap = [3, 5, 7, 3, 5, 7]
    k = [10, 20, 30, 10, 20, 30]
    r = [2, 3, 5, 10, 50, 100]
    run_mIHT = true
    run_uIHT = true
    run_gemma = true
    run_mvPLINK = true

    # for testing
    # seed = 1
    # set = 1
    # ld = 0.25
    # k = k[set]
    # r = r[set]
    # βoverlap = βoverlap[set]

    println("Simulation model = $model, with LD threshold $ld, set $set has k = $(k[set]), r = $(r[set]), βoverlap = $(βoverlap[set])")

    set_dir = joinpath(pwd(), "set$(set)_LD$(ld)")
    isdir(set_dir) || mkpath(set_dir)
    k_cur = k[set]
    r_cur = r[set]
    βoverlap_cur = βoverlap[set]

    for seed in 1:100
        # don't run gemma/mvPLINK for certain simulation
        seed > 2 && set ≥ 5 && (run_gemma = false)
        seed > 2 && set ≥ 6 && (run_mvPLINK = false)

        try
            one_NFBC_simulation(set, ld, k_cur, r_cur, 
                run_gemma, run_mvPLINK, run_mIHT, run_uIHT,
                seed = seed, path = path, βoverlap=βoverlap_cur, 
                σg=σg, σe=σe, init_beta=init_beta, model=model, debias=debias
            )
        catch e
            bt = catch_backtrace()
            msg = sprint(showerror, e, bt)
            println("set $set sim $seed threw an error!")
            println(msg)
            continue
        end
        GC.gc();GC.gc();GC.gc()
    end
end

set = parse(Int, ARGS[1]) # Int between 1-6
ld = parse(Float64, ARGS[2]) # how much LD pruning, should be 0.25, 0.5, 0.75 or 1 (no pruning)
@show Threads.nthreads()
run_simulation(set, ld)
