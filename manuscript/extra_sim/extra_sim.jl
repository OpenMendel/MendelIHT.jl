using Revise
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
BLAS.set_num_threads(1)

function simulate_beta(r::Int, p::Int)
    k = [10^i for i in 1:r]
    d = Normal(0, 0.1)
    B = zeros(r, p)
    for i in 1:r
        k = 10^i
        B[i, 1:k] .= rand(d, k)
        shuffle!(@view(B[i, :]))
    end
    return B
end

# function simulate_genotypes(r::Int; seed::Int=2021)
#     # set seed
#     Random.seed!(seed)
#     n, p = 10000, 30000

#     # simulate `.bed` file with no missing data
#     x = simulate_random_snparray(undef, n, p)
#     xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)

#     # intercept is the only nongenetic covariate
#     Z = ones(n, 1)
#     intercepts = zeros(r)' # each trait have 0 intercept

#     # simulate β
#     B = simulate_beta(r, p)
#     writedlm("trueb.txt", B)

#     # between trait covariance matrix
#     Σ = random_covariance_matrix(r)
#     writedlm("true_cov.txt", Σ)

#     # between sample covariance is identity
#     V = Matrix(I, n, n)

#     # simulate using Distributions.jl
#     μ = Z * intercepts + xla * B'
#     Y = rand(MatrixNormal(μ', Σ, V))
#     standardize!(Transpose(Y))

#     return xla, Matrix(Z'), B, Σ, Y
# end

function simulate_genotypes(plinkfile::String, r::Int; seed::Int=2021)
    # set seed
    Random.seed!(seed)

    # simulate `.bed` file with no missing data
    x = SnpArray(plinkfile)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
    n, p = size(x)

    # intercept is the only nongenetic covariate
    Z = ones(n, 1)
    intercepts = zeros(r)' # each trait have 0 intercept

    # simulate β
    B = simulate_beta(r, p)
    writedlm("trueb.txt", B)

    # between trait covariance matrix
    Σ = random_covariance_matrix(r)
    writedlm("true_cov.txt", Σ)

    # between sample covariance is identity
    V = Matrix(I, n, n)

    # simulate using Distributions.jl
    μ = Z * intercepts + xla * B'
    Y = rand(MatrixNormal(μ', Σ, V))
    standardize!(Transpose(Y))

    return xla, Matrix(Z'), B, Σ, Y
end

"""
Simulate 3 phenotypes of different polygenecity
y1: 10 causal SNPs
y2: 100 causal SNPs
y3: 1000 causal SNPs
"""
function one_simulation(seed::Int)
    isdir("sim$seed") || mkdir("sim$seed")
    cd("sim$seed")
    r = 3 # number of phenotypes
    plinkname = "/scratch/users/bbchu/fastphase/10k_K20/ukb.10k.chr10.bed"

    # simulate data
    Random.seed!(seed)
    xla, Z, B, Σ, Y = simulate_genotypes(plinkname, r, seed=seed)
    GC.gc()

    # B[1, findall(!iszero, B[1, :])]
    # B[2, findall(!iszero, B[2, :])]
    # B[3, findall(!iszero, B[3, :])]

    # run multivariate IHT
    path=200:200:2000
    mIHT_time = @elapsed begin
        mses = cv_iht(Y, Transpose(xla), Z, path=path)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        path = (k_rough_guess - 150):50:(k_rough_guess + 150)
        mses = cv_iht(Y, Transpose(xla), Z, path=path)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        path = (k_rough_guess - 40):10:(k_rough_guess + 40)
        mses = cv_iht(Y, Transpose(xla), Z, path=path)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses = cv_iht(Y, Transpose(xla), Z, path=path)
        GC.gc()
        iht_result = fit_iht(Y, Transpose(xla), Z, k=path[argmin(mses)])
        writedlm("iht_beta.txt", iht_result.beta)
    end
    println("mIHT run time = $mIHT_time")

    # check power and FDR
    power, FDR = Float64[], Float64[]
    β1 = findall(!iszero, iht_result.beta[1, :])
    β2 = findall(!iszero, iht_result.beta[2, :])
    β3 = findall(!iszero, iht_result.beta[3, :])
    β1_true = findall(!iszero, B[1, :])
    β2_true = findall(!iszero, B[2, :])
    β3_true = findall(!iszero, B[3, :])
    # power
    push!(power, length(β1 ∩ β1_true) / 10)
    push!(power, length(β2 ∩ β2_true) / 100)
    push!(power, length(β3 ∩ β3_true) / 1000)
    # fdr
    push!(FDR, (count(!iszero, β1) - length(β1 ∩ β1_true)) / max(1, count(!iszero, β1)))
    push!(FDR, (count(!iszero, β2) - length(β2 ∩ β2_true)) / max(1, count(!iszero, β2)))
    push!(FDR, (count(!iszero, β3) - length(β3 ∩ β3_true)) / max(1, count(!iszero, β3)))
    # overall power and FDR
    correct_snps = unique([x[2] for x in findall(!iszero, B)])
    discovered_snps = unique([x[2] for x in findall(!iszero, iht_result.beta)])
    overall_power = length(correct_snps ∩ discovered_snps) / length(correct_snps)
    overall_FDR = (count(!iszero, discovered_snps) - length(discovered_snps ∩ correct_snps)) / 
        max(1, count(!iszero, discovered_snps))

    # save summary stats
    open("power_summary.txt", "w") do io
        println(io, "trait1_k10,trait2_k100,trait3_k1000,overall")
        println(io, power[1], ',', power[2], ',', power[3], ',', overall_power)
    end
    open("FDR_summary.txt", "w") do io
        println(io, "trait1_k10,trait2_k100,trait3_k1000,overall")
        println(io, FDR[1], ',', FDR[2], ',', FDR[3], ',', overall_FDR)
    end

    return nothing
end

seed = parse(Int, ARGS[1])
one_simulation(seed)
