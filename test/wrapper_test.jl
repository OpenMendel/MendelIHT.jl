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

@testset "wrapper univariate" begin
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs per trait
    for d in [Normal, Bernoulli, Poisson, NegativeBinomial]
        l = d == NegativeBinomial ? LogLink() : canonicallink(d())

        # set random seed for reproducibility
        Random.seed!(2021)

        # simulate `.bed` file with no missing data
        x = simulate_random_snparray("univariate$d.bed", n, p)
        xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 

        # intercept is the only nongenetic covariate
        z = ones(n)
        intercept = 1.0

        # simulate response y, true model b, and the correct non-0 positions of b
        y, true_b, correct_position = simulate_random_response(xla, k, d, l, Zu=z*intercept);

        # create covariate files, `.bim` and `.bam` files, and separate phenotype file
        writedlm("covariates$d.txt", z)
        make_bim_fam_files(x, y, "univariate$d")
        writedlm("univariate$d.phen", y, ',')

        result1 = iht("univariate$d", 11, d, verbose=false, summaryfile="iht.$(d)summary.txt",
            betafile="iht.$(d)beta.txt", max_iter=5)
        result2 = iht("univariate$d", 11, d, covariates="covariates$d.txt", verbose=false,
            summaryfile="iht.$(d)summary.txt", betafile="iht.$(d)beta.txt", max_iter=5)
        result3 = iht("univariate$d", 11, d, covariates="covariates$d.txt", phenotypes="univariate$d.phen",
            summaryfile="iht.$(d)summary.txt", betafile="iht.$(d)beta.txt", verbose=false, max_iter=5)

        @test all(result1.beta .≈ result2.beta .≈ result3.beta)
        @test result1.logl ≈ result2.logl ≈ result3.logl
        @test result1.iter == result2.iter == result3.iter
        @test result1.σg ≈ result2.σg ≈ result3.σg

        Random.seed!(2021)
        result1 = cross_validate("univariate$d", d, verbose=false, cv_summaryfile="cviht.$(d)summary.txt", max_iter=5)
        Random.seed!(2021)
        result2 = cross_validate("univariate$d", d, covariates="covariates$d.txt", verbose=false, max_iter=5)
        Random.seed!(2021)
        result3 = cross_validate("univariate$d", d, covariates="covariates$d.txt", 
            phenotypes="univariate$d.phen", verbose=false, max_iter=5)

        @test all(result1 .≈ result2 .≈ result3)

        try
            rm("univariate$d.bim", force=true)
            rm("univariate$d.bed", force=true)
            rm("univariate$d.fam", force=true)
            rm("univariate$d.phen", force=true)
            rm("covariates$d.txt", force=true)
            rm("cviht.$(d)summary.txt", force=true)
            rm("iht.$(d)summary.txt", force=true)
            rm("iht.$(d)beta.txt", force=true)
        catch
            println("Unsuccessful deletion of intermediate files generated in" * 
                " unit tests! Windows users can remove these files manually " * 
                "located at " * normpath(pathof(MendelIHT) * "../../../test"))
        end
    end
end

@testset "wrapper multivariate" begin
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs
    r = 2     # number of traits
    d = MvNormal

    # set random seed for reproducibility
    Random.seed!(111)

    # simulate `.bed` file with no missing data
    x = simulate_random_snparray("multivariate_$(r)traits.bed", n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 

    # intercept is the only nongenetic covariate
    z = ones(n, 1)
    intercepts = [0.5 1.0] # each trait have different intercept

    # simulate β
    B = simulate_random_beta(k, p, r, overlap=2)

    # between trait covariance matrix
    Σ = random_covariance_matrix(r)

    # between sample covariance is identity + GRM (2 times because in SnpArrays grm is halved)
    Φ = 2grm(x)
    σg = 0.6
    σe = 0.4
    V = σg * Φ + σe * I

    # simulate Y
    μ = z * intercepts + xla * B
    Yt = rand(MatrixNormal(μ', Σ, V))

    # create covariate files (each sample occupies a row)
    writedlm("covariates.txt", z)

    # create `.bim` and `.bam` files using phenotype
    make_bim_fam_files(x, Transpose(Yt), "multivariate_$(r)traits")

    # save phenotypes in separate file (each sample occupies a row)
    writedlm("multivariate_$(r)traits.phen", Yt', ',')

    result1 = iht("multivariate_$(r)traits", 17, d, phenotypes=[6, 7], verbose=false)
    result2 = iht("multivariate_$(r)traits", 17, d, phenotypes=[6, 7], covariates="covariates.txt", verbose=false)
    result3 = iht("multivariate_$(r)traits", 17, d, phenotypes="multivariate_$(r)traits.phen",
        covariates="covariates.txt", verbose=false);

    true_b1_idx = findall(!iszero, B[:, 1])
    true_b2_idx = findall(!iszero, B[:, 2])
    @test all(result1.beta[1, true_b1_idx] .≈ result2.beta[1, true_b1_idx] .≈ result3.beta[1, true_b1_idx])
    @test all(result1.beta[2, true_b2_idx] .≈ result2.beta[2, true_b2_idx] .≈ result3.beta[2, true_b2_idx])
    @test all(vec(result1.Σ) .≈ vec(result2.Σ) .≈ vec(result3.Σ))
    @test result1.logl ≈ result2.logl ≈ result3.logl
    @test result1.σg ≈ result2.σg ≈ result3.σg
    @test result1.iter == result2.iter == result3.iter

    Random.seed!(2020)
    result1 = cross_validate("multivariate_$(r)traits", d, phenotypes=[6, 7], verbose=false)
    Random.seed!(2020)
    result2 = cross_validate("multivariate_$(r)traits", d, phenotypes=[6, 7], covariates="covariates.txt", verbose=false)
    Random.seed!(2020)
    result3 = cross_validate("multivariate_$(r)traits", d, phenotypes="multivariate_$(r)traits.phen", 
        covariates="covariates.txt", verbose=false)

    @test all(result1 .≈ result2 .≈ result3)

    try
        rm("multivariate_$(r)traits.bim", force=true)
        rm("multivariate_$(r)traits.bed", force=true)
        rm("multivariate_$(r)traits.fam", force=true)
        rm("multivariate_$(r)traits.phen", force=true)
        rm("covariates.txt", force=true)
        rm("cviht.summary.txt", force=true)
        rm("iht.summary.txt", force=true)
        rm("iht.beta.txt", force=true)
    catch
        println("Unsuccessful deletion of intermediate files generated in" * 
            " unit tests! Windows users can remove these files manually " * 
            "located at " * normpath(pathof(MendelIHT) * "../../../test"))
    end
end
