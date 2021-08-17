# Note: none of the cross validation tests precise numbers. This is because
# random number generation can be different with Julia versions even if we 
# set the seed. AKA I don't know any good way of ensuring continuous integration

@testset "Cross validation on SnpLinAlg, normal model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Normal
    l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
    z = ones(n) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, max_iter=10);
    @test all(debias .> 0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0)
end

@testset "Cross validation on Float32 matrix, normal model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Normal
    l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    T = Float32
    x = randn(T, n, p)
    z = ones(T, n, 1)

    # simulate response, true model b, and the correct non-0 positions of b
    true_b = zeros(T, p)
    true_b[1:k] .= collect(0.1:0.1:1.0)
    true_c = [T.(4.0)]
    shuffle!(true_b)
    correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, x, z, d=d(), l=l, path=path,
        q=q, folds=folds, verbose=true, debias=true, max_iter=10);
    @test all(debias .> 0.0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0.0)
end

@testset "Cross validation on SnpLinAlg, logistic model" begin
    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path,
        q=q, folds=folds, verbose=true, debias=true, max_iter=10);
    @test all(debias .> 0.0)

	# cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0)
end

@testset "Cross validation on Float64 matrix, logistic model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Bernoulli
    l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    T = Float64
    x = randn(T, n, p)
    z = ones(T, n, 1)

    # simulate response, true model b, and the correct non-0 positions of b
    true_b = zeros(T, p)
    true_b[1:k] .= collect(0.1:0.1:1.0)
    true_c = [T.(4.0)]
    shuffle!(true_b)
    correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = T.([rand(d(i)) for i in prob])

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, max_iter=10);
    @test all(debias .> 0.0)

    # cross validation routine (without debias) 
    @time nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0)
end

@testset "Cross validation on SnpLinAlg, Poisson model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Poisson
    l = canonicallink(d())

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, max_iter=10);
    @test all(debias .> 0.0)

    # cross validation routine (without debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0)
end

@testset "Cross validation on SnpLinAlg, NegativeBinomial model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = NegativeBinomial
    l = LogLink()

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true);
    @test all(debias .> 0.0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false);
    @test all(nodebias .> 0)
end

@testset "Cross validation on floating point matrices, NegativeBinomial model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = NegativeBinomial
    l = LogLink()

    #set random seed
    Random.seed!(2019)

    #construct design matrix and covariates (intercept)
    T = Float32
    x = randn(T, n, p)
    z = ones(T, n, 1)

    # simulate response, true model b, and the correct non-0 positions of b
    true_b = zeros(T, p)
    true_b[1:k] .= collect(0.1:0.1:1.0)
    true_c = [T.(4.0)]
    shuffle!(true_b)
    correct_position = findall(!iszero, true_b)
    prob = GLM.linkinv.(l, x * true_b)
    clamp!(prob, -20, 20)
    y = [rand(d(i)) for i in prob]
    y = T.(y)

    #specify path and folds
    path = 0:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias)
	d = d(1, T(0.5)) # need Float32 for eltype of d
    @time debias = cv_iht(y, x, z, d=d, l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, max_iter=10)
    @test all(debias .> 0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, x, z, d=d, l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, max_iter=10);
    @test all(nodebias .> 0)
end

@testset "multivariate cross validation SnpLinAlg" begin
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs
    r = 2     # number of traits

    # simulate `.bed` file with no missing data
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 

    # intercept is the only nongenetic covariate
    z = ones(n, 1)
    intercepts = [10.0 1.0] # each trait have different intercept

    # simulate response y, true model b, and the correct non-0 positions of b
    Y, true_Σ, true_b, correct_position = simulate_random_response(xla, k, r, Zu=z*intercepts, overlap=2)
    correct_snps = [x[1] for x in correct_position] # causal snps
    Yt = Matrix(Y'); # in MendelIHT, multivariate traits should be rows

    @test_throws DimensionMismatch cv_iht(Yt, xla)
    @test_throws DimensionMismatch cv_iht(Y, Transpose(xla))

    # no debias
    Random.seed!(2021)
    @time mses = cv_iht(Yt, Transpose(xla), debias=false, max_iter=10, path=0:20)
    @test all(mses .> 0)

    # yes debias
    Random.seed!(2021)
    @time mses2 = cv_iht(Yt, Transpose(xla), debias=true, max_iter=10, path=0:20)
    @test all(mses2 .> 0)
end
