# Note: none of the cross validation tests precise numbers. This is because
# random number generation can be different with Julia versions even if we 
# set the seed. AKA I don't know any good way of ensuring continuous integration

@testset "Cross validation on SnpArrays, normal model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Normal
    l = canonicallink(d())

    #set random seed
    Random.seed!(2019)

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
    z = ones(n) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0)
end

@testset "Cross validation on floating point matrices, normal model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Normal
    l = canonicallink(d())

    #set random seed
    Random.seed!(2019)

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
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, x, z, d=d(), l=l, path=path,
        q=q, folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0.0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0.0)
end

@testset "Cross validation on SnpArrays, logistic model" begin
    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true)
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#specify path and folds
    path = 1:20
	q = 3
	folds = rand(1:q, size(x, 1))

	# cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path,
        q=q, folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0.0)

	# cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0)
end

@testset "Cross validation on floating point matrices, logistic model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Bernoulli
    l = canonicallink(d())

    #set random seed
    Random.seed!(2019)

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
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0.0)

    # cross validation routine (without debias) 
    @time nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0)
end

@testset "Cross validation on SnpArrays, Poisson model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = Poisson
    l = canonicallink(d())

    #set random seed
    Random.seed!(2019)

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0.0)

    # cross validation routine (without debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0)
end

@testset "Cross validation on SnpArrays, NegativeBinomial model" begin
    #simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    k = 10
    d = NegativeBinomial
    l = LogLink()

    #set random seed
    Random.seed!(2019)

    #construct snpmatrix, covariate files, and true model b
    x = simulate_random_snparray(undef, n, p)
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(xla, k, d, l)

    #specify path and folds
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias) 
    @time debias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true);
    @test length(debias) == 20
    @test all(debias .> 0.0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, xla, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
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
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine (with debias)
	d = d(1, T(0.5)) # need Float32 for eltype of d
    @time debias = cv_iht(y, x, z, d=d, l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true)
    @test length(debias) == 20
    @test all(debias .> 0)

    # cross validation routine (no debias) 
    @time nodebias = cv_iht(y, x, z, d=d, l=l, path=path, q=q,
        folds=folds, verbose=true, debias=false, parallel=true);
    @test length(nodebias) == 20
    @test all(nodebias .> 0)
end


@testset "multivariate cross validation" begin
    n = 1000  # number of samples
    p = 10000 # number of SNPs
    k = 10    # number of causal SNPs
    r = 2     # number of traits
    
    # set random seed for reproducibility
    Random.seed!(2021)
    
    # simulate `.bed` file with no missing data
    x = simulate_random_snparray("multivariate_$(r)traits.bed", n, p)
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
    @time mses = cv_iht(Yt, Transpose(xla), debias=false)
    @test argmin(mses) == 11
    @test all(mses .≈ [2864.43080531955, 2813.227619074331, 2050.084384951764, 
        1795.1950114749766, 1541.68204069869, 1267.9884894035772, 1146.51230607942, 
        1107.7717755264625, 1009.8367577334823, 995.5031897807678, 988.5684961886975, 
        998.1110917264338, 996.66606804976, 1003.3836446432961, 1007.8201284531202, 
        1021.2948130528478, 1033.5460395022537, 1044.1422167031246, 1041.0469594998751, 
        1050.0933942926943])

    # yes debias
    Random.seed!(2021)
    @time mses2 = cv_iht(Yt, Transpose(xla), debias=true)
    @test argmin(mses2) == 12
    @test all(mses2 .≈ [2864.43080531955, 2430.4217597715106, 2036.4634134537098, 
        1771.114276707285, 1506.8785367062724, 1244.3674375402838, 1115.9288600600455, 
        1066.9409457176935, 998.5873538029537, 991.5027919453225, 972.7041581652586,
        963.5480511500693, 963.8560773184153, 970.7495944711505, 968.6228199959959, 
        966.2166147966545, 967.4356596323951, 965.9747572534242, 971.6611053573112, 
        975.9872277669035])
end
