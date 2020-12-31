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

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=true, debias=true, parallel=true);
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test all(distribute_fold_debias .≈ distribute_path_debias)

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_fold_nodebias) == 20
    @test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
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

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path,
        q=q, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q, 
        folds=folds, verbose=false, debias=false, parallel=true)
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l, 
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_fold_debias) == 20
    @test all(isapprox.(distribute_path_debias, distribute_fold_debias, atol=1e-3))

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_fold_nodebias) == 20
    @test all(isapprox.(distribute_path_nodebias, distribute_fold_nodebias, atol=1e-3))
end

# This test times out (> 10 min) on travis with linux machines. Not sure why
# @testset "Cross validation on SnpArrays, logistic model" begin
# 	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

#     #simulat data with k true predictors, from distribution d and with link l.
# 	n = 1000
# 	p = 10000
# 	k = 10
# 	d = Bernoulli
# 	l = canonicallink(d())

# 	#set random seed
# 	Random.seed!(2019)

# 	#construct snpmatrix, covariate files, and true model b
# 	x = simulate_random_snparray(undef, n, p)
# 	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
# 	z = ones(n, 1) # the intercept

# 	# simulate response, true model b, and the correct non-0 positions of b
# 	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

# 	#specify path and folds
#     path = 1:20
# 	q = 3
# 	folds = rand(1:q, size(x, 1))

# 	# cross validation routine that distributes `path` (with debias) 
#     @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path,
#         q=q, folds=folds, verbose=false, debias=true, parallel=true);
#     @test length(distribute_path_debias) == 20
#     @test all(distribute_path_debias .> 0.0)

# 	# cross validation routine that distributes `path` (no debias) 
#     @time distribute_path_nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
#         folds=folds, verbose=false, debias=false, parallel=true);
#     @test length(distribute_path_nodebias) == 20
#     @test all(distribute_path_nodebias .> 0.0)

# 	# cross validation routine that distributes `fold` (with debias) 
#     @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
#         path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
#     @test length(distribute_fold_debias) == 20
#     @test all(distribute_fold_debias .> 0.0)

# 	# cross validation routine that distributes `fold` (no debias) 
#     @time distribute_fold_nodebias = cv_iht_distribute_fold(y, x, z, d=d(), l=l, 
#         path=path, q=q, folds=folds, verbose=false, debias=false, parallel=true);
#     @test length(distribute_fold_nodebias) == 20
#     @test all(distribute_fold_nodebias .> 0.0)
# end

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
    q = 5
    folds = rand(1:q, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l, 
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test all(distribute_fold_debias .≈ distribute_path_debias)
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

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l, 
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_path_debias) == 20
    @test all(distribute_fold_debias .≈ distribute_path_debias)
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

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test all(distribute_fold_debias .≈ distribute_path_debias)
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
    y = T.(y)

    #specify path and folds
    path = 1:20
    q = 3
    folds = rand(1:q, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(y, x, z, d=d(), l=l, path=path, q=q,
        folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test isapprox(distribute_fold_debias, distribute_path_debias, atol=1e-4)

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(y, x, z, d=d(), l=l,
        path=path, q=q, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_fold_nodebias) == 20
    @test isapprox(distribute_fold_nodebias, distribute_path_nodebias, atol=1e-4)
end
