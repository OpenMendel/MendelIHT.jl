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
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

    #specify path and folds
    path = collect(1:20)
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=true, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test all(distribute_fold_debias .≈ distribute_path_debias)

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
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
    path = collect(1:20)
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test all(isapprox.(distribute_path_debias, distribute_fold_debias, atol=1e-4))

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_fold_nodebias) == 20
    @test all(isapprox.(distribute_path_nodebias, distribute_fold_nodebias, atol=1e-4))
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
# 	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
# 	z = ones(n, 1) # the intercept

# 	# simulate response, true model b, and the correct non-0 positions of b
# 	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

# 	#specify path and folds
# 	path = collect(1:20)
# 	num_folds = 3
# 	folds = rand(1:num_folds, size(x, 1))

# 	# cross validation routine that distributes `path` (with debias) 
# 	@time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
# 	@test argmin(distribute_path_debias) == 6
# 	@test all(distribute_path_debias .≈ [433.1645990291242; 396.4218354530673; 354.03582998371684; 337.2781542152179; 321.40856422243473;
# 		 300.13602523499037; 300.7545574376133; 304.0569785536426; 313.5244382980213; 315.21199531008796;
# 		 328.7337680232285; 355.3073847457196; 360.42023935617004; 354.7111928546249; 374.7397814724943;
# 		 380.5160395899203; 404.44221489146645; 404.77237379170714;412.59479205229724; 439.5097230713853])

# 	# cross validation routine that distributes `path` (no debias) 
# 	@time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_path_nodebias) == 7
# 	@test all(distribute_path_nodebias ≈  [433.1646582679374; 396.4213330004181; 354.0359578984675; 341.43957528366184;
# 		 331.0096177161414; 310.55436001851103; 300.74693115572484; 308.3221565445261;
# 		 320.1437416545087; 325.44566775172825; 330.59296483296725; 360.1505146741712;
# 		 369.65579010749514; 372.0215814075303; 378.8006645943797; 392.6758894292982;
# 		 384.283182617048; 432.5312890223311; 445.4448285798073; 434.96984334483506])

# 	# cross validation routine that distributes `fold` (with debias) 
# 	@time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
# 	@test argmin(distribute_fold_debias) == 6
# 	@test all(distribute_fold_debias .≈ distribute_path_debias)

# 	# cross validation routine that distributes `fold` (no debias) 
# 	@time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
# 	@test argmin(distribute_fold_nodebias) == 7
# 	@test all(distribute_fold_nodebias .≈ distribute_path_nodebias)
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
    path = collect(1:20)
    num_folds = 5
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
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
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

    #specify path and folds
    path = collect(1:20)
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
    @test argmin(distribute_fold_debias) == 7
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
    xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
    z = ones(n, 1) # the intercept

    # simulate response, true model b, and the correct non-0 positions of b
    y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

    #specify path and folds
    path = collect(1:20)
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0.0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
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
    path = collect(1:20)
    num_folds = 3
    folds = rand(1:num_folds, size(x, 1))

    # cross validation routine that distributes `path` (with debias) 
    @time distribute_path_debias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true)
    @test length(distribute_path_debias) == 20
    @test all(distribute_path_debias .> 0)

    # cross validation routine that distributes `path` (no debias) 
    @time distribute_path_nodebias = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_path_nodebias) == 20
    @test all(distribute_path_nodebias .> 0)

    # cross validation routine that distributes `fold` (with debias) 
    @time distribute_fold_debias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=true, parallel=true);
    @test length(distribute_fold_debias) == 20
    @test isapprox(distribute_fold_debias, distribute_path_debias, atol=1e-4)

    # cross validation routine that distributes `fold` (no debias) 
    @time distribute_fold_nodebias = cv_iht_distribute_fold(d(), l, x, z, y, 1, path, num_folds, folds=folds, verbose=false, debias=false, parallel=true);
    @test length(distribute_fold_nodebias) == 20
    @test isapprox(distribute_fold_nodebias, distribute_path_nodebias, atol=1e-4)
end
