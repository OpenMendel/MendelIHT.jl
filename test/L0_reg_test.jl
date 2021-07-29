@testset "fit normal" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)
	show(result)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test findall(!iszero, result.beta) == [2384;3352;3353;4093;5413;5609;7403;8753;9089;9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-1.2601406011046452, -0.2674202492177914,
		0.1412081066471588, 0.289955803600036, 0.3666894767520663, -0.1371805027382694, -0.308254575616033,
		0.33288147012004443, 0.9645980728400257, -0.5094607091364866])
	@test result.c[1] ≈ -0.026283123286410772
	@test result.k == 10
	@test result.logl ≈ -1406.899627901812
end

@testset "fit Bernoulli" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test findall(!iszero, result.beta) == [1733, 1816, 2384, 5413, 7067, 8753, 8908, 9089, 9132, 9765]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [-0.2788091712310063, 0.31151387193421737, 
	-1.1283186622644128, 0.4999869530547436, -0.32685079548465007, 0.4138519813525528, -0.32779271347276445,
	 0.8644288162088136, -0.5064840683902377, -0.3284653237251541])
	@test result.c[1] ≈ 0.016116760859127627
	@test result.k == 10
	@test result.logl ≈ -489.8564887253825
end

@testset "fit Poisson" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Poisson
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)
	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test findall(!iszero, result.beta) == [298, 2384, 3157, 5891, 7067, 8753, 8755, 8931, 9089, 9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [0.09820585763162126, -0.37392590916093194, 
		0.08489568686444253, 0.11864310078500476, 0.09414834393479564, 0.10683788271304923, 0.12908044513140238, 
		0.12019432114450958, 0.29296236678486925, -0.12813041278457823])
	@test result.c[1] ≈ -0.011752998591789125
	@test result.k == 10
	@test result.logl ≈ -1294.452369780194
end

@testset "fit NegativeBinomial" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test count(!iszero, result.beta) == k
	@test findall(!iszero, result.beta) == [597, 934, 1610, 1774, 2384, 5413, 5614, 8993, 9089, 9132]
	@test all(result.beta[findall(!iszero, result.beta)] .≈ [0.08224940823109043, 0.08341097787419621, 
	0.1012409317334749, -0.15790186573300657, -0.2927019848104549, 0.10528516175659709, 0.09574395232655965, 
	-0.08510453836644973, 0.25998023048548063, -0.1935947191261095])
	@test result.c[1] ≈ -0.025855871894708525
	@test result.k == 10
	@test result.logl ≈ -1393.5142690674227
end

@testset "fit with non-genetic covariates" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 2) # 1st column intercept
	z[:, 2] .= randn(n)

	#define true_b and true_c
	true_b = zeros(p)
	true_b[1:k-2] = randn(k-2)
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
	true_c = [3.0; 3.5]

	#simulate phenotype
	prob = GLM.linkinv.(l, xla * true_b .+ z * true_c)
	y = [rand(d(i)) for i in prob]

	#run result
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test length(findall(!iszero, result.beta)) == 10
	@test findall(!iszero, result.c) == [1;2]
	@test result.k == 10
end

@testset "model selection on non-genetic covariates" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 2) # 1st column intercept
	z[:, 2] .= randn(n)

	# define true_b and true_c
	true_b = zeros(p)
	true_b[1:k-2] = randn(k-2)
	shuffle!(true_b)
	correct_position = findall(!iszero, true_b)
	true_c = [3.0; 0]

	# simulate phenotype
	prob = GLM.linkinv.(l, xla * true_b .+ z * true_c)
	y = [rand(d(i)) for i in prob]

	#run result, keeping only intercept
	zkeep = convert(BitVector, [true, false])
	result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, zkeep = zkeep)

	@test length(result.beta) == 10000
	@test length(result.c) == 2
	@test length(findall(!iszero, result.beta)) == 10
	@test count(!iszero, result.c) == 1 # the 2nd covariate should be excluded
	@test result.k == 10
end

@testset "Correlated predictors and double sparsity" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

	#simulat data with k true predictors, from distribution d and with link l.
    n = 1000
    p = 10000
    d = Normal
	l = canonicallink(d())
    block_size = 20
    num_blocks = Int(p / block_size)

	#set random seed
	Random.seed!(1111)

    # assign group membership
    membership = collect(1:num_blocks)
    g = zeros(Int64, p)
    for i in 1:length(membership)
        for j in 1:block_size
            cur_row = block_size * (i - 1) + j
            g[block_size*(i - 1) + j] = membership[i]
        end
    end
    
    #simulate correlated snparray
    x = simulate_correlated_snparray(undef, n, p)
    z = ones(n) # the intercept
    x_float = convert(Matrix{Float64}, x, model=ADDITIVE_MODEL, center=true, scale=true)
    
    #simulate true model, where 5 groups each with 3 snps contribute
    true_b = zeros(p)
    true_groups = randperm(num_blocks)[1:5]
    within_group = [randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3] randperm(block_size)[1:3]]
    correct_position = zeros(Int64, 15)
    for i in 1:5
        cur_group = block_size * (true_groups[i] - 1)
        cur_group_snps = cur_group .+ within_group[:, i]
        correct_position[3*(i-1)+1:3i] .= cur_group_snps
    end
    for i in 1:15
        true_b[correct_position[i]] = rand(-1:2:1) * 0.2
    end
    sort!(correct_position)
        
    # simulate phenotype
    if d == Normal || d == Bernoulli || d == Poisson
        prob = GLM.linkinv.(l, x_float * true_b)
        clamp!(prob, -20, 20)
        y = [rand(d(i)) for i in prob]
    elseif d == NegativeBinomial
        nn = 10
        μ = GLM.linkinv.(l, x_float * true_b)
        clamp!(μ, -20, 20)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    end
    y = Float64.(y)
    
    #run IHT without groups
    k = 15
    ungrouped = fit_iht(y, x_float, z, J=1, k=k, d=d(), l=l)

    #run IHT with 5 groups each with 3 predictor
    J = 5
    k = 3
    grouped = fit_iht(y, x_float, z, J=J, k=k, d=d(), l=l, group=g)

    @test length(findall(!iszero, ungrouped.beta)) == 15
    @test length(findall(!iszero, grouped.beta)) == 15
end

@testset "Negative binomial nuisance parameter" begin
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	# set random seed for reproducibility
	Random.seed!(1111) 

	# simulate SnpArrays data
	x = simulate_correlated_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n, 1) 
	y, true_b, correct_position = simulate_random_response(xla, k, d, l, r=10);

	@time newton = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:Newton)
	@test typeof(newton.d) == NegativeBinomial{Float64}
	@test newton.d.p == 0.5 # p parameter not used 
	@test newton.d.r ≥ 7 # r converges to 10 faster

	@time mm = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, est_r=:MM)
	@test typeof(mm.d) == NegativeBinomial{Float64}
	@test mm.d.p == 0.5
	@test mm.d.r ≥ 2 # r converges to 10 slower

	# simulate floating point data
	Random.seed!(1111) 
	T = Float32
	x = randn(T, n, p)
	z = ones(T, n) 

	# simulate true model b, and the correct non-0 positions of b
	true_b = zeros(T, p)
	true_b[1:k] .= collect(0.1:0.1:1.0)
	true_c = [T.(4.0)]
	shuffle!(true_b)

	# simulate response
	r = 20
    μ = GLM.linkinv.(l, xla * true_b)
    clamp!(μ, -20, 20)
    prob = 1 ./ (1 .+ μ ./ r)
    y = [rand(d(r, Float64(i))) for i in prob] 
    y = T.(y)

	d = d(r, Float32(0.5)) # need Float32 for eltype of d
	@time newton = fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=:Newton)
	@test typeof(newton.d) == NegativeBinomial{Float32}
	@test newton.d.p == 0.5 
	@test newton.d.r ≥ 0

	@time mm = fit_iht(y, x, z, J=1, k=k, d=d, l=l, est_r=:MM)
	@test typeof(mm.d) == NegativeBinomial{Float32}
	@test mm.d.p == 0.5
	@test mm.d.r ≥ 0
end

@testset "initialze beta" begin
	#simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(1111)

	#construct SnpArraym, snpmatrix, and non genetic covariate (intercept)
	x = simulate_random_snparray(undef, n, p)
	xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true) 
	z = ones(n)

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(xla, k, d, l)

	#run with and without initializing beta
	@time result = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, init_beta=false)
	@time result2 = fit_iht(y, xla, z, J=1, k=k, d=d(), l=l, init_beta=true)
	@test all(findall(!iszero, result.beta) .== findall(!iszero, result2.beta))
end
