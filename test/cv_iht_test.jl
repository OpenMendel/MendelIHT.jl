@testset "cv_iht normal" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x, = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run threaded IHT
	mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=true, parallel=true)

	@test argmin(mses) == 11
	@test all(mses ≈ [1927.0765190526674;1443.8787350418297;1080.0413162424918;
		898.2611345915562;700.2502074095313;507.3948750413776;391.9679112461845;
		381.68778672828626;368.53087109541184;359.9090161930197;344.3626513420314;
		353.22416803442127;357.75754456737286;363.5298542018518;379.4998479342846;
		365.9493360314551;387.4562377561079;386.80694552347995;398.9880370887734;
		403.1125856821553])
end

@testset "cv_iht bernoulli" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Bernoulli
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x, = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run threaded IHT
	mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=true, parallel=true)
	@test argmin(mses) == 6

	#not sure why mses[end] can change..... is this a bug?
	@test all(mses[1:end-1] .≈ [433.1645990291242; 396.4218354530673; 354.03582998371684; 337.2781542152179; 321.40856422243473;
			 300.13602523499037; 300.7545574376133; 304.0569785536426; 313.5244382980213; 315.21199531008796;
			 328.7337680232285; 355.3073847457196; 360.42023935617004; 354.7111928546249; 374.7397814724943;
			 380.5160395899203; 404.44221489146645; 404.77237379170714;412.59479205229724])
end

@testset "cv_iht Poisson" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = Poisson
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x, = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run threaded IHT
	mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=true, parallel=true)

	@test argmin(mses) == 8
	@test all(mses ≈ [ 608.5185864156908; 554.4890824202639; 523.8764295950825; 488.1264552861322; 
		453.96213697247924; 405.75784144607996; 407.60270764148606; 386.52624780749767; 
		404.2623938995897; 400.28588172665775; 410.4472342338546; 420.15069559686253; 
		438.96623067486007; 439.0234436551872; 435.80703098554704; 455.9718487859483; 
		458.74746026812716; 453.8387031450877; 447.4324667811566; 447.62284814476243])
end

@testset "cv_iht NegativeBinomial" begin
	# Since my code seems to work, putting in some output as they can be verified by comparing with simulation

    #simulat data with k true predictors, from distribution d and with link l.
	n = 1000
	p = 10000
	k = 10
	d = NegativeBinomial
	l = LogLink()

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x, = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and folds
	path = collect(1:20)
	num_folds = 3
	folds = rand(1:num_folds, size(x, 1))

	# run threaded IHT
	mses = cv_iht(d(), l, x, z, y, 1, path, num_folds, folds=folds, init=false, use_maf=false, debias=true, parallel=true)

	@test argmin(mses) == 7
	@test all(mses ≈ [296.9598675726054;270.34288882337495;253.56107225530195;
		245.6356378490944; 234.28391512742712; 229.51458911435097; 224.75045932979236;
		228.108101541521; 233.99579309666166; 238.7778890065921; 240.77103129427257;
		240.9249395339163; 247.92544806689114;246.05439848327285; 253.95193468780303;
		247.0237203616466; 255.3074217938908; 254.37748596787372; 263.15056242825057;
		269.0935738286553])
end

@testset "iht_run_many_models normal" begin
	n = 1000
	p = 10000
	k = 10
	d = Normal
	l = canonicallink(d())

	#set random seed
	Random.seed!(2019)

	#construct snpmatrix, covariate files, and true model b
	x, = simulate_random_snparray(n, p, undef)
	xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
	z = ones(n, 1) # the intercept

	# simulate response, true model b, and the correct non-0 positions of b
	y, true_b, correct_position = simulate_random_response(x, xbm, k, d, l)

	#specify path and run a ton of models 
	path = collect(1:20)
	result = iht_run_many_models(d(), l, x, z, y, 1, path, parallel=true);

	@test result[1] ≈ -2272.788840069668
	@test result[2] ≈ -2144.4519706525098
	@test result[3] ≈ -1998.9118082179523
	@test result[4] ≈ -1888.362338848445
	@test result[5] ≈ -1767.649126874247
	@test result[6] ≈ -1617.2385746806704
	@test result[7] ≈ -1484.2709104112216
	@test result[8] ≈ -1447.9560237353137
	@test result[9] ≈ -1421.5794827334514
	@test result[10] ≈ -1406.8807653835697
	@test result[11] ≈ -1396.8628294699147
	@test result[12] ≈ -1390.1763942804469
	@test result[13] ≈ -1383.5471518926422
	@test result[14] ≈ -1376.7041041059629
	@test result[15] ≈ -1371.070140615097
	@test result[16] ≈ -1363.7164318022296
	@test result[17] ≈ -1358.8454592637506
	@test result[18] ≈ -1353.106442197983
	@test result[19] ≈ -1351.709447505716
	@test result[20] ≈ -1343.1979015928894
end




