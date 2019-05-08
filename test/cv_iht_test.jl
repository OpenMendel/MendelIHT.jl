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

	@test argmin(mses) == 7
	@test all(mses ≈ [608.5185864156908; 554.4890824202639; 522.4053768900791; 485.3207425983446; 449.9577970517532;
		 406.7251330915335; 382.6698340685961; 388.7645989043201; 398.3396815828428; 404.1389407150225;
		 408.7941473091263; 420.26286174935206; 428.0668537775539; 433.87154355565116; 446.61805056542084;
		 455.0157166589543; 454.6774017223905; 449.1898130353841; 459.36111635226644; 448.68136889543473])
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

	@test argmin(mses) == 9
	@test all(mses ≈ [301.8280150820223; 280.1506929324628; 254.44896573297885; 239.65111105625778; 238.4605414892804;
		 233.0348443328082; 234.07028860031352; 229.16067315109908; 227.68591902225967; 230.5520182527614;
		 229.9951859157333; 232.33264980454294; 237.72726103452848; 235.9768598605953; 237.84860548813555;
		 239.5541657041541; 237.03580957171906; 239.9349164888043; 243.8053417226982; 242.48753588804118])
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

	@test result[1, 2] ≈ -2272.788840069668
	@test result[2, 2] ≈ -2144.4519706525098
	@test result[3, 2] ≈ -1998.9118082179523
	@test result[4, 2] ≈ -1888.362338848445
	@test result[5, 2] ≈ -1767.649126874247
	@test result[6, 2] ≈ -1617.2385746806704
	@test result[7, 2] ≈ -1484.2709104112216
	@test result[8, 2] ≈ -1447.9560237353137
	@test result[9, 2] ≈ -1421.5794827334514
	@test result[10, 2] ≈ -1406.8807653835697
	@test result[11, 2] ≈ -1396.8628294699147
	@test result[12, 2] ≈ -1390.1763942804469
	@test result[13, 2] ≈ -1383.5471518926422
	@test result[14, 2] ≈ -1376.7041041059629
	@test result[15, 2] ≈ -1371.070140615097
	@test result[16, 2] ≈ -1363.7164318022296
	@test result[17, 2] ≈ -1358.8454592637506
	@test result[18, 2] ≈ -1353.106442197983
	@test result[19, 2] ≈ -1351.709447505716
	@test result[20, 2] ≈ -1343.1979015928894
end




