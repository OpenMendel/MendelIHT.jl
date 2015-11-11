export L0_reg_gpu
export iht_path

# shortcut for OpenCL module name
cl = OpenCL

# subroutine to mask entries of a vector x
function mask!(
	x        :: DenseVector{Float64}, 
	v        :: DenseVector{Int}, 
	val      :: Int, 
	mask_val :: Float64; 
	n        :: Int = length(x)
)
	n == length(v) || throw(BoundsError("Vector x and its mask must have same length"))
	@inbounds for i = 1:n
		if v[i] == val 
			x[i] = mask_val 
		end
	end
end
		

# ITERATIVE HARD THRESHOLDING USING A PLINK BED FILE
#
# This function computes a hard threshold update
#
#    b+ = P_{S_k}(b + mu*X'(y - Xb))
#
# where mu is the step size (or learning rate) and P_{S_k} denotes the projection onto the set S_k defined by
#
#     S_k = { x in R^p : || x ||_0 <= k }. 
#
# The projection in question preserves the largest k components of b in magnitude, and it sends the remaining 
# p-k components to zero. This update is intimately related to a projected gradient step used in Landweber iteration.
# Unlike the Landweber method, this function performs a line search on mu whenever the step size exceeds a specified
# threshold omega given by
#
#     omega = || b+ - b ||_2^2 / || X(b+ - b) ||_2^2.
#
# By backtracking on mu, this function guarantees a stable estimation of a sparse b. 
#
# This function is tuned to operate on a PLINK BEDFile object. As such, it decompresses genotypes on the fly.
#
# Arguments:
#
# -- b is the iterate of p model components;
# -- x is the BEDFile object that contains the compressed n x p design matrix;
# -- y is the vector of n responses;
# -- k is the model size;
# -- g is the negative gradient X'*(Y - Xbeta);
#
# Optional Arguments:
#
# -- p is the number of predictors. Defaults to length(b).
# -- n is the number of samples. Defaults to length(y).
# -- b0 is the previous iterate beta. Defaults to b.
# -- xb = x*b.
# -- xb0 = x*b0.
# -- bk is a temporary array to store the k floats corresponding to the support of b.
# -- xk is a temporary array to store the k columns of x corresponding to the support of b.
# -- gk is a temporary array of k floats used to subset the k components of the gradient g with the support of b.
# -- xgk = x*gk. 
# -- max_step is the maximum number of backtracking steps to take. Defaults to 50.
# -- sortidx is a vector to store the indices that would sort beta. Defaults to p zeros of type Int. 
# -- betak is a vector to store the largest k values of beta. Defaults to k zeros of type Float64. 
# -- IDX and IDX0 are BitArrays indicating the nonzero status of components of beta. They default to falses.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
# based on the HardLab demonstration code written in MATLAB by Thomas Blumensath
# http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html 
function iht(
	b         :: DenseVector{Float64}, 
	x         :: BEDFile, 
	y         :: DenseVector{Float64}, 
	k         :: Int, 
	g         :: DenseVector{Float64},
	mask_n    :: DenseVector{Int};
	pids      :: DenseVector{Int}     = procs(),
	n         :: Int                  = length(y), 
	p         :: Int                  = length(b), 
	b0        :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = b[localindexes(S)], pids=pids), 
	means     :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds   :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids), 
	Xb        :: DenseVector{Float64} = xb(x,b,IDX,k,mask_n, means=means, invstds=invstds, pids=pids), 
	Xb0       :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = Xb[localindexes(S)], pids=pids), 
	xk        :: DenseMatrix{Float64} = zeros(Float64,n,k), 
	xgk       :: DenseVector{Float64} = zeros(Float64,n), 
	gk        :: DenseVector{Float64} = zeros(Float64,k), 
	bk        :: DenseVector{Float64} = zeros(Float64,k),
	sortidx   :: DenseVector{Int}     = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids), 
	IDX       :: BitArray{1}          = falses(p), 
	IDX0      :: BitArray{1}          = copy(IDX), 
	max_step  :: Int                  = 50, 
	iter      :: Int                  = 1
) 
	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		selectperm!(sortidx,sdata(g),k, by=abs, rev=true, initialized=true)
		IDX[sortidx[1:k]] = true;
	end

	# if support has not changed between iterations,
	# then xk and gk are the same as well
	# avoid extracting and computing them if they have not changed
	# one exception: we should always extract columns on first iteration
	if !isequal(IDX, IDX0) || iter < 2 
		decompress_genotypes!(xk, x, IDX, mask_n, means=means, invstds=invstds) 
	end

	# store relevant components of gradient
	fill_perm!(sdata(gk), sdata(g), IDX, k=k, p=p)	# gk = g[IDX]

	# now compute subset of x*g
	BLAS.gemv!('N', one(Float64), sdata(xk), sdata(gk), zero(Float64), sdata(xgk))

	# warn if xgk only contains zeros
	all(xgk .== zero(Float64)) && warn("Entire active set has values equal to 0")

	# compute step size
	mu = sumabs2(sdata(gk)) / sumabs2(sdata(xgk))

	# notify problems with step size 
	isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))
	mu <= eps(typeof(mu))  && warn("Step size $(mu) is below machine precision, algorithm may not converge correctly")

	# take gradient step
	BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

	# preserve top k components of b
	project_k!(b, bk, sortidx, k)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	xb!(Xb,x,b,IDX,k,mask_n, means=means, invstds=invstds, pids=pids)

	# calculate omega
	omega_top = sqeuclidean(sdata(b),(b0))
	omega_bot = sqeuclidean(sdata(Xb),sdata(Xb0))

	# backtrack until mu sits below omega and support stabilizes
	mu_step = 0
	while mu*omega_bot > 0.99*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# warn if mu falls below machine epsilon 
		mu <= eps(Float64) && warn("Step size equals zero, algorithm may not converge correctly")

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

		# recompute projection onto top k components of b
		project_k!(b, bk, sortidx, k)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		xb!(Xb,x,b,IDX,k,mask_n, means=means, invstds=invstds, pids=pids)

		# calculate omega
		omega_top = sqeuclidean(sdata(b),(b0))
		omega_bot = sqeuclidean(sdata(Xb),sdata(Xb0))

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step
end


# L0 PENALIZED LEAST SQUARES REGRESSION FOR WHOLE GWAS
#
# This routine solves the optimization problem
#
#     min 0.5*|| Y - XB ||_2^2 
#
# subject to
#
#     B in S_k = { x in R^p : || x ||_0 <= k }. 
#
# It uses Thomas Blumensath's iterative hard thresholding framework to keep B feasible.
#
# Arguments:
# -- X is the BEDFile object that contains the compressed n x p design matrix
# -- Y is the n x 1 continuous response vector
# -- k is the desired size of the support (active set)
#
# Optional Arguments:
# -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to zeros(p).
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 1000.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tol is the global tol. Defaults to 5e-5.
# -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
# -- several temporary arrays for intermediate steps of algorithm calculations:
#		Xk        = zeros(Float64,n,k)  # store k columns of X
#		r         = zeros(Float64,n)	# for || Y - XB ||_2^2
#		Xb        = zeros(Float64,n)	# X*beta 
#		Xb0       = zeros(Float64,n)	# X*beta0 
#		b0        = zeros(Float64,p)	# previous iterate beta0 
#		df        = zeros(Float64,p)	# (negative) gradient 
#		tempkf    = zeros(Float64,k)    # temporary array of k floats 
#		idx       = zeros(Float64,k)    # another temporary array of k floats 
#		tempn     = zeros(Float64,n)    # temporary array of n floats 
#		indices   = collect(1:p)	    # indices that sort beta 
#		support   = falses(p)			# indicates nonzero components of beta
#		support0  = copy(support)		# store previous nonzero indicators
#
# Outputs are wrapped into a Dict with the following fields:
# -- time is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults
# -- iter is the number of iterations that the algorithm took
# -- loss is the optimal loss (residual sum of squares divided by sqrt of RSS with previous iterate)
# -- beta is the final iterate
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function L0_reg(
	X           :: BEDFile, 
	Y           :: DenseVector{Float64}, 
	k           :: Int,
	kernfile    :: ASCIIString; 
	pids        :: DenseVector{Int}     = procs(),
	n           :: Int                  = length(Y), 
	p           :: Int                  = size(X,2), 
	Xk          :: DenseMatrix{Float64} = zeros(Float64, (n,k)), 
	b           :: DenseVector{Float64} = SharedArray(Float64, p, pids=pids), 
	b0          :: DenseVector{Float64} = SharedArray(Float64, p, pids=pids), 
	df          :: DenseVector{Float64} = SharedArray(Float64, p, pids=pids), 
	r           :: DenseVector{Float64} = SharedArray(Float64, n, pids=pids), 
	Xb          :: DenseVector{Float64} = SharedArray(Float64, n, pids=pids), 
	Xb0         :: DenseVector{Float64} = SharedArray(Float64, n, pids=pids), 
	tempn       :: DenseVector{Float64} = SharedArray(Float64, n, pids=pids), 
	tempkf      :: DenseVector{Float64} = zeros(Float64,k), 
	idx         :: DenseVector{Float64} = zeros(Float64,k), 
	indices     :: DenseVector{Int}     = SharedArray(Int, p, init = S->S[localindexes(S)] = localindexes(S), pids=pids), 
	support     :: BitArray{1}          = falses(p), 
	support0    :: BitArray{1}          = falses(p), 
	mask_n      :: DenseVector{Int}     = ones(Int,n), 
	means       :: DenseVector{Float64} = mean(Float64,X, shared=true, pids=pids), 
	invstds     :: DenseVector{Float64} = invstd(X,means, shared=true, pids=pids), 
	tol         :: Float64              = 1e-4, 
	max_iter    :: Int                  = 100, 
	max_step    :: Int                  = 50, 
	quiet       :: Bool                 = true,
	wg_size     :: Int                  = 512,
	y_chunks    :: Int                  = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int                  = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
	r_chunks    :: Int                  = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0), 
	device      :: cl.Device            = last(cl.devices(:gpu)),
	ctx         :: cl.Context           = cl.Context(device), 
	queue       :: cl.CmdQueue          = cl.CmdQueue(ctx),
	x_buff      :: cl.Buffer            = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r)),
	m_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer            = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer            = cl.Buffer(Float64, ctx, (:rw), p * y_chunks),
	mask_buff   :: cl.Buffer            = cl.Buffer(Int,    ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
	genofloat   :: cl.LocalMem          = cl.LocalMem(Float64, wg_size),
	program     :: cl.Program           = cl.Program(ctx, source=kernfile) |> cl.build!,
	xtyk        :: cl.Kernel            = cl.Kernel(program, "compute_xt_times_vector"),
	rxtyk       :: cl.Kernel            = cl.Kernel(program, "reduce_xt_vec_chunks"),
	reset_x     :: cl.Kernel            = cl.Kernel(program, "reset_x"),
	wg_size32   :: Int32                = convert(Int32, wg_size),
	n32         :: Int32                = convert(Int32, n),
	p32         :: Int32                = convert(Int32, p),
	y_chunks32  :: Int32                = convert(Int32, y_chunks),
	y_blocks32  :: Int32                = convert(Int32, y_blocks),
	blocksize32 :: Int32                = convert(Int32, X.blocksize),
	r_length32  :: Int32                = convert(Int32, p*y_chunks)
)

	# start timer
	tic()

	# first handle errors
	k        >= 0            || throw(ArgumentError("Value of k must be nonnegative!\n"))
	max_iter >= 0            || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
	max_step >= 0            || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
	tol      >  eps(Float64) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

	sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

	# initialize return values
	mm_iter   = 0	    # number of iterations of L0_reg
	mm_time   = zero(Float64)		# compute time *within* L0_reg
	next_loss = zero(Float64)		# loss function value 

	# initialize floats 
	current_loss = Inf	# tracks previous objective function value
	the_norm     = zero(Float64) 	# norm(b - b0)
	scaled_norm  = zero(Float64) 	# the_norm / (norm(b0) + 1)
	mu           = zero(Float64) 	# Landweber step size, 0 < tau < 2/rho_max^2

	# initialize integers
	i       = 0       	# used for iterations in loops
	mu_step = 0        	# counts number of backtracking steps for mu

	# initialize booleans
	converged = false   # scaled_norm < tol?
   
	# update Xb, r, and gradient 
	if sum(support) == 0
		fill!(Xb,zero(Float64))
		copy!(r,sdata(Y))
		mask!(r, mask_n, 0, zero(Float64), n=n)
	else
		xb!(Xb,X,b,support,k,mask_n, means=means, invstds=invstds, pids=pids)
		difference!(r, Y, Xb)
		mask!(r, mask_n, 0, zero(Float64), n=n)
	end

	# calculate the gradient using the GPU
	xty!(df, df_buff, X, x_buff, r, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, X.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)

	# update loss 
	next_loss = Inf 

	# formatted output to monitor algorithm progress
	if !quiet
		 println("\nBegin MM algorithm\n") 
		 println("Iter\tHalves\tMu\t\tNorm\t\tObjective")
		 println("0\t0\tInf\t\tInf\t\tInf")
	end

	# main loop
	for mm_iter = 1:max_iter
 
		# notify and break if maximum iterations are reached.
		if mm_iter >= max_iter

			if !quiet
				print_with_color(:red, "MM algorithm has hit maximum iterations $(max_iter)!\n") 
				print_with_color(:red, "Current Objective: $(current_loss)\n") 
			end

			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end
		
		# save values from previous iterate 
		copy!(sdata(b0),sdata(b))	# b0 = b	
		copy!(sdata(Xb0),sdata(Xb))	# Xb0 = Xb
		current_loss = next_loss

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df,mask_n, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, Xb=Xb, Xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortidx=indices, gk=idx, means=means, invstds=invstds,iter=mm_iter, pids=pids) 

		# update residuals
		difference!(r,Y,Xb)
		mask!(r, mask_n, 0, zero(Float64), n=n)

		# use updated residuals to recompute the gradient on the GPU 
		xty!(df, df_buff, X, x_buff, r, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, X.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)

		# update objective
		next_loss = 0.5*sumabs2(sdata(r))

		# guard against numerical instabilities
		# ensure that objective is finite
		# if not, throw error
		isnan(next_loss) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_loss) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tol
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_loss)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# stop time
			mm_time = toq()

			if !quiet
				println("\nMM algorithm has converged successfully.")
				println("MM Results:\nIterations: $(mm_iter)") 
				println("Final Loss: $(next_loss)") 
				println("Total Compute Time: $(mm_time)") 
			end

			# these are output variables for function
			# wrap them into a Dict and return
			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end

		# algorithm is unconverged at this point, so check descent property 
		# if objective increases, then abort
		if next_loss > current_loss + tol
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_loss)\n") 
				print_with_color(:red, "Next Objective: $(next_loss)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_loss - current_loss))\n")
			end
			throw(ErrorException("Descent failure!"))
#			output = Dict{ASCIIString, Any}("time" => -one(Float64), "loss" => -one(Float64), "iter" => -1, "beta" => fill!(b,Inf))
			return output
		end
	end # end main loop
end # end function



# COMPUTE AN IHT REGULARIZATION PATH FOR LEAST SQUARES REGRESSION USING GWAS DATA
# This subroutine computes a regularization path for design matrix X and response Y from initial model size k0 to final model size k.
# The default increment on model size is 1. The path can also be warm-started with a vector b.
# This variant requires a calculated path in order to work.
#
# Arguments:
# -- x is the BEDFILE that contains the compressed n x p design matrix.
# -- y is the n-vector of responses
# -- path is an Int array that contains the model sizes to test
#
# Optional Arguments:
# -- b is the p-vector of effect sizes. This argument permits warmstarts to the path computation. Defaults to zeros.
# -- max_iter caps the number of iterations for the algorithm. Defaults to 1000.
# -- max_step caps the number of backtracking steps in the IHT kernel. Defaults to 50.
# -- quiet is a Boolean that controls the output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function iht_path(
	x        :: BEDFile, 
	y        :: DenseVector{Float64}, 
	path     :: DenseVector{Int}, 
	kernfile :: ASCIIString; 
	pids     :: DenseVector{Int}     = procs(),
	means    :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds  :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
	mask_n   :: DenseVector{Int}     = ones(Int,length(y)),
	device   :: cl.Device            = last(cl.devices(:gpu)),
	tol      :: Float64              = 1e-4, 
	max_iter :: Int                  = 100, 
	max_step :: Int                  = 50, 
	n        :: Int                  = length(y), 
	p        :: Int                  = size(x,2), 
	wg_size  :: Int                  = 512,
	quiet    :: Bool                 = true
)

	# how many models will we compute?
	const num_models = length(path)			

	# preallocate SharedArrays for intermediate steps of algorithm calculations 
	b           = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# previous iterate beta0 
	b0          = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# previous iterate beta0 
	df          = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# (negative) gradient 
	Xb          = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# X*beta 
	Xb0         = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# X*beta0 
	r           = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)		# for || Y - XB ||_2^2
	tempn       = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)	   	# temporary array of n floats 

	# index vector for b has more complicated initialization
	indices     = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)

	# allocate the BitArrays for indexing in IHT
	# also preallocate matrix to store betas 
	support     = falses(p)						# indicates nonzero components of beta
	support0    = copy(support)					# store previous nonzero indicators
	betas       = spzeros(Float64,p,num_models)	# a matrix to store calculated models

	# allocate GPU variables
	y_chunks    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0)
    y_blocks    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0)
	r_chunks    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0)
	ctx         = cl.Context(device)
	queue       = cl.CmdQueue(ctx)
	program     = cl.Program(ctx, source=kernfile) |> cl.build!
	xtyk        = cl.Kernel(program, "compute_xt_times_vector")
	rxtyk       = cl.Kernel(program, "reduce_xt_vec_chunks")
	reset_x     = cl.Kernel(program, "reset_x")
	wg_size32   = convert(Int32, wg_size)
	n32         = convert(Int32, n)
	p32         = convert(Int32, p)
	y_chunks32  = convert(Int32, y_chunks)
	y_blocks32  = convert(Int32, y_blocks)
	blocksize32 = convert(Int32, x.blocksize)
	r_length32  = convert(Int32, p*y_chunks)
	x_buff      = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(x.x))
	m_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means))
	p_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds))
	y_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r))
	df_buff     = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df))
	red_buff    = cl.Buffer(Float64, ctx, (:rw),        p * y_chunks)
	mask_buff   = cl.Buffer(Int,     ctx, (:rw, :copy), hostbuf = sdata(mask_n))
	genofloat   = cl.LocalMem(Float64, wg_size)

	# compute the path
	@inbounds for i = 1:num_models
	
		# model size?
		q = path[i]

		quiet || print_with_color(:blue, "Computing model size $q.\n\n")

		# these arrays change in size from iteration to iteration
		# we must allocate them for every new model size
		Xk     = zeros(Float64,n,q)		# store q columns of X
		tempkf = zeros(Float64,q)   	# temporary array of q floats 
		idx    = zeros(Float64,q)		# another temporary array of q floats 

		# store projection of beta onto largest k nonzeroes in magnitude 
		project_k!(b, tempkf, indices, q)

		# now compute current model
		output = L0_reg(x,y,q,kernfile, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, support=support, support0=support0, means=means, invstds=invstds, wg_size=wg_size, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, device=device, ctx=ctx, queue=queue, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, df_buff=df_buff, red_buff=red_buff, genofloat=genofloat, program=program, xtyk=xtyk, rxtyk=rxtyk, reset_x=reset_x, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, r_length32=r_length32, mask_n=mask_n, mask_buff=mask_buff, pids=pids)

		# extract and save model
		copy!(sdata(b), output["beta"])

		# ensure that we correctly index the nonzeroes in b
		update_indices!(support, b, p=p)	
		fill!(support0, false)

		# put model into sparse matrix of betas
		betas[:,i] = sparsevec(sdata(b))
		
	end

	return betas
end	


# COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A REGULARIZATION PATH FOR ENTIRE GWAS
#
# For a regularization path given by the vector "path", 
# this function computes an out-of-sample error based on the indices given in the vector "test_idx". 
# The vector test_idx indicates the portion of the data to use for testing.
# The remaining data are used for training the model.
# This variant of one_fold() operates on a BEDFile object
#
# Arguments:
# -- x is the BEDFile object that contains the compressed n x p design matrix.
# -- y is the n-vector of responses.
# -- path is the Int array that indicates the model sizes to compute on the regularization path. 
#
# -- path is an integer array that specifies which model sizes to include in the path, e.g.
#    > path = collect(k0:increment:k_end).
# -- test_idx is the Int array that indicates which data to hold out for testing.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- tol is the convergence tol to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- max_step caps the number of permissible backtracking steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
# -- logreg is a switch to activate logistic regression. Defaults to false (perform linear regression).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function one_fold(
	x        :: BEDFile, 
	y        :: DenseVector{Float64}, 
	path     :: DenseVector{Int}, 
	kernfile :: ASCIIString,
	folds    :: DenseVector{Int}, 
	fold     :: Int; 
	pids     :: DenseVector{Int}     = procs(),
	means    :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds  :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids), 
	tol      :: Float64              = 1e-4,
	max_iter :: Int                  = 100, 
	max_step :: Int                  = 50, 
	n        :: Int                  = length(y),
	p        :: Int                  = size(x,2),
	wg_size  :: Int                  = 512,
	devidx   :: Int                  = 1,
	header   :: Bool                 = false,
	quiet    :: Bool                 = true
)

	# get list of available GPU devices
	# var device gets pointer to device indexed by variable devidx 
	device = cl.devices(:gpu)[devidx]

	# make vector of indices for folds
	test_idx = folds .== fold

	# train_idx is the vector that indexes the TRAINING set
	train_idx = !test_idx

	# how many indices are in test set? 
	test_size = sum(test_idx) 

	# GPU code requires Int variant of training indices, so do explicit conversion
	train_idx = convert(Vector{Int}, train_idx)
	test_idx  = convert(Vector{Int}, test_idx)

	# compute the regularization path on the training set
	betas = iht_path(x,y,path,kernfile, max_iter=max_iter, quiet=quiet, max_step=max_step, means=means, invstds=invstds, mask_n=train_idx, wg_size=wg_size, device=device, pids=pids, tol=tol, max_iter=max_iter, n=n, p=p)

	# tidy up
	gc()

	# preallocate vector for output
	myerrors = zeros(Float64, length(path))

	# allocate an index vector for b
	indices = falses(p)

	# allocate temporary arrays for the test set
	Xb = SharedArray(Float64, n, pids=pids) 
	b  = SharedArray(Float64, p, pids=pids) 
	r  = SharedArray(Float64, n, pids=pids) 

	# compute the mean out-of-sample error for the TEST set 
	# do this for every computed model in regularization path
	for i = 1:size(betas,2)

		# pull ith model in dense vector format 
		b2 = full(vec(betas[:,i]))

		# copy it into SharedArray b
		copy!(b,b2)

		# indices stores Boolean indexes of nonzeroes in b
		update_indices!(indices, b, p=p)

		# compute estimated response Xb with $(path[i]) nonzeroes
		xb!(Xb,x,b,indices,path[i],test_idx, means=means, invstds=invstds, pids=pids)

		# compute residuals
		difference!(r,y,Xb)

		# mask data from training set 
		# training set consists of data NOT in fold
		r[folds .!= fold] = zero(Float64) 

		# compute out-of-sample error as squared residual averaged over size of test set
		myerrors[i] = sumabs2(r) / test_size
	end

	return myerrors
end


# subroutine to calculate the approximate memory load of one fold on the GPU 
function onefold_device_memload(x::BEDFile, wg_size::Int, y_chunks::Int; prec64::Bool = true)

	# floating point bytes multiplier depends on precision
	# prec64 = true -> use double precision (8 bytes per float)
	# prec64 = false -> use single precision (4 bytes per float)
	bytemult = ifelse(prec64, 8, 4)
	
	# get dimensions of problem
	n = x.n
	p = size(x,2)

	# start counting bytes
	# genotype array has bitstype Int8, one byte per component
	nx = length(x.x)

	# residual
	nr = bytemult * n

	# gradient, means, precisions
	ndf = 3 * bytemult * p

	# reduction vector
	nrv = bytemult * p * y_chunks

	# bitmask, which has bitstype Int32 (4 bytes per component)
	nbm = 4 * n

	# local memory footprint, just in case
	loc = bytemult * wg_size

	# total memory in Mb, rounded up to nearest int
	totmem = nx + ndf + nrv + nbm + loc

	return ceil(Int, totmem / 1024^2)
end

# subroutine to compute the number of folds that will fit on the GPU at one time
function compute_max_gpu_load(x::BEDFile, wg_size::Int, device::cl.Device; prec64::Bool = true) 
	
	# number of chunks in residual
	y_chunks = div(x.n, wg_size) + (x.n % wg_size != 0 ? 1 : 0)

	# total available memory on current device
	gpu_memtot = ceil(Int, device[:global_mem_size] / 1024^2)

	# memory load of one CV fold on current device
	onefold_mem = onefold_device_memload(x,wg_size,y_chunks, prec64=prec64)	

	# how many folds could we fit on the current GPU?
	max_folds = div(gpu_memtot, onefold_mem)

	return max_folds
end


function pfold(
	xfile      :: ASCIIString, 
	xtfile     :: ASCIIString, 
	x2file     :: ASCIIString, 
	yfile      :: ASCIIString, 
	meanfile   :: ASCIIString, 
	invstdfile :: ASCIIString, 
	path       :: DenseVector{Int}, 
	kernfile   :: ASCIIString, 
	folds      :: DenseVector{Int},
	numfolds   :: Int;
	devindices :: DenseVector{Int} = ones(Int,numfolds), 
	pids       :: DenseVector{Int} = procs(),
	max_iter   :: Int  = 100, 
	max_step   :: Int  = 50, 
	quiet      :: Bool = true, 
	header     :: Bool = false
)

	# how many CPU processes can pfold use?
	np = length(pids)

	# report on CPU processes
	quiet || println("pfold: np = ", np)
	quiet || println("pids = ", pids)

	# set up function to share state (indices of folds) 
	i = 1
	nextidx() = (idx=i; i+=1; idx)

	# preallocate cell array for results
	results = cell(numfolds)

	# master process will distribute tasks to workers
	# master synchronizes results at end before returning
	@sync begin

		# loop over all workers
		for worker in pids 

			# exclude process that launched pfold, unless only one process is available
			if worker != myid() || np == 1

				# asynchronously distribute tasks
				@async begin
					while true	

						# grab next fold
						current_fold = nextidx() 

						# if current fold exceeds total number of folds then exit loop
						current_fold > numfolds && break

						# grab index of GPU device
						devidx = devindices[current_fold]

						# report distribution of fold to worker and device
						quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker and device $devidx.\n\n")
						
						# launch job on worker 
						# worker loads data from file paths and then computes the errors in one fold
						results[current_fold] = remotecall_fetch(worker) do 
								pids = [worker]
								x = BEDFile(Float64, xfile, xtfile, x2file, pids=pids, header=header)
								n = x.n
								p = size(x,2)
								y = SharedArray(abspath(yfile), Float64, (n,), pids=pids)
								means = SharedArray(abspath(meanfile), Float64, (p,), pids=pids)
								invstds = SharedArray(abspath(invstdfile), Float64, (p,), pids=pids)

								one_fold(x, y, path, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, devidx=devidx, pids=pids)
						end # end remotecall_fetch()
					end # end while
				end # end @async
			end # end if
		end # end for
	end # end @sync

	# return reduction (row-wise sum) over results
	return reduce(+, results[1], results) 
end




# PARALLEL CROSSVALIDATION ROUTINE FOR IHT OVER ENTIRE GWAS
#
# This function will perform n-fold cross validation for the ideal model size in IHT least squares regression.
# It computes several paths as specified in the "paths" argument using the design matrix x and the response vector y.
# Each path is asynchronously spawned using any available processor.
# For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
# The function to compute each path, "one_fold()", will return a vector of out-of-sample errors (MSEs).
# After all paths are computed, this function queries the RemoteRefs corresponding to these returned vectors.
# It then "reduces" all components along each path to yield averaged MSEs for each model size.
#
# Arguments:
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- y is the n-vector of responses.
# -- path is an integer array that specifies which model sizes to include in the path, e.g.
#    > path = collect(k0:increment:k_end).
# -- nfolds is the number of folds to compute.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to size(x,2).
# -- folds is the partition of the data. Defaults to a random partition into "nfolds" disjoint sets.
# -- tol is the convergence tol to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- max_step caps the number of permissible backtracking steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#    NOTA BENE: each processor outputs feed to the console without regard to the others,
#    so setting quiet=true can yield very messy output!
# -- logreg is a Boolean to indicate whether or not to perform logistic regression. Defaults to false (do linear regression).
# -- compute_model is a Boolean to indicate whether or not to recompute the best model. Defaults to false (do not recompute). 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function cv_iht(
	xfile         :: ASCIIString,
	xtfile        :: ASCIIString,
	x2file        :: ASCIIString,
	yfile         :: ASCIIString,
	meanfile      :: ASCIIString,
	invstdfile    :: ASCIIString,
	path          :: DenseVector{Int}, 
	kernfile      :: ASCIIString,
	folds         :: DenseVector{Int},
	numfolds      :: Int; 
	pids          :: DenseVector{Int} = procs(),
	tol           :: Float64          = 1e-4, 
	max_iter      :: Int              = 100, 
	max_step      :: Int              = 50, 
	wg_size       :: Int              = 512,
	quiet         :: Bool             = true, 
	compute_model :: Bool             = false,
	header        :: Bool             = false
) 
	# how many elements are in the path?
	num_models = length(path)

	# how many GPU devices are available to us?
	devs = cl.devices(:gpu)
	ndev = length(devs)

	# how many folds can we fit on a GPU at once?
	# count one less per GPU device, just in case
#	max_folds = zeros(Int, ndev)
#	for i = 1:ndev
#		max_folds[i] = max(compute_max_gpu_load(x, wg_size, devs[i], prec64 = true) - 1, 0) 
#	end

	# how many rounds of folds do we need to schedule?
#	fold_rounds = zeros(Int, ndev)
#	for i = 1:ndev
#		fold_rounds[i] = div(numfolds, max_folds[i]) + (numfolds % max_folds[i] != 0 ? 1 : 0)
#	end

	# assign index of a GPU device for each fold	
	# default is first GPU device (devidx = 1)
	devindices = ones(Int, numfolds)
#	for i = 1:numfolds
#		devindices[i] += i % ndev
#	end

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# only use the worker processes
	errors = pfold(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, kernfile, folds, numfolds,	max_iter=max_iter, max_step=max_step, quiet=quiet, devindices=devindices, pids=pids, header=header)

	# average the mses over number of folds
	errors ./= numfolds

	# what is the best model size?
	k = convert(Int, floor(mean(path[errors .== minimum(errors)])))

	# print results
	if !quiet
		println("\n\nCrossvalidation Results:")
		println("k\tMSE")
		@inbounds for i = 1:num_models
			println(path[i], "\t", errors[i])
		end
		println("\nThe lowest MSE is achieved at k = ", k) 
	end

	# recompute ideal model
	if compute_model

		# load data on *all* processes 
		x       = BEDFile(xfile, xtfile, x2file, header=header, pids=pids)
		n       = x.n
		p       = size(x,2)
		y       = SharedArray(abspath(yfile), Float64, (n,), pids=pids)
		means   = SharedArray(abspath(meanfile), Float64, (p,), pids=pids)
		invstds = SharedArray(abspath(invstdfile), Float64, (p,), pids=pids)

		# initialize parameter vector as SharedArray
		b = SharedArray(Float64, p)

		# first use L0_reg to extract model
		output = L0_reg(x,y,q,kernfile, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, wg_size=wg_size, device=device) 

		# which components of beta are nonzero?
		inferred_model = output["beta"] .!= zero(Float64) 
		bidx = find( x -> x .!= zero(Float64), b) 

		# allocate the submatrix of x corresponding to the inferred model
		x_inferred = zeros(Float64,n,sum(inferred_model))
		decompress_genotypes!(x_inferred,x,inferred_model,means=means,invstds=invstds)

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		xty = BLAS.gemv('T', one(Float64), x_inferred, y)	
		xtx = BLAS.gemm('T', 'N', zero(Float64), x_inferred, x_inferred)
		b = xtx \ xty
		return errors, b, bidx 
	end
	return errors
end
