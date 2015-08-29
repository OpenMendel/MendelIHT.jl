module IHT

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using NumericExtensions: sumsq
using StatsBase: sample, logistic
using RegressionTools

export L0_reg
export L0_log
export L0_log2
export iht_path
export iht_path_log
export iht_path_log2
export cv_iht
export cv_get_folds


function standardize_column!(x, means, invstds, j; n::Int = size(X,1))
	m = means[j]
	s = invstds[j]
	@inbounds @simd for i = 1:n
		x[i,j] = (x[i,j] - m) * s
	end
end

# ITERATIVE HARD THRESHOLDING
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
# Arguments:
#
# -- b is the iterate of p model components;
# -- x is the n x p design matrix;
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
# -- sortk is a vector to store the largest k indices of beta. Defaults to k zeros of type Int.
# -- betak is a vector to store the largest k values of beta. Defaults to k zeros of type Float64. 
# -- IDX and IDX0 are BitArrays indicating the nonzero status of components of beta. They default to falses.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
# based on the HardLab demonstration code written in MATLAB by Thomas Blumensath
# http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html 
function iht(b::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, k::Int, g::DenseArray{Float64,1}; n::Int = length(y), p::Int = length(b), max_step::Int = 50, sortidx::DenseArray{Int,1} = collect(1:p), IDX::BitArray{1} = falses(p), IDX0::BitArray{1} = copy(IDX), b0::DenseArray{Float64,1} = copy(b), xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), xb0::DenseArray{Float64,1} = copy(xb), xk::DenseArray{Float64,2} = zeros(n,k), gk::DenseArray{Float64,1} = zeros(k), xgk::DenseArray{Float64,1} = zeros(n), bk::DenseArray{Float64,1} = zeros(k), sortk::DenseArray{Int,1} = zeros(Int,k), step_multiplier::Float64 = 1.0)

	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		sortk = selectperm!(sortidx,g,k, p=p) 
		IDX[sortk] = true;
	end

	# store relevant columns of x
	update_xk!(xk, x, IDX, k=k, p=p, n=n)	# xk = x[:,IDX]

	# store relevant components of gradient
	fill_perm!(gk, g, IDX, k=k, p=p)	# gk = g[IDX]

	# now compute subset of x*g
	BLAS.gemv!('N', 1.0, xk, gk, 0.0, xgk)

	# compute step size
	mu = step_multiplier * sumsq(gk) / sumsq(xgk)

	# take gradient step
	BLAS.axpy!(p, mu, g, 1, b, 1)

	# preserve top k components of b
	sortk = selectperm!(sortidx,b,k, p=p)
	fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
	fill!(b,0.0)
	b[sortk] = bk
#	RegressionTools.project_k!(b, bk, sortk, sortidx, k, p=p)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	update_xb!(xb, x, b, sortk, k, p=p, n=n)

	# calculate omega
	omega = sqeuclidean(b,b0) / sqeuclidean(xb,xb0)

	# backtrack until mu sits below omega and support stabilizes
	mu_step = 0
	while mu > 0.99*omega && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, g, 1, b, 1)

		# recompute projection onto top k components of b
		sortk = selectperm!(sortidx,b,k, p=p)
		fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		RegressionTools.project_k!(b, bk, sortk, sortidx, k, p=p)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		update_xb!(xb, x, b, sortk, k, p=p, n=n)

		# calculate omega
		omega = sqeuclidean(b,b0) / sqeuclidean(xb,xb0)

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step
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
# -- sortk is a vector to store the largest k indices of beta. Defaults to k zeros of type Int.
# -- betak is a vector to store the largest k values of beta. Defaults to k zeros of type Float64. 
# -- IDX and IDX0 are BitArrays indicating the nonzero status of components of beta. They default to falses.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
# based on the HardLab demonstration code written in MATLAB by Thomas Blumensath
# http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html 
#function iht(b::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}, k::Int, g::DenseArray{Float64,1}; n::Int = length(y), p::Int = length(b), max_step::Int = 50, sortidx::DenseArray{Int,1} = collect(1:p), IDX::BitArray{1} = falses(p), IDX0::BitArray{1} = copy(IDX), b0::DenseArray{Float64,1} = copy(b), Xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), Xb0::DenseArray{Float64,1} = copy(xb), xk::DenseArray{Float64,2} = zeros(n,k), gk::DenseArray{Float64,1} = zeros(k), xgk::DenseArray{Float64,1} = zeros(n), bk::DenseArray{Float64,1} = zeros(k), sortk::DenseArray{Int,1} = zeros(Int,k), step_multiplier::Float64 = 1.0)
#function iht(b::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}, k::Int, g::DenseArray{Float64,1}; n::Int = length(y), p::Int = length(b), max_step::Int = 50, sortidx::DenseArray{Int,1} = collect(1:p), IDX::BitArray{1} = falses(p), IDX0::BitArray{1} = copy(IDX), b0::DenseArray{Float64,1} = copy(b), Xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), Xb0::DenseArray{Float64,1} = copy(xb), xk::DenseArray{Float64,2} = zeros(n,k), gk::DenseArray{Float64,1} = zeros(k), xgk::DenseArray{Float64,1} = zeros(n), bk::DenseArray{Float64,1} = zeros(k), sortk::DenseArray{Int,1} = zeros(Int,k), step_multiplier::Float64 = 1.0, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means), obj::Float64 = Inf, r::DenseArray{Float64,1} = zeros(n), stdsk::DenseArray{Float64,1} = zeros(k)) 
function iht(b::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}, k::Int, g::DenseArray{Float64,1}; n::Int = length(y), p::Int = length(b), max_step::Int = 50, sortidx::DenseArray{Int,1} = collect(1:p), IDX::BitArray{1} = falses(p), IDX0::BitArray{1} = copy(IDX), b0::DenseArray{Float64,1} = copy(b), Xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), Xb0::DenseArray{Float64,1} = copy(xb), xk::DenseArray{Float64,2} = zeros(n,k), gk::DenseArray{Float64,1} = zeros(k), xgk::DenseArray{Float64,1} = zeros(n), bk::DenseArray{Float64,1} = zeros(k), sortk::DenseArray{Int,1} = zeros(Int,k), step_multiplier::Float64 = 1.0, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means), stdsk::DenseArray{Float64,1} = zeros(k)) 

	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		sortk = selectperm!(sortidx,sdata(g),k, p=p) 
		IDX[sortk] = true;
	end

	# if support has not changed between iterations,
	# then xk and gk are the same as well
	# avoid extracting and computing them if they have not changed
	if !isequal(IDX, IDX0) || sum(IDX) == 0

		# store relevant columns of x
#		decompress_genotypes!(xk, x, IDX, y=xgk)
		decompress_genotypes!(xk, x, IDX, means=means, invstds=invstds) 

#		bk = means[IDX]
		fill_perm!(bk, means, IDX, k=k, p=p)
#		stdsk = invstds[IDX]
		fill_perm!(stdsk, invstds, IDX, k=k, p=p)
#		for j = 1:k
#			m = bk[j]
#			s = stdsk[j]
#			for i = 1:n
#				xk[i,j] = (xk[i,j] - m) * s
#			end
#		end
#		@sync @inbounds @parallel for i = 1:k
#			standardize_column!(xk, bk, stdsk, i, n=n)
#		end
#		xk[isnan(xk)] = 0.0
#		all(xk .== 0.0) && warn("Entire active set has genotypes equal to 0")

		# store relevant components of gradient
		fill_perm!(sdata(gk), sdata(g), IDX, k=k, p=p)	# gk = g[IDX]

		# now compute subset of x*g
		BLAS.gemv!('N', 1.0, xk, gk, 0.0, xgk)
	end
	
	# compute step size
	mu = step_multiplier * sumsq(sdata(gk)) / sumsq(sdata(xgk))

	# warn if step size falls below machine epsilon
	mu <= eps() && warn("Step size is below machine precision, algorithm may not converge correctly")

	# take gradient step
	BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

	# preserve top k components of b
	sortk = selectperm!(sortidx,b,k, p=p)
	fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
	fill!(b,0.0)
	b[sortk] = bk
#	project_k!(b, bk, sortk, sortidx, k, p=p)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	xb!(Xb,x,b,IDX,k, means=means, invstds=invstds)

	# calculate omega
#	omega = sqeuclidean(sdata(b),sdata(b0)) / sqeuclidean(sdata(Xb),sdata(Xb0))
	omega = sqeuclidean(b,b0) / sqeuclidean(Xb,Xb0)

	# backtrack until mu sits below omega and support stabilizes
	mu_step = 0
	while mu > 0.99*omega && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# warn if mu falls below machine epsilon 
		mu <= eps() && warn("Step size equals zero, algorithm may not converge correctly")

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

		# recompute projection onto top k components of b
		sortk = selectperm!(sortidx,b,k, p=p)
		fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		project_k!(b, bk, sortk, sortidx, k, p=p)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		xb!(Xb,x,b,IDX,k, means=means, invstds=invstds)

		# calculate omega
#		omega = sqeuclidean(sdata(b),sdata(b0)) / sqeuclidean(Xb,Xb0)
		omega = sqeuclidean(b,b0) / sqeuclidean(Xb,Xb0)

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step
end


function dmu(mu::Float64, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseArray{Float64,1} = update_xb(x,b,indices,k, n=n, p=p), lg::DenseArray{Float64,1} = loggrad(x,y,b,indices,k, n=n, p=p, xb=xb), xlg::DenseArray{Float64,1} = update_xb(x,lg,indices,k, n=n, p=k), lxb::DenseArray{Float64,1} = logistic(xb + mu*xlg))

	# initialize return value
	val = 0.0

	# dmu is a dot product of two vectors
	# iteratively accumulate this sum with one_dmu
	@inbounds @simd for i = 1:n
		val += xlg[j] * ( y[i] - lxb[i] ) 
	end

	return -val
end


function d2mu(mu::Float64, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseArray{Float64,1} = update_xb(x,b,indices,k, n=n, p=p), lg::DenseArray{Float64,1} = loggrad(x,y,b,indices,k, n=n, p=p, xb=xb), xlg::DenseArray{Float64,1} = update_xb(x,lg,indices,k, n=n, p=k), lxb::DenseArray{Float64,1} = logistic(xb + mu*xlg))

	# initialize return value
	val = 0.0

	# d2mu is a reduction (dot product) of a lot of matrix-vector operations
	# we can essentially decompose into three parts:
	# xlg, pi, (1 - pi) 
	# the actual value is a scalar
	# iteratively accumulate this sum with one_d2mu

	@inbounds @simd for i = 1:n
		val += xlg[i] * lxb[i] * (1.0 - lxb[i]) * xlg[i] 
	end

	return val
end



function iht2(b::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, k::Int, old_obj::Float64; n::Int = length(y), p::Int = length(b), max_step::Int = 50, sortidx::DenseArray{Int,1} = collect(1:p), IDX::BitArray{1} = falses(p), IDX0::BitArray{1} = copy(IDX), xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), sortk::DenseArray{Int,1} = collect(1:k), lg::DenseArray{Float64,1} = zeros(p), xlg::DenseArray{Float64,1} = zeros(n), bk::DenseArray{Float64,1} = zeros(k), lxb::DenseArray{Float64,1} = zeros(n), y2::DenseArray{Float64,1} = zeros(n))

	# which indices of beta are nonzero?
	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		sortk = selectperm!(sortidx,lg,k, p=p)
		IDX[sortk] = true;
	end

	# first update x*b
	# then save logistic(x*b)
	# y2 holds y - logistic(x*b)
	# then update lg, the gradient of logistic loglikelihood
	# finally, update x*lg
	update_xb!(xb, x, b, sortidx, k, n=n, p=k)
	logistic!(lxb, xb, n=n)
	update_y2!(y2,y,lxb, n=n)
	BLAS.gemv!('T', 1.0, x, y2, 0.0, lg)
	BLAS.gemv!('N', 1.0, x, lg, 0.0, xlg)

	# compute step size
	mu = calculate_mu(x, y, b, sortidx, k, n=n, p=p, xb=xb, lg=lg, xlg=xlg)

	# take gradient step
	BLAS.axpy!(p, mu, lg, 1, b, 1)

	# preserve top k components of b
	sortk = selectperm!(sortidx,b,k, p=p)
	fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
	fill!(b,0.0)
	b[sortk] = bk
#	project_k!(b, bk, sortk, sortidx, k, p=p)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	update_xb!(xb, x, b, sortk, k, p=p, n=n)

	# calculate objective function 
	new_obj = compute_loglik(y,x,b, n=n, xb=xb, center=false)

	# backtrack until mu sits below omega and support stabilizes
	# observe that we use -new_obj since the objective equals the NEGATIVE loglikelihood
	mu_step = 0
	while -new_obj > old_obj && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, lg, 1, b, 1)

		# recompute projection onto top k components of b
		sortk = selectperm!(sortidx, b,k, p=p)
		fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		project_k!(b, bk, sortk, sortidx, k, p=p)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		update_xb!(xb, x, b, sortk, k, p=p, n=n)

		# recalculate objective 
		new_obj = compute_loglik(y,x,b, n=n, xb=xb, center=false)

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step
end






######################
### MAIN FUNCTIONS ###
######################


# L0 PENALIZED LEAST SQUARES REGRESSION
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
# -- X is the n x p data matrix
# -- Y is the n x 1 continuous response vector
# -- k is the desired size of the support (active set)
#
# Optional Arguments:
# -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to marginals:
#        b = cov(X,Y) / var(X)
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 1000.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tolerance is the global tolerance. Defaults to 1e-4.
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
#		tempki    = zeros(Int,k)        # temporary array of k integers 
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
function L0_reg(X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), b0::DenseArray{Float64,1} = zeros(p), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1}= zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p) )

	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(ArgumentError("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(ArgumentError("Value of global tolerance must exceed machine precision!\n"))

	# initialize return values
	mm_iter::Int       = 0		# number of iterations of L0_reg
	mm_time::Float64   = 0.0	# compute time *within* L0_reg
	next_obj::Float64  = 0.0	# objective value
	next_loss::Float64 = 0.0	# loss function value 

	# initialize floats 
	current_obj::Float64 = Inf      # tracks previous objective function value
	the_norm::Float64    = 0.0      # norm(b - b0)
	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
	mu::Float64          = 0.0	    # Landweber step size, 0 < tau < 2/rho_max^2

	# initialize integers
	i::Int               = 0        # used for iterations in loops
	mu_step::Int         = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged::Bool      = false    # scaled_norm < tolerance?
   
	# update X*beta
	update_xb!(Xb, X, b, indices, k, p=p, n=n)

	# update r and gradient 
#	update_residuals!(r, X, Y, b, xb=Xb, n=n)
    update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
	BLAS.gemv!('T', 1.0, X, r, 0.0, df)

	# update loss and objective
#	next_loss = 0.5 * sumsq(r)
	next_loss = Inf
	next_obj  = next_loss

	# guard against numerical instabilities
	isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

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
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
			end

			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# calculate r piecemeal
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)

			# calculate loss and objective
			next_loss = 0.5 * sumsq(r)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortk=tempki, sortidx=indices, gk=idx)

		# the IHT kernel gives us an updated x*b
		# use it to recompute residuals and gradient 
#		update_residuals!(r, X, Y, b, xb=Xb, n=n)
		update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
		BLAS.gemv!('T', 1.0, X, r, 0.0, df)

		# update loss, objective, and gradient 
		next_loss = 0.5 * sumsq(r)
		next_obj  = next_loss

		# guard against numerical instabilities
		isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
		isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tolerance
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update r
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)

			# calculate objective
			next_loss = 0.5 * sumsq(r)
			
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
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tolerance
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}

			return output
		end
	end # end main loop
end # end function


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
# -- tolerance is the global tolerance. Defaults to 1e-4.
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
#		tempki    = zeros(Int,k)        # temporary array of k integers 
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
#function L0_reg(X::BEDFile, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), b0::DenseArray{Float64,1} = zeros(p), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1}= zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p) )
#function L0_reg(X::BEDFile, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), b0::DenseArray{Float64,1} = zeros(p), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1}= zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p), means::DenseArray{Float64,1} = mean(X), invstds::DenseArray{Float64,1} = invstd(X, y = means), tempkf2::DenseArray{Float64,1} = zeros(k))
function L0_reg(X::BEDFile, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = SharedArray(Float64, p), tolerance::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = SharedArray(Float64, n,k), r::DenseArray{Float64,1} = SharedArray(Float64, n), Xb::DenseArray{Float64,1} = SharedArray(Float64,n), Xb0::DenseArray{Float64,1} = SharedArray(Float64,n), b0::DenseArray{Float64,1} = SharedArray(Float64,p), df::DenseArray{Float64,1} = SharedArray(Float64,p), tempkf::DenseArray{Float64,1} = SharedArray(Float64,k), idx::DenseArray{Float64,1} = SharedArray(Float64,k), tempn::DenseArray{Float64,1}= SharedArray(Float64,n), indices::DenseArray{Int,1} = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S)), tempki::DenseArray{Int,1} = SharedArray(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p), means::DenseArray{Float64,1} = mean(X, shared=true), invstds::DenseArray{Float64,1} = invstd(X, y = means, shared=true), tempkf2::DenseArray{Float64,1} = SharedArray(Float64,k))


	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(ArgumentError("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(ArgumentError("Value of global tolerance must exceed machine precision!\n"))

	# initialize return values
	mm_iter::Int       = 0		# number of iterations of L0_reg
	mm_time::Float64   = 0.0	# compute time *within* L0_reg
	next_obj::Float64  = 0.0	# objective value
	next_loss::Float64 = 0.0	# loss function value 

	# initialize floats 
	current_obj::Float64 = Inf      # tracks previous objective function value
	the_norm::Float64    = 0.0      # norm(b - b0)
	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
	mu::Float64          = 0.0	    # Landweber step size, 0 < tau < 2/rho_max^2

	# initialize integers
	i::Int               = 0        # used for iterations in loops
	mu_step::Int         = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged::Bool      = false    # scaled_norm < tolerance?
   
	# update Xb, r, and gradient 
	if sum(support) == 0
		fill!(Xb,0.0)
		copy!(r,sdata(Y))
	else
		xb!(Xb,X,b,support,k, means=means, invstds=invstds)
		PLINK.update_partial_residuals!(r, Y, X, support, b, k, Xb=Xb)
	end
	xty!(df, X, r, means=means, invstds=invstds) 


	# update loss and objective
	next_loss = Inf 
	next_obj  = next_loss

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
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
			end

			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# calculate r piecemeal
			PLINK.update_partial_residuals!(r, Y, X, indices, b, k, Xb=Xb)

			# calculate loss and objective
			next_loss = 0.5 * sumsq(sdata(r))

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, Xb=Xb, Xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortk=tempki, sortidx=indices, gk=idx, stdsk=tempkf2) 

		# the IHT kernel gives us an updated x*b
		# use it to recompute residuals and gradient 
		PLINK.update_partial_residuals!(r, Y, X, support, b, k, Xb=Xb)

		xty!(df, X, r, means=means, invstds=invstds) 

		# update loss, objective, and gradient 
		next_loss = 0.5 * sumsq(sdata(r))
		next_obj  = next_loss

		# guard against numerical instabilities
		# ensure that objective is finite
		# if not, throw error
		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tolerance
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update r
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
#			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
			PLINK.update_partial_residuals!(r, Y, X, indices, b, k, Xb=Xb)

			# calculate objective
			next_loss = 0.5 * sumsq(sdata(r))
			
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
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tolerance
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}

			return output
		end
	end # end main loop
end # end function



# L0 PENALIZED LOGISTIC REGRESSION
#
# This routine solves the optimization problem
#
#     min 0.5*|| Y - XB ||_2^2 
#
# subject to
#
#     B in S_k = { x in R^p : || x ||_0 <= k }. 
#
# The algorithm applies a majorization to the centered logistic loglikelihood.
# In doing so, it converts the problem into a least squares problem and applies
# Thomas Blumensath's iterative hard thresholding framework to keep B feasible.
#
# Arguments:
# -- X is the n x p data matrix
# -- Y is the n x 1 continuous response vector
# -- k is the desired size of the support (active set)
#
# Optional Arguments:
# -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to marginals:
#        b = cov(X,Y) / var(X)
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 10000.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tolerance is the global tolerance. Defaults to 1e-4.
# -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
# -- several temporary arrays for intermediate steps of algorithm calculations:
#		Xk        = zeros(Float64,n,k)  # store k columns of X
#		r         = zeros(Float64,n)	# for || Y - XB ||_2^2
#		Xb        = zeros(Float64,n)	# X*beta 
#		Xb0       = zeros(Float64,n)	# X*beta0 
#		b0        = zeros(Float64,p)	# previous iterate beta0 
#		df        = zeros(Float64,p)	# (negative) gradient 
#		w         = zeros(Float64,n)	# vector of weights on responses 
#		tempkf    = zeros(Float64,k)    # temporary array of k floats 
#		idx       = zeros(Float64,k)    # another temporary array of k floats 
#		tempn     = zeros(Float64,n)    # temporary array of n floats 
#		indices   = collect(1:p)	    # indices that sort beta 
#		tempki    = zeros(Int,k)        # temporary array of k integers 
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
function L0_log(X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 10000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), w::DenseArray{Float64,1} = ones(n), b0::DenseArray{Float64,1} = zeros(p), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1}= zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p))

	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(error("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(error("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(error("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(error("Value of global tolerance must exceed machine precision!\n"))
    any(!isfinite(b))        || throw(error("Argument b has nonfinite values"))
    any(!isfinite(means))    || throw(error("Argument means has nonfinite values"))
    any(!isfinite(stds))     || throw(error("Argument stds has nonfinite values"))

	# initialize return values
	mm_iter::Int       = 0		# number of iterations of L0_reg
	mm_time::Float64   = 0.0	# compute time *within* L0_reg
	next_obj::Float64  = Inf	# objective value
	next_loss::Float64 = -Inf	# loss function value 

	# initialize floats 
	current_obj::Float64 = Inf      # tracks previous objective function value
	the_norm::Float64    = 0.0      # norm(b - b0)
	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
	mu::Float64          = Inf	    # IHT step size, 0 < tau < 2/rho_max^2

	# initialize integers
	mu_step::Int         = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged::Bool      = false    # scaled_norm < tolerance?
   
	# initialize x*b
	update_xb!(Xb, X, b, indices, k, p=p, n=n)

	# initialize weights
	update_weights!(w, X, b, xb=Xb, n=n, p=p) 

	# initialize weighted r 
	update_residuals!(r, X, Y, b, w, n=n, p=p, xb=Xb)

	# initialize gradient
	BLAS.gemv!('T', 1.0, X, r, 0.0, df)

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
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
			end

			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for calculating loglikelihood 
			update_xb!(Xb, X, b, indices, k, p=p, n=n)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b, n=n, xb=Xb)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortk=tempki, sortidx=indices, gk=idx, step_multiplier=1.0)

#		# recompute x*b
#		BLAS.gemv!('N', 1.0, X, b, 0.0, Xb)

		# recompute weights
		update_weights!(w, X, b, xb=Xb, n=n, p=p) 

		# update weighted r 
		update_residuals!(r, X, Y, b, w, n=n, p=p, xb=Xb)

		# update gradient
		BLAS.gemv!('T', 1.0, X, r, 0.0, df)

		# update loss, objective
		next_loss = compute_loglik(Y,X,b, n=n, xb=Xb)
		next_obj  = -next_loss

		# guard against numerical instabilities
		isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
		isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

		# ensure that objective is finite
		# if not, throw error
		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tolerance
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for loglikelihood 
			update_xb!(Xb, X, b, indices, k, p=p, n=n)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b, n=n, xb=Xb)
			
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
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tolerance
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}

			return output
		end
	end # end main loop
end # end function



# L0 PENALIZED LOGISTIC REGRESSION FOR ENTIRE GWAS
#
# This routine solves the optimization problem
#
#     min 0.5*|| Y - XB ||_2^2 
#
# subject to
#
#     B in S_k = { x in R^p : || x ||_0 <= k }. 
#
# The algorithm applies a majorization to the centered logistic loglikelihood.
# In doing so, it converts the problem into a least squares problem and applies
# Thomas Blumensath's iterative hard thresholding framework to keep B feasible.
#
# Arguments:
# -- X is the BEDFile that contains the compressed n x p design matrix
# -- Y is the n x 1 binary response vector
# -- k is the desired size of the support (active set)
#
# Optional Arguments:
# -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to zeros(p). 
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 10000.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tolerance is the global tolerance. Defaults to 1e-4.
# -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
# -- several temporary arrays for intermediate steps of algorithm calculations:
#		Xk        = zeros(Float64,n,k)  # store k columns of X
#		r         = zeros(Float64,n)	# for || Y - XB ||_2^2
#		Xb        = zeros(Float64,n)	# X*beta 
#		Xb0       = zeros(Float64,n)	# X*beta0 
#		b0        = zeros(Float64,p)	# previous iterate beta0 
#		df        = zeros(Float64,p)	# (negative) gradient 
#		w         = zeros(Float64,n)	# vector of weights on responses 
#		tempkf    = zeros(Float64,k)    # temporary array of k floats 
#		idx       = zeros(Float64,k)    # another temporary array of k floats 
#		tempn     = zeros(Float64,n)    # temporary array of n floats 
#		indices   = collect(1:p)	    # indices that sort beta 
#		tempki    = zeros(Int,k)        # temporary array of k integers 
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
function L0_log(X::BEDFile, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 10000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), w::DenseArray{Float64,1} = ones(n), b0::DenseArray{Float64,1} = zeros(p), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1}= zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p))

	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(error("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(error("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(error("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(error("Value of global tolerance must exceed machine precision!\n"))

	# initialize return values
	mm_iter::Int       = 0		# number of iterations of L0_reg
	mm_time::Float64   = 0.0	# compute time *within* L0_reg
	next_obj::Float64  = Inf	# objective value
	next_loss::Float64 = -Inf	# loss function value 

	# initialize floats 
	current_obj::Float64 = Inf      # tracks previous objective function value
	the_norm::Float64    = 0.0      # norm(b - b0)
	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
	mu::Float64          = Inf	    # IHT step size, 0 < tau < 2/rho_max^2

	# initialize integers
	mu_step::Int         = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged::Bool      = false    # scaled_norm < tolerance?
   
	# initialize x*b
	xb!(Xb,X,b,indices,k)

	# initialize weights
	update_weights!(w, X, b, perm, k, xb=Xb, n=n, p=p) 

	# initialize weighted r 
	update_residuals!(r, X, Y, b, indices, w, k, n=n, p=p, xb=Xb)

	# initialize gradient
	xty!(df, X, r)

	# initialize loss value
	next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)

	# guard against numerical instabilities
	isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
	isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

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
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
			end

			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for calculating loglikelihood 
#			update_xb!(Xb, X, b, indices, k, p=p, n=n)
			xb!(Xb,X,b,indices,k)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortk=tempki, sortidx=indices, gk=idx, step_multiplier=1.0)

		# recompute weights
		update_weights!(w, X, b, perm, k, xb=Xb, n=n, p=p) 

		# initialize weighted r 
		update_residuals!(r, X, Y, b, indices, w, k, n=n, p=p, xb=Xb)

		# initialize gradient
		xty!(df, X, r)

		# update loss, objective
		next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
		next_obj  = -next_loss


		# ensure that objective is finite
		# if not, throw error
		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tolerance
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for loglikelihood 
#			update_xb!(Xb, X, b, indices, k, p=p, n=n)
			xb!(Xb,X,b,indices,k)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
			
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
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tolerance
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}

			return output
		end
	end # end main loop
end # end function



# L0 PENALIZED LOGISTIC REGRESSION WITH NEWTON'S METHOD
#
# This routine solves the optimization problem
#
#     min 0.5*|| Y - XB ||_2^2 
#
# subject to
#
#     B in S_k = { x in R^p : || x ||_0 <= k }. 
#
# The algorithm operates on the uncentered logistic loglikelihood.
# It deviates from Thomas Blumensath's iterative hard thresholding framework slightly.
# Instead of calculating a step size from the least squares surrogate, this function
# will use Newton's method to calculate the step size directly.
#
# Arguments:
# -- X is the n x p data matrix
# -- Y is the n x 1 continuous response vector
# -- k is the desired size of the support (active set)
#
# Optional Arguments:
# -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to zeros(p). 
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 10000.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tolerance is the global tolerance. Defaults to 1e-4.
# -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
# -- several temporary arrays for intermediate steps of algorithm calculations:
#		Xb        = zeros(Float64,n)	# X*beta 
#		Xb0       = zeros(Float64,n)	# X*beta0 
#		b0        = zeros(Float64,p)	# previous iterate beta0 
#		lg        = zeros(Float64,p)	# (negative) gradient 
#		xlg       = zeros(Float64,p)	# X' * (negative) gradient 
#		tempkf    = zeros(Float64,k)    # temporary array of k floats 
#		tempki    = zeros(Int,k)        # temporary array of k integers 
#		lxb       = zeros(Float64,n)    # logistic(XB) 
#		y2        = zeros(Float64,n)    # Y - logistic(XB) 
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
function L0_log2(X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 10000, max_step::Int = 50,  quiet::Bool = true, Xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, X, b), b0::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = collect(1:k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p), lg::DenseArray{Float64,1} = zeros(p), xlg::DenseArray{Float64,1} = zeros(n), lxb::DenseArray{Float64,1} = zeros(n), y2::DenseArray{Float64,1} = zeros(n)) 


	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(error("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(error("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(error("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(error("Value of global tolerance must exceed machine precision!\n"))

	# initialize return values
	mm_iter::Int       = 0		# number of iterations of L0_reg
	mm_time::Float64   = 0.0	# compute time *within* L0_reg
	next_obj::Float64  = Inf	# objective value
	next_loss::Float64 = -Inf	# loss function value 

	# initialize floats 
	current_obj::Float64 = Inf      # tracks previous objective function value
	the_norm::Float64    = 0.0      # norm(b - b0)
	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
	mu::Float64          = 0.0	    # IHT step size, 0 < tau < 2/rho_max^2

	# initialize integers
	mu_step::Int         = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged::Bool      = false    # scaled_norm < tolerance?
   
	# initialize x*b
	update_xb!(Xb, X, b, indices, k, p=p, n=n)

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
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
			end

			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for calculating loglikelihood 
			update_xb!(Xb, X, b, indices, k, p=p, n=n)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b, n=n, xb=Xb, center=false)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht2(b,X,Y,k, current_obj, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, xb=Xb, sortk=tempki, sortidx=indices, lg=lg, xlg=xlg, bk=tempkf, y2=y2, lxb=lxb)

		# update loss, objective
		next_loss = compute_loglik(Y,X,b, n=n, xb=Xb, center=false)
		next_obj  = -next_loss

		# ensure that objective is finite
		# if not, throw error
		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tolerance

		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tolerance to zero
			threshold!(b, tolerance, n=p)

			# update X*b for loglikelihood 
			update_xb!(Xb, X, b, indices, k, p=p, n=n)

			# calculate loglikelihood 
			next_loss = compute_loglik(Y,X,b, n=n, xb=Xb, center=false)
			
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
			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tolerance
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

			output = {"time" => -1.0, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}

			return output
		end
	end # end main loop
end # end function





# COMPUTE AN IHT REGULARIZATION PATH FOR LEAST SQUARES REGRESSION
# This subroutine computes a regularization path for design matrix X and response Y from initial model size k0 to final model size k.
# The default increment on model size is 1. The path can also be warm-started with a vector b.
# This variant requires a calculated path in order to work.
#
# Arguments:
# -- x is the nxp design matrix.
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
function iht_path(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}; b::DenseArray{Float64,1} = zeros(size(x,2)), quiet::Bool = true, max_iter::Int = 1000, max_step::Int = 50, tolerance::Float64 = 1e-4)

	# size of problem?
	const (n,p) = size(x)

	# how many models will we compute?
	const num_models = length(path)			

	# preallocate space for intermediate steps of algorithm calculations 
	r          = zeros(Float64,n)		# for || Y - XB ||_2^2
	Xb         = zeros(Float64,n)		# X*beta 
	Xb0        = zeros(Float64,n)		# X*beta0 
	b          = zeros(Float64,p)		# model 
	b0         = zeros(Float64,p)		# previous iterate beta0 
	df         = zeros(Float64,p)		# (negative) gradient 
	tempn      = zeros(Float64,n)   	# temporary array of n floats 
	indices    = collect(1:p)	    	# indices that sort beta 
	support    = falses(p)				# indicates nonzero components of beta
	support0   = copy(support)			# store previous nonzero indicators
	betas      = zeros(p,num_models)	# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk      = zeros(q)
		sortk   = selectperm!(indices, b,q, p=p)
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk

		# these arrays change in size from iteration to iteration
		# we must allocate them for every new model size
		Xk     = zeros(Float64,n,q)  # store q columns of X
		tempkf = zeros(Float64,q)    # temporary array of q floats 
		idx    = zeros(Float64,q)    # another temporary array of q floats 
		tempki = zeros(Int,q)        # temporary array of q integers 

		# store projection of beta onto largest k nonzeroes in magnitude 
#		project_k!(b,tempkf,tempki,indices,q, p=p)

#		println("Any NaN in b? ", any(isnan(b)))

		# now compute current model
		output = L0_reg(x,y,q, n=n, p=p, b=b, tolerance=tolerance, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, tempki=tempki, support=support, support0=support0) 

		# extract and save model
		copy!(b, output["beta"])
		update_col!(betas, b, i, n=p, p=num_models, a=1.0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)
end	



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
function iht_path(x::BEDFile, y::DenseArray{Float64,1}, path::DenseArray{Int,1}; b::DenseArray{Float64,1} = ifelse(typeof(y) == SharedArray{Float64,1}, SharedArray(Float64, size(x,2)), zeros(size(x,2))), quiet::Bool = true, max_iter::Int = 1000, max_step::Int = 50, tolerance::Float64 = 1e-4, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x,y=means))

	# size of problem?
	const (n,p) = size(x)

	# how many models will we compute?
	const num_models = length(path)			

	# preallocate SharedArrays for intermediate steps of algorithm calculations 
	r          = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# for || Y - XB ||_2^2
	Xb         = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# X*beta 
	Xb0        = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# X*beta0 
	b0         = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# previous iterate beta0 
	df         = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# (negative) gradient 
	tempn      = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)	   	# temporary array of n floats 

	# index vector for b has more complicated initialization
	indices    = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S))

	# allocate the BitArrays for indexing in IHT
	# also preallocate matrix to store betas 
	support    = falses(p)				# indicates nonzero components of beta
	support0   = copy(support)			# store previous nonzero indicators
	betas      = zeros(p,num_models)	# a matrix to store calculated models

	# compute the path
	@inbounds for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk      = zeros(q)
		sortk   = selectperm!(indices, b,q, p=p)
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk

		# these arrays change in size from iteration to iteration
		# we must allocate them for every new model size
		Xk     = SharedArray(Float64, n, q, init = S -> S[localindexes(S)] = 0.0)	# store q columns of X
		tempkf = SharedArray(Float64, q,    init = S -> S[localindexes(S)] = 0.0)   # temporary array of q floats 
		idx    = SharedArray(Float64, q,    init = S -> S[localindexes(S)] = 0.0)	# another temporary array of q floats 
		tempki = SharedArray(Int,     q,    init = S -> S[localindexes(S)] = 0.0)	# temporary array of q integers 

		# store projection of beta onto largest k nonzeroes in magnitude 
#		project_k!(b,tempkf,tempki,indices,q, p=p)

		# ensure that we correctly index the nonzeroes in b
		support = b .!= 0.0
		copy!(support0, support)

		# now compute current model
		output = L0_reg(x,y,q, n=n, p=p, b=b, tolerance=tolerance, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, tempki=tempki, support=support, support0=support0, means=means, invstds=invstds) 

		# extract and save model
		copy!(sdata(b), output["beta"])
		update_col!(betas, sdata(b), i, n=p, p=num_models, a=1.0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)
end	


# COMPUTE AN IHT REGULARIZATION PATH FOR LOGISTIC REGRESSION
# This subroutine computes a regularization path for design matrix X and response Y from initial model size k0 to final model size k.
# The path can also be warm-started with a vector b.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses
# -- path is an Int array that contains the model sizes to test
#
# Optional Arguments:
# -- b is the p-vector of effect sizes. This argument permits warmstarts to the path computation. Defaults to zeros.
# -- max_iter caps the number of iterations for the algorithm. Defaults to 10000.
# -- max_step caps the number of backtracking steps in the IHT kernel. Defaults to 50.
# -- quiet is a Boolean that controls the output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function iht_path_log(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}; b::DenseArray{Float64,1} = zeros(size(x,2)), quiet::Bool = true, max_iter::Int = 10000, max_step::Int = 50, tolerance::Float64 = 1e-4)

	# size of problem?
	const (n,p) = size(x)

	# preallocate space for intermediate steps of algorithm calculations 
	r         = zeros(Float64,n)	# for || Y - XB ||_2^2
	w         = zeros(Float64,n)    # vector of weights 
	Xb        = zeros(Float64,n)	# X*beta 
	Xb0       = zeros(Float64,n)	# X*beta0 
	b         = zeros(Float64,p)	# model 
	b0        = zeros(Float64,p)	# previous iterate beta0 
	df        = zeros(Float64,p)	# (negative) gradient 
	tempn     = zeros(Float64,n)    # temporary array of n floats 
	indices   = collect(1:p)	    # indices that sort beta 
	support   = falses(p)			# indicates nonzero components of beta
	support0  = copy(support)		# store previous nonzero indicators

	# preallocate space for path computation
	num_models = length(path)			# how many models will we compute?
	betas      = zeros(p,num_models)	# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk       = zeros(q)
		sortk    = selectperm!(indices, b,q, p=p) 
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
		

		# other variables change in size, so allocate them at every new model
		Xk     = zeros(Float64,n,q)  # store q columns of X
		tempkf = zeros(Float64,q)    # temporary array of q floats 
		idx    = zeros(Float64,q)    # another temporary array of q floats 
		tempki = zeros(Int,q)        # temporary array of q integers 

		# store projection of beta onto largest k nonzeroes in magnitude 
#		project_k!(b,tempkf,indices,q, p=p)

		# now compute current model
		output = L0_log(x,y,path[i], n=n, p=p, b=b, tolerance=tolerance, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, w=w, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, tempki=tempki, support=support, support0=support0) 

		# extract and save model
		copy!(b, output["beta"])
		update_col!(betas, b, i, n=p, p=num_models, a=1.0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)
end	# end iht_path_log	





# COMPUTE AN IHT REGULARIZATION PATH FOR LOGISTIC REGRESSION
# This subroutine computes a regularization path for design matrix X and response Y from initial model size k0 to final model size k.
# The path can also be warm-started with a vector b.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses
# -- path is an Int array that contains the model sizes to test
#
# Optional Arguments:
# -- b is the p-vector of effect sizes. This argument permits warmstarts to the path computation. Defaults to zeros.
# -- max_iter caps the number of iterations for the algorithm. Defaults to 10000.
# -- max_step caps the number of backtracking steps in the IHT kernel. Defaults to 50.
# -- quiet is a Boolean that controls the output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function iht_path_log2(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}; b::DenseArray{Float64,1} = zeros(size(x,2)), quiet::Bool = true, max_iter::Int = 10000, max_step::Int = 50, tolerance::Float64 = 1e-4)

	# size of problem?
	const (n,p) = size(x)

	# preallocate space for intermediate steps of algorithm calculations 
	Xb        = zeros(Float64,n)	# X*beta 
	Xb0       = zeros(Float64,n)	# X*beta0 
	b         = zeros(Float64,p)	# model 
	b0        = zeros(Float64,p)	# previous iterate beta0 
	indices   = collect(1:p)	    # indices that sort beta 
	support   = falses(p)			# indicates nonzero components of beta
	support0  = copy(support)		# store previous nonzero indicators
	lxb       = zeros(p)			# logistic(Xbeta)
	y2        = zeros(n)			# y - logistic(Xbeta)
	lg        = zeros(p)			# (negative) gradient of loglikelihood
	xlg       = zeros(n)			# X' * lg

	# preallocate space for path computation
	const num_models = length(path)	# how many models will we compute?
	betas = zeros(p,num_models)		# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk       = zeros(q)
		sortk    = selectperm!(indices, b,q, p=p)
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk

		# other variables change in size, so allocate them at every new model
		tempkf = zeros(Float64,q)    # temporary array of q floats 
		tempki = zeros(Int,q)        # temporary array of q integers 

		# store projection of beta onto largest k nonzeroes in magnitude 
#		project_k!(b,tempkf,indices,q, p=p)

		# now compute current model
		output = L0_log2(x, y, path[i], n=n, p=p, b=b, tolerance=tolerance, max_iter=max_iter, quiet=quiet, Xb=Xb, b0=b0, tempkf=tempkf, indices=indices, tempki=tempki, support=support, support0=support0, lg=lg, xlg=xlg, lxb=lxb)

		# extract and save model
		copy!(b, output["beta"])
		update_col!(betas, b, i, n=p, p=num_models, a=1.0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)

end # end iht_path_log2	

################################
### CROSSVALIDATION ROUTINES ###
################################


# COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A REGULARIZATION PATH
#
# For a regularization path given by the vector "path", 
# this function computes an out-of-sample error based on the indices given in the vector "test_idx". 
# The vector test_idx indicates the portion of the data to use for testing.
# The remaining data are used for training the model.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses.
# -- path is the Int array that indicates the model sizes to compute on the regularization path. 
#
# -- path is an integer array that specifies which model sizes to include in the path, e.g.
#    > path = collect(k0:increment:k_end).
# -- test_idx is the Int array that indicates which data to hold out for testing.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- max_step caps the number of permissible backtracking steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
# -- logreg is a switch to activate logistic regression. Defaults to false (perform linear regression).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, test_idx::DenseArray{Int,1}; max_iter::Int = 1000, max_step::Int = 50, quiet::Bool = true, logreg::Bool = false) 
function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, folds::DenseArray{Int,1}, fold::Int; max_iter::Int = 1000, max_step::Int = 50, quiet::Bool = true, logreg::Bool = false) 

	# make vector of indices for folds
#	test_idx  = find( function f(x) x .== fold; end, folds)
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
#	train_idx = setdiff(collect(1:n), test_idx)
	train_idx = !test_idx

	# allocate the arrays for the training set
	x_train   = x[train_idx,:]
	y_train   = y[train_idx] 

	if logreg
		# compute the regularization path on the training set
		betas     = iht_path_log(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step) 

		# compute the mean out-of-sample error for the TEST set 
		myerrors  = vec(sumsq(broadcast(-, round(y[test_idx]), round(logistic(x[test_idx,:] * betas))), 1)) ./ length(test_idx)
	else
		# compute the regularization path on the training set
		betas     = iht_path(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step) 

		# compute the mean out-of-sample error for the TEST set 
		myerrors  = vec(sumsq(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ length(test_idx)
	end

	return myerrors
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
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- max_step caps the number of permissible backtracking steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
# -- logreg is a switch to activate logistic regression. Defaults to false (perform linear regression).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, test_idx::DenseArray{Int,1}; max_iter::Int = 1000, max_step::Int = 50, quiet::Bool = true, logreg::Bool = false) 
function one_fold(x::BEDFile, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, folds::DenseArray{Int,1}, fold::Int; max_iter::Int = 1000, max_step::Int = 50, quiet::Bool = true, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x,y=means)) 

	# make vector of indices for folds
#	test_idx  = find( function f(x) x .== fold; end, folds)
	test_idx = folds .== fold
	test_size = sum(test_idx)

	# preallocate vector for output
	myerrors = zeros(test_size)

	# train_idx is the vector that indexes the TRAINING set
	train_idx = !test_idx

	# allocate the arrays for the training set
	x_train = x[train_idx,:]
	y_train = y[train_idx] 
	Xb      = SharedArray(Float64, test_size) 
	b       = SharedArray(Float64, x.p) 
	r       = SharedArray(Float64, test_size) 
	perm    = collect(1:test_size) 

	# compute the regularization path on the training set
	betas = iht_path(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step, means=means, invstds=invstds) 

	# compute the mean out-of-sample error for the TEST set 
	@inbounds for i = 1:test_size
#		RegressionTools.update_col!(b,betas,i,n=x.p,p=test_size)
		b2 = vec(full(betas[:,i]))
		copy!(b,b2)
		xb!(Xb,x_test,b, means=means, invstds=invstds)
		PLINK.update_partial_residuals!(r,y_train,x_train,perm,b,test_size, Xb=Xb)
		myerrors[i] = sumsq(r) / test_size
	end

	return myerrors
end

# CREATE UNSTRATIFIED CROSSVALIDATION PARTITION
# This function will partition n indices into k disjoint sets for k-fold crossvalidation
# Arguments:
# -- n is the dimension of the data set to partition.
# -- k is the number of disjoint sets in the partition.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function cv_get_folds(y::Vector, nfolds::Int)
	n, r = divrem(length(y), nfolds)
	shuffle!([repmat(1:nfolds, n), 1:r])
end

# PARALLEL CROSSVALIDATION ROUTINE FOR IHT
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
# -- x is the nxp design matrix.
# -- y is the n-vector of responses.
# -- path is an integer array that specifies which model sizes to include in the path, e.g.
#    > path = collect(k0:increment:k_end).
# -- nfolds is the number of folds to compute.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to size(x,2).
# -- folds is the partition of the data. Defaults to a random partition into "nfolds" disjoint sets.
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
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
function cv_iht(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, numfolds::Int; n::Int = length(y), p::Int = size(x,2), tol::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50, quiet=true, folds::DenseArray{Int,1} = cv_get_folds(sdata(y),numfolds), logreg::Bool = false, compute_model::Bool = true) 

	# how many elements are in the path?
	const num_models = length(path)

	# preallocate vectors used in xval	
	errors  = zeros(num_models)		# vector to save mean squared errors
	my_refs = cell(numfolds)		# cell array to store RemoteRefs

#	# ensure that y is standardized
#	m = mean(y)
#	s = std(y)
#	y = (y - m) / s

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds
		# test_idx saves the numerical identifier of the vector of indices corresponding to the ith fold 
		# this vector will indiate which part of the data to hold out for testing 
#		test_idx  = folds[i]

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
#		my_refs[i] = @spawn(one_fold(x, y, path, folds, test_idx, n=n, max_iter=max_iter, max_step=max_step, quiet=quiet, logreg=logreg))
		my_refs[i] = @spawn(one_fold(x, y, path, folds, i, max_iter=max_iter, max_step=max_step, quiet=quiet, logreg=logreg))
	end
	
	# recover MSEs on each worker
	@inbounds @simd for i = 1:numfolds
		errors += fetch(my_refs[i])
	end

	# average the mses
	errors ./= numfolds

	# what is the best model size?
#	println("Best model size is ", floor(mean(path[mses .== minimum(mses)])))
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
		b = zeros(p)
		if logreg
			
			# can preallocate some of the temporary arrays for use in both model selection and fitting
			# notice that they all depend on n, which is fixed,
			# as opposed to p, which changes depending on the number of nonzeroes in b
			xb   = zeros(n)      # xb = x*b 
			lxb  = zeros(n)      # logistic(xb), which we call pi 
			l2xb = zeros(n)      # logistic(xb) [ 1 - logistic(xb) ], or pi(1 - pi)

			# first use L0_reg to extract model
			output = L0_log(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tolerance=tol, Xb=xb, Xb0=xlb, r=l2xb)
			copy!(b, output["beta"])

			# which components of beta are nonzero?
			bidx = find( x -> x .!= 0.0, b) 

			# allocate the submatrix of x corresponding to the inferred model
			x_inferred = x[:,bidx]

#			# unstandardize y
#			y = s*y + m 

			# compute logistic fit
			b2 = fit_logistic(x_inferred, y, xb=xb, lxb=lxb, l2xb=l2xb)	
		else

			# first use L0_reg to extract model
			output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tolerance=tol)
			copy!(b, output["beta"])

			# which components of beta are nonzero?
			bidx = find( x -> x .!= 0.0, b) 

			# allocate the submatrix of x corresponding to the inferred model
			x_inferred = x[:,bidx]

#			# unstandardize y
#			y = s*y + m 

			# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
			xty = BLAS.gemv('T', 1.0, x_inferred, y)	
			xtx = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
			b2 = xtx \ xty
		end
		return errors, b2, bidx 
	end
	return errors
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
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
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
function cv_iht(x::BEDFile, y::DenseArray{Float64,1}, path::DenseArray{Int,1}, numfolds::Int; n::Int = length(y), p::Int = size(x,2), tol::Float64 = 1e-4, max_iter::Int = 1000, max_step::Int = 50, quiet=true, folds::DenseArray{Int,1} = cv_get_folds(sdata(y),numfolds), compute_model::Bool = false, means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x,y=means)) 

	# how many elements are in the path?
	const num_models = length(path)

	# preallocate vectors used in xval	
	errors  = zeros(num_models)		# vector to save mean squared errors
	my_refs = cell(numfolds)		# cell array to store RemoteRefs

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds
		# test_idx saves the numerical identifier of the vector of indices corresponding to the ith fold 
		# this vector will indiate which part of the data to hold out for testing 
#		test_idx  = folds[i]

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
#		my_refs[i] = @spawn(one_fold(x, y, path, folds, test_idx, n=n, max_iter=max_iter, max_step=max_step, quiet=quiet, logreg=logreg))
		my_refs[i] = @spawn(one_fold(x, y, path, folds, i, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds)) 
	end
	
	# recover MSEs on each worker
	@inbounds @simd for i = 1:numfolds
		errors += fetch(my_refs[i])
	end

	# average the mses
	errors ./= numfolds

	# what is the best model size?
#	println("Best model size is ", floor(mean(path[mses .== minimum(mses)])))
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
		b = SharedArray(Float64, p)
		# first use L0_reg to extract model
		output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tolerance=tol)

		# which components of beta are nonzero?
		inferred_model = output["beta"] .!= 0.0
		bidx = find( x -> x .!= 0.0, b) 

		# allocate the submatrix of x corresponding to the inferred model
		x_inferred = zeros(n,sum(inferred_model))
		decompress_genotypes!(x_inferred,x)

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		xty = BLAS.gemv('T', 1.0, x_inferred, y)	
		xtx = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
		b = xtx \ xty
		return errors, b, bidx 
	end
	return errors
end


end # end module IHT
