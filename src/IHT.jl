module IHT

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
#using StatsBase: sample, logistic
using RegressionTools
using OpenCL
using Compat

export L0_reg
export L0_reg_aiht
export L0_log
export L0_log2
export iht_path
export iht_path_log
export iht_path_log2
export cv_iht
export cv_get_folds
export L0_reg_gpu


include("aiht.jl")
include("aiht32.jl")
include("crossvalidation.jl")
include("crossvalidation32.jl")
include("gwas.jl")
include("gwas32.jl")
#include("logistic.jl")
#include("logistic32.jl")
include("gpu32.jl")

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
# -- betak is a vector to store the largest k values of beta. Defaults to k zeros of type Float64. 
# -- IDX and IDX0 are BitArrays indicating the nonzero status of components of beta. They default to falses.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
# based on the HardLab demonstration code written in MATLAB by Thomas Blumensath
# http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html 
function iht(
	b         :: DenseArray{Float64,1}, 
	x         :: DenseArray{Float64,2}, 
	y         :: DenseArray{Float64,1}, 
	k         :: Int, 
	g         :: DenseArray{Float64,1}; 
	n         :: Int                   = length(y), 
	p         :: Int                   = length(b), 
	xk        :: DenseArray{Float64,2} = zeros(Float64,n,k), 
	b0        :: DenseArray{Float64,1} = copy(b), 
	xb0       :: DenseArray{Float64,1} = copy(xb), 
	xb        :: DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), 
	xgk       :: DenseArray{Float64,1} = zeros(Float64,n), 
	gk        :: DenseArray{Float64,1} = zeros(Float64,k), 
	bk        :: DenseArray{Float64,1} = zeros(Float64,k), 
	sortidx   :: DenseArray{Int,1}     = collect(1:p), 
	IDX       :: BitArray{1}           = falses(p), 
	IDX0      :: BitArray{1}           = copy(IDX), 
#	step_mult :: Float64               = 1.0,
	iter      :: Int                   = 0,
	max_step  :: Int                   = 50
) 

	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		selectperm!(sortidx,g,k, p=p) 
		IDX[sortidx[1:k]] = true;
	end

	# store relevant columns of x
	# need to do this on 1st iteration
	# afterwards, only do if support changes
	if !isequal(IDX,IDX0) || iter < 2
		update_xk!(xk, x, IDX, k=k, p=p, n=n)	# xk = x[:,IDX]
	end

	# store relevant components of gradient
	fill_perm!(gk, g, IDX, k=k, p=p)	# gk = g[IDX]

	# now compute subset of x*g
	BLAS.gemv!('N', 1.0, xk, gk, 0.0, xgk)

	# compute step size
#	mu = step_mult * sumabs2(sdata(gk)) / sumsq(sdata(xgk))
	mu = sumabs2(sdata(gk)) / sumabs2(sdata(xgk))
	isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))

	# take gradient step
	BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

	# preserve top k components of b
	project_k!(b, bk, sortidx, k)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	update_xb!(xb, x, b, sortidx, k)

	# calculate omega
	omega_top = sqeuclidean(sdata(b),(b0))
	omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

	# backtrack until mu sits below omega and support stabilizes
	mu_step = 0
	while mu*omega_bot > 0.99*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

		# recompute projection onto top k components of b
		project_k!(b, bk, sortidx, k)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		update_xb!(xb, x, b, sortidx, k)

		# calculate omega
		omega_top = sqeuclidean(sdata(b),(b0))
		omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step
end



function iht(
	b         :: DenseArray{Float32,1}, 
	x         :: DenseArray{Float32,2}, 
	y         :: DenseArray{Float32,1}, 
	k         :: Int, 
	g         :: DenseArray{Float32,1}; 
	n         :: Int                   = length(y), 
	p         :: Int                   = length(b), 
	xk        :: DenseArray{Float32,2} = zeros(Float32,n,k), 
	b0        :: DenseArray{Float32,1} = copy(b), 
	xb0       :: DenseArray{Float32,1} = copy(xb), 
	xb        :: DenseArray{Float32,1} = BLAS.gemv('N', 1.0f0, x, b), 
	xgk       :: DenseArray{Float32,1} = zeros(Float32,n), 
	gk        :: DenseArray{Float32,1} = zeros(Float32,k), 
	bk        :: DenseArray{Float32,1} = zeros(Float32,k), 
	sortidx   :: DenseArray{Int,1}     = collect(1:p), 
	IDX       :: BitArray{1}           = falses(p), 
	IDX0      :: BitArray{1}           = copy(IDX), 
#	step_mult :: Float32               = 1.0f0,
	iter      :: Int                   = 0,
	max_step  :: Int                   = 50
) 

	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		selectperm!(sortidx,g,k, p=p) 
		IDX[sortidx[1:k]] = true;
	end

	# store relevant columns of x
	# need to do this on 1st iteration
	# afterwards, only do if support changes
	if !isequal(IDX,IDX0) || iter < 2
		update_xk!(xk, x, IDX, k=k, p=p, n=n)	# xk = x[:,IDX]
	end

	# store relevant components of gradient
	fill_perm!(gk, g, IDX, k=k, p=p)	# gk = g[IDX]

	# now compute subset of x*g
	BLAS.gemv!('N', one(Float32), xk, gk, zero(Float32), xgk)

	# compute step size
#	mu = step_mult * sumabs2(sdata(gk)) / sumsq(sdata(xgk))
	mu = sumabs2(sdata(gk)) / sumabs2(sdata(xgk))
	isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))

	# take gradient step
	BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

	# preserve top k components of b
	project_k!(b, bk, sortidx, k)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	update_xb!(xb, x, b, sortidx, k)

	# calculate omega
	omega_top = sqeuclidean(sdata(b),(b0))
	omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

	# backtrack until mu sits below omega and support stabilizes
	mu_step = 0
	while mu*omega_bot > 0.99*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5f0

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

		# recompute projection onto top k components of b
		project_k!(b, bk, sortidx, k)

		# which indices of new beta are nonzero?
		update_indices!(IDX, b, p=p) 

		# recompute xb
		update_xb!(xb, x, b, sortidx, k)

		# calculate omega
		omega_top = sqeuclidean(sdata(b),(b0))
		omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

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
# -- tol is the global tol. Defaults to 1e-4.
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
	X         :: DenseArray{Float64,2}, 
	Y         :: DenseArray{Float64,1}, 
	k         :: Int; 
	n         :: Int                   = length(Y), 
	p         :: Int                   = size(X,2), 
	Xk        :: DenseArray{Float64,2} = zeros(Float64,n,k), 
	b         :: DenseArray{Float64,1} = zeros(Float64,p), 
	b0        :: DenseArray{Float64,1} = zeros(Float64,p), 
	df        :: DenseArray{Float64,1} = zeros(Float64,p), 
	r         :: DenseArray{Float64,1} = zeros(Float64,n), 
	Xb        :: DenseArray{Float64,1} = zeros(Float64,n), 
	Xb0       :: DenseArray{Float64,1} = zeros(Float64,n), 
	tempn     :: DenseArray{Float64,1} = zeros(Float64,n), 
	tempkf    :: DenseArray{Float64,1} = zeros(Float64,k), 
	idx       :: DenseArray{Float64,1} = zeros(Float64,k), 
	indices   :: DenseArray{Int,1}     = collect(1:p), 
	support   :: BitArray{1}           = falses(p), 
	support0  :: BitArray{1}           = falses(p), 
	tol       :: Float64               = 1e-4, 
	max_iter  :: Int                   = 1000, 
	max_step  :: Int                   = 50,  
	quiet     :: Bool                  = true
)

	# start timer
	tic()

	# first handle errors
	k        >= 0     || throw(ArgumentError("Value of k must be nonnegative!\n"))
	max_iter >= 0     || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
	max_step >= 0     || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
	tol      >  eps() || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

	# initialize return values
	mm_iter   = 0       # number of iterations of L0_reg
	mm_time   = 0.0	    # compute time *within* L0_reg
	next_obj  = 0.0	    # objective value
	next_loss = 0.0	    # loss function value 

	# initialize floats 
	current_obj = Inf   # tracks previous objective function value
	the_norm    = 0.0   # norm(b - b0)
	scaled_norm = 0.0   # the_norm / (norm(b0) + 1)
	mu          = 0.0	# Landweber step size, 0 < tau < 2/rho_max^2

	# initialize integers
	i       = 0         # used for iterations in loops
	mu_step = 0         # counts number of backtracking steps for mu

	# initialize booleans
	converged = false   # scaled_norm < tol?
   
	# update X*beta
	update_xb!(Xb, X, b, indices, k, p=p, n=n)

	# update r and gradient 
	difference!(r,Y,Xb, n=n)
	BLAS.gemv!('T', 1.0, X, r, 0.0, df)

	# update loss and objective
#	next_loss = 0.5 * sumabs2(r)
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

			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# calculate r piecemeal
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
#			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
#			difference!(r,Y,Xb, n=n)

			# calculate loss and objective
#			next_loss = 0.5 * sumabs2(r)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
			@compat output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortidx=indices, gk=idx, iter=mm_iter)

		# the IHT kernel gives us an updated x*b
		# use it to recompute residuals and gradient 
		difference!(r,Y,Xb, n=n)
		BLAS.gemv!('T', 1.0, X, r, 0.0, df)

		# update loss, objective, and gradient 
		next_loss = 0.5 * sumabs2(r)
		next_obj  = next_loss

		# guard against numerical instabilities
		isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
		isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tol
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# update r
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
#			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
#			difference!(r,Y,Xb)

			# calculate objective
#			next_loss = 0.5 * sumabs2(r)
			
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
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
			@compat output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tol
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

#			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}
			@compat output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

			return output
		end
	end # end main loop
end # end function



function L0_reg(
	X         :: DenseArray{Float32,2}, 
	Y         :: DenseArray{Float32,1}, 
	k         :: Int; 
	n         :: Int                   = length(Y), 
	p         :: Int                   = size(X,2), 
	Xk        :: DenseArray{Float32,2} = zeros(Float32,n,k), 
	b         :: DenseArray{Float32,1} = zeros(Float32,p), 
	b0        :: DenseArray{Float32,1} = zeros(Float32,p), 
	df        :: DenseArray{Float32,1} = zeros(Float32,p), 
	r         :: DenseArray{Float32,1} = zeros(Float32,n), 
	Xb        :: DenseArray{Float32,1} = zeros(Float32,n), 
	Xb0       :: DenseArray{Float32,1} = zeros(Float32,n), 
	tempn     :: DenseArray{Float32,1} = zeros(Float32,n), 
	tempkf    :: DenseArray{Float32,1} = zeros(Float32,k), 
	idx       :: DenseArray{Float32,1} = zeros(Float32,k), 
	indices   :: DenseArray{Int,1}     = collect(1:p), 
	support   :: BitArray{1}           = falses(p), 
	support0  :: BitArray{1}           = falses(p), 
	tol       :: Float32               = 1f-4, 
	max_iter  :: Int                   = 1000, 
	max_step  :: Int                   = 50,  
	quiet     :: Bool                  = true
)

	# start timer
	tic()

	# first handle errors
	k        >= 0            || throw(ArgumentError("Value of k must be nonnegative!\n"))
	max_iter >= 0            || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
	max_step >= 0            || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
	tol      >  eps(Float32) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

	# initialize return values
	mm_iter   = 0	    # number of iterations of L0_reg
	mm_time   = 0.0f0	# compute time *within* L0_reg
	next_obj  = 0.0f0	# objective value
	next_loss = 0.0f0	# loss function value 

	# initialize floats 
	current_obj = Inf32	# tracks previous objective function value
	the_norm    = 0.0f0 # norm(b - b0)
	scaled_norm = 0.0f0 # the_norm / (norm(b0) + 1)
	mu          = 0.0f0 # Landweber step size, 0 < tau < 2/rho_max^2

	# initialize integers
	i       = 0        # used for iterations in loops
	mu_step = 0        # counts number of backtracking steps for mu

	# initialize booleans
	converged = false    # scaled_norm < tol?
   
	# update X*beta
	update_xb!(Xb, X, b, indices, k, p=p, n=n)

	# update r and gradient 
	difference!(r,Y,Xb, n=n)
	BLAS.gemv!('T', 1.0f0, X, r, 0.0f0, df)

	# update loss and objective
#	next_loss = 0.5 * sumabs2(r)
	next_loss = Inf32
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

			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# calculate r piecemeal
#			difference!(r,Y,Xb, n=n)

			# calculate loss and objective
#			next_loss = 0.5 * sumabs2(r)

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
			@compat output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortidx=indices, gk=idx, iter=mm_iter)

		# the IHT kernel gives us an updated x*b
		# use it to recompute residuals and gradient 
#		update_residuals!(r, X, Y, b, xb=Xb, n=n)
#		update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
		difference!(r,Y,Xb, n=n)
		BLAS.gemv!('T', 1.0f0, X, r, 0.0f0, df)

		# update loss, objective, and gradient 
		next_loss = 0.5f0 * sumabs2(r)
		next_obj  = next_loss

		# guard against numerical instabilities
		isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
		isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
		converged   = scaled_norm < tol
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

		# check for convergence
		# if converged and in feasible set, then algorithm converged before maximum iteration
		# perform final computations and output return variables 
		if converged
			
			# send elements below tol to zero
			threshold!(b, tol, n=p)

			# update r
#			update_residuals!(r, X, Y, b, xb=Xb, n=n)
#			update_partial_residuals!(r, Y, X, indices, b, k, n=n, p=p)
#			difference!(r,Y,Xb)

			# calculate objective
#			next_loss = 0.5 * sumabs2(r)
			
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
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
			@compat output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end

		# algorithm is unconverged at this point.
		# if algorithm is in feasible set, then rho should not be changing
		# check descent property in that case
		# if rho is not changing but objective increases, then abort
		if next_obj > current_obj + tol
			if !quiet
				print_with_color(:red, "\nMM algorithm fails to descend!\n")
				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
				print_with_color(:red, "Current Objective: $(current_obj)\n") 
				print_with_color(:red, "Next Objective: $(next_obj)\n") 
				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
			end

#			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}
			@compat output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

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
function iht_path(
	x        :: DenseArray{Float64,2}, 
	y        :: DenseArray{Float64,1}, 
	path     :: DenseArray{Int,1}; 
	b        :: DenseArray{Float64,1} = zeros(Float64,size(x,2)), 
	tol      :: Float64               = 1e-4,
	max_iter :: Int                   = 1000, 
	max_step :: Int                   = 50, 
	quiet    :: Bool                  = true 
)

	# size of problem?
	const (n,p) = size(x)

	# how many models will we compute?
	const num_models = length(path)			

	# preallocate space for intermediate steps of algorithm calculations 
	r          = zeros(Float64,n)				# for || Y - XB ||_2^2
	Xb         = zeros(Float64,n)				# X*beta 
	Xb0        = zeros(Float64,n)				# X*beta0 
	b          = zeros(Float64,p)				# model 
	b0         = zeros(Float64,p)				# previous iterate beta0 
	df         = zeros(Float64,p)				# (negative) gradient 
	tempn      = zeros(Float64,n)   			# temporary array of n floats 
	indices    = collect(1:p)	    			# indices that sort beta 
	support    = falses(p)						# indicates nonzero components of beta
	support0   = copy(support)					# store previous nonzero indicators
	betas      = zeros(Float64,p,num_models)	# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk      = zeros(Float64,q)
		project_k!(b, bk, indices, q)

		# these arrays change in size from iteration to iteration
		# we must allocate them for every new model size
		Xk     = zeros(Float64,n,q)  # store q columns of X
		tempkf = zeros(Float64,q)    # temporary array of q floats 
		idx    = zeros(Float64,q)    # another temporary array of q floats 

		# now compute current model
		output = L0_reg(x,y,q, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, support=support, support0=support0) 

		# extract and save model
		copy!(b, output["beta"])
		update_col!(betas, b, i, n=p, p=num_models, a=1.0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)
end	


function iht_path(
	x        :: DenseArray{Float32,2}, 
	y        :: DenseArray{Float32,1}, 
	path     :: DenseArray{Int,1}; 
	b        :: DenseArray{Float32,1} = zeros(Float32,size(x,2)), 
	tol      :: Float32               = 1f-4,
	max_iter :: Int                   = 1000, 
	max_step :: Int                   = 50, 
	quiet    :: Bool                  = true 
)

	# size of problem?
	const (n,p) = size(x)

	# how many models will we compute?
	const num_models = length(path)			

	# preallocate space for intermediate steps of algorithm calculations 
	r          = zeros(Float32,n)				# for || Y - XB ||_2^2
	Xb         = zeros(Float32,n)				# X*beta 
	Xb0        = zeros(Float32,n)				# X*beta0 
	b          = zeros(Float32,p)				# model 
	b0         = zeros(Float32,p)				# previous iterate beta0 
	df         = zeros(Float32,p)				# (negative) gradient 
	tempn      = zeros(Float32,n)   			# temporary array of n floats 
	indices    = collect(1:p)	    			# indices that sort beta 
	support    = falses(p)						# indicates nonzero components of beta
	support0   = copy(support)					# store previous nonzero indicators
	betas      = zeros(Float32,p,num_models)	# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# store projection of beta onto largest k nonzeroes in magnitude 
		bk      = zeros(Float32,q)
		project_k!(b, bk, indices, k)

		# these arrays change in size from iteration to iteration
		# we must allocate them for every new model size
		Xk     = zeros(Float32,n,q)  # store q columns of X
		tempkf = zeros(Float32,q)    # temporary array of q floats 
		idx    = zeros(Float32,q)    # another temporary array of q floats 

		# now compute current model
		output = L0_reg(x,y,q, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, support=support, support0=support0) 

		# extract and save model
		copy!(b, output["beta"])
		update_col!(betas, b, i, n=p, p=num_models, a=1.0f0) 
	end

	# return a sparsified copy of the models
	return sparse(betas)
end	

end	# end module IHT
