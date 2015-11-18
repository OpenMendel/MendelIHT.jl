function logistic!(
	y :: Vector{Float64},
	x :: Vector{Float64};
	n :: Int = length(y)
)
	n == length(x) || throw(ArgumentError("length(y) != length(x)"))
	@inbounds for i = 1:n
		y[i] = exp(x[i]) / (one(Float64) + exp(x[i]))
	end
	return nothing
end

function xpypbz!(
	w :: Vector{Float64},
	x :: Vector{Float64},
	y :: Vector{Float64},
	b :: Float64,
	z :: Vector{Float64};
	n :: Int = length(w)
)
	@inbounds for i = 1:n
		w[i] = x[i] + y[i] + b*z[i]
	end
	return nothing
end


function compute_loglik(
	Xb      :: Vector{Float64},
	mxty    :: Vector{Float64},
	b       :: Vector{Float64},
	sortidx :: Vector{Int},
	lambda  :: Float64,
	k       :: Int;
	n       :: length(xlxb)
)
	s = zero(Float64)
	@inbounds for i = 1:n
		s += log(one(Float64) + exp(Xb[i]))
	end
	@inbounds for i = 1:k
		idx = sortidx[i]
		s  += mxty[idx] * b[idx]
	end
	return nothing
end


function loggrad!(
	g       :: DenseVector{Float64},
	Xb      :: DenseVector{Float64},
	lxb     :: DenseVector{Float64},
	xlxb    :: DenseVector{Float64},
	x       :: DenseMatrix{Float64},
	b       :: DenseVector{Float64},
	sortidx :: DenseVector{Int},
	mxty    :: DenseVector{Float64},
	lambda  :: Float64,
	n       :: Int,
	k       :: Int
)
	# first update x*b
	update_xb!(Xb, x, b, sortidx, k, n=n, p=k)

	# then save lxb = logistic(x*b)
	logistic!(lxb, Xb, n=n)

	# xlxb = x' * logistic(x*b) 
	BLAS.gemv!('T', one(Float64), x, lxb, zero(Float64), xlxb)

	# gradient is now xlxb + (- x' * y) + lambda*b
	xpypbz!(g, xlxb, mxty, lambda, b) 
	return nothing
end


function iht(
	b        :: DenseVector{Float64}, 
	x        :: DenseMatrix{Float64}, 
	y        :: DenseVector{Float64}, 
	g        :: DenseVector{Float64},
	k        :: Int, 
	old_obj  :: Float64,
	lambda   :: Float64; 
	n        :: Int = length(y), 
	p        :: Int = length(b), 
	Xb       :: DenseVector{Float64} = BLAS.gemv('N', one(Float64), x, b), 
	mxty     :: DenseVector{Float64} = BLAS.gemv('T', -one(Float64), x, y), 
	xlxb     :: DenseVector{Float64} = zeros(Float64, p), 
	lxb      :: DenseVector{Float64} = zeros(Float64, n), 
	bk       :: DenseVector{Float64} = zeros(Float64, k), 
	sortidx  :: DenseVector{Int}     = collect(1:p), 
	IDX      :: BitArray{1}          = falses(p), 
	IDX0     :: BitArray{1}          = copy(IDX), 
	max_step :: Int                  = 50
)

	### TEMPORARY
	### Rs = max_{i = 1,2,...,p} || x_i ||_2
	Rs = one(Float64)

	# which indices of beta are nonzero?
	# which components of beta are nonzero? 
	update_indices!(IDX, b, p=p)

	# if current vector is 0,
	# then take largest elements of d as nonzero components for b
	if sum(IDX) == 0
		selectperm!(sortidx,sdata(g),k, by=abs, rev=true, initialized=true)
		IDX[sortidx[1:k]] = true;
	end

	# update gradient
	loggrad!(g, Xb, lxb, xlxb, x, b, sortidx, mxty, lambda, n, k)

	# compute step size
	mu = lambda / (4.0*sqrt(k)*Rs*Rs + lambda)^2

	# notify problems with step size 
	isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))
	mu <= eps(typeof(mu))  && warn("Step size $(mu) is below machine precision, algorithm may not converge correctly")

	# take gradient step
	BLAS.axpy!(p, mu, g, 1, b, 1)

	# preserve top k components of b
	project_k!(b, bk, sortidx, k)

	# which indices of new beta are nonzero?
	copy!(IDX0, IDX)
	update_indices!(IDX, b, p=p) 

	# update xb
	update_xb!(Xb, x, b, sortidx, k, p=p, n=n)

	# calculate objective function 
	new_obj = compute_loglik(Xb, mxty, b, sortidx, lambda, k, n=n)

	# backtrack until mu sits below omega and support stabilizes
	# observe that we use -new_obj since the objective equals the NEGATIVE loglikelihood
	mu_step = 0
	while new_obj > old_obj && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

		# stephalving
		mu *= 0.5

		# recompute gradient step
		copy!(b,b0)
		BLAS.axpy!(p, mu, g, 1, b, 1)

		# preserve top k components of b
		project_k!(b, bk, sortidx, k)

		# which indices of new beta are nonzero?
		copy!(IDX0, IDX)
		update_indices!(IDX, b, p=p) 

		# update xb
		update_xb!(Xb, x, b, sortidx, k, p=p, n=n)

		# calculate objective function 
		new_obj = compute_loglik(Xb, mxty, b, sortidx, lambda, k, n=n)

		# increment the counter
		mu_step += 1
	end

	return mu, mu_step, new_obj
end

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
# -- max_iter is the maximum number of iterations for the algorithm. Defaults to 100.
# -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
# -- tol is the global tol. Defaults to 1e-4.
# -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
# -- several temporary arrays for intermediate steps of algorithm calculations:
#		Xb        = zeros(Float64,n)	# X*beta 
#		Xb0       = zeros(Float64,n)	# X*beta0 
#		b0        = zeros(Float64,p)	# previous iterate beta0 
#		df        = zeros(Float64,p)	# (negative) gradient 
#		bk        = zeros(Float64,k)    # temporary array of k floats 
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
function L0_log(
	x        :: DenseMatrix{Float64}, 
	y        :: DenseVector{Float64}, 
	k        :: Int; 
	n        :: Int                  = length(y), 
	p        :: Int                  = size(x,2), 
	mxty     :: DenseVector{Float64} = BLAS.gemv('T', -one(Float64), x, y),
	b        :: DenseVector{Float64} = zeros(Float64, p), 
	b0       :: DenseVector{Float64} = zeros(Float64, p), 
	df       :: DenseVector{Float64} = zeros(Float64, p), 
	xlxb     :: DenseVector{Float64} = zeros(Float64, p), 
	Xb       :: DenseVector{Float64} = zeros(Float64, n), 
	Xb0      :: DenseVector{Float64} = zeros(Float64, n), 
	lxb      :: DenseVector{Float64} = zeros(Float64, n), 
	bk       :: DenseVector{Float64} = zeros(Float64, k), 
	indices  :: DenseVector{Int}     = collect(1:p), 
	support  :: BitArray{1}          = falses(p), 
	support0 :: BitArray{1}          = falses(p),
	tol      :: Float64              = 1e-4, 
	lambda   :: Float64              = sqrt(log(p) / n)
	max_iter :: Int                  = 100, 
	max_step :: Int                  = 50,  
	quiet    :: Bool                 = true
)
	# start timer
	tic()

	# first handle errors
	k        >= 0            || throw(error("Value of k must be nonnegative!\n"))
	max_iter >= 0            || throw(error("Value of max_iter must be nonnegative!\n"))
	max_step >= 0            || throw(error("Value of max_step must be nonnegative!\n"))
	tol      >  eps(Float64) || throw(error("Value of global tol must exceed machine precision!\n"))
    all(isfinite(b))         || throw(error("Argument b has nonfinite values"))

	# initialize return values
	mm_iter  = 0                            # number of iterations of L0_reg
	mm_time  = zero(Float64)                # compute time *within* L0_reg
	next_obj = oftype(zero(Float64),Inf)    # objective value

	# initialize algorithm variables 
	current_obj = oftype(zero(Float64),Inf) # tracks previous objective function value
	the_norm    = zero(Float64)             # norm(b - b0)
	scaled_norm = zero(Float64)             # the_norm / (norm(b0) + 1)
	mu          = zero(Float64)             # IHT step size, 0 < tau < 2/rho_max^2
	mu_step     = 0                         # counts number of backtracking steps for mu
	converged   = false                     # scaled_norm < tol?
   
	# initialize x*b
	update_xb!(Xb, x, b, indices, k, p=p, n=n)

	# initialize gradient
	loggrad!(g, Xb, lxb, xlxb, x, b, sortidx, mxty, lambda, n, k)

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

			# stop timer
			mm_time = toq()

			# these are output variables for function
			# wrap them into a Dict and return
			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# xb0 = xb
		current_obj = next_obj

		# now perform IHT step
		(mu, mu_step, next_obj) = iht(b, x, y, g, k, current_obj, lambda, n=n, p=p Xb=Xb, mxty=mxty, xlxb=xlxb, lxb=lxb, bk=bk, sortidx=sortidx, IDX=support, IDX0=support0, max_step=max_step)

		# recompute gradient
		loggrad!(g, Xb, lxb, xlxb, x, b, sortidx, mxty, lambda, n, k)

		# ensure that objective is finite
		# if not, throw error
		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))

		# track convergence
		the_norm    = chebyshev(b,b0)
		scaled_norm = the_norm / (norm(b0,Inf) + one(Float64))
		converged   = scaled_norm < tol
		
		# output algorithm progress 
		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)

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
			throw(error("Descent failure!"))
#			output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

			return output
		end
	end # end main loop
end # end function



## L0 PENALIZED LOGISTIC REGRESSION FOR ENTIRE GWAS
##
## This routine solves the optimization problem
##
##     min 0.5*|| Y - XB ||_2^2 
##
## subject to
##
##     B in S_k = { x in R^p : || x ||_0 <= k }. 
##
## The algorithm applies a majorization to the centered logistic loglikelihood.
## In doing so, it converts the problem into a least squares problem and applies
## Thomas Blumensath's iterative hard thresholding framework to keep B feasible.
##
## Arguments:
## -- X is the BEDFile that contains the compressed n x p design matrix
## -- Y is the n x 1 binary response vector
## -- k is the desired size of the support (active set)
##
## Optional Arguments:
## -- b is the p x 1 iterate. Warm starts should use this argument. Defaults to zeros(p). 
## -- max_iter is the maximum number of iterations for the algorithm. Defaults to 100.
## -- max_step is the maximum number of backtracking steps for the step size calculation. Defaults to 50.
## -- tol is the global tol. Defaults to 1e-4.
## -- quiet is a Boolean that controls algorithm output. Defaults to true (no output).
## -- several temporary arrays for intermediate steps of algorithm calculations:
##		Xk        = zeros(Float64,n,k)  # store k columns of X
##		r         = zeros(Float64,n)	# for || Y - XB ||_2^2
##		Xb        = zeros(Float64,n)	# X*beta 
##		Xb0       = zeros(Float64,n)	# X*beta0 
##		b0        = zeros(Float64,p)	# previous iterate beta0 
##		df        = zeros(Float64,p)	# (negative) gradient 
##		w         = zeros(Float64,n)	# vector of weights on responses 
##		tempkf    = zeros(Float64,k)    # temporary array of k floats 
##		idx       = zeros(Float64,k)    # another temporary array of k floats 
##		tempn     = zeros(Float64,n)    # temporary array of n floats 
##		indices   = collect(1:p)	    # indices that sort beta 
##		tempki    = zeros(Int,k)        # temporary array of k integers 
##		support   = falses(p)			# indicates nonzero components of beta
##		support0  = copy(support)		# store previous nonzero indicators
##
## Outputs are wrapped into a Dict with the following fields:
## -- time is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults
## -- iter is the number of iterations that the algorithm took
## -- loss is the optimal loss (residual sum of squares divided by sqrt of RSS with previous iterate)
## -- beta is the final iterate
##
## coded by Kevin L. Keys (2015)
## klkeys@g.ucla.edu
#function L0_log(X::BEDFile, Y::DenseVector{Float64}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseVector{Float64} = zeros(p), tol::Float64 = 1e-4, max_iter::Int = 100, max_step::Int = 50,  quiet::Bool = true, Xk::DenseMatrix{Float64} = zeros(n,k), r::DenseVector{Float64} = zeros(n), Xb::DenseVector{Float64} = zeros(n), Xb0::DenseVector{Float64} = zeros(n), w::DenseVector{Float64} = ones(n), b0::DenseVector{Float64} = zeros(p), df::DenseVector{Float64} = zeros(p), tempkf::DenseVector{Float64} = zeros(k), idx::DenseVector{Float64} = zeros(k), tempn::DenseVector{Float64}= zeros(n), indices::DenseVector{Int} = collect(1:p), tempki::DenseVector{Int} = zeros(Int,k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p))
#
#	# start timer
#	tic()
#
#	# first handle errors
#	k            >= 0     || throw(error("Value of k must be nonnegative!\n"))
#	max_iter     >= 0     || throw(error("Value of max_iter must be nonnegative!\n"))
#	max_step     >= 0     || throw(error("Value of max_step must be nonnegative!\n"))
#	tol    >  eps() || throw(error("Value of global tol must exceed machine precision!\n"))
#
#	# initialize return values
#	mm_iter::Int       = 0		# number of iterations of L0_reg
#	mm_time::Float64   = 0.0	# compute time *within* L0_reg
#	next_obj::Float64  = Inf	# objective value
#	next_loss::Float64 = -Inf	# loss function value 
#
#	# initialize floats 
#	current_obj::Float64 = Inf      # tracks previous objective function value
#	the_norm::Float64    = 0.0      # norm(b - b0)
#	scaled_norm::Float64 = 0.0      # the_norm / (norm(b0) + 1)
#	mu::Float64          = Inf	    # IHT step size, 0 < tau < 2/rho_max^2
#
#	# initialize integers
#	mu_step::Int         = 0        # counts number of backtracking steps for mu
#
#	# initialize booleans
#	converged::Bool      = false    # scaled_norm < tol?
#   
#	# initialize x*b
#	xb!(Xb,X,b,indices,k)
#
#	# initialize weights
#	update_weights!(w, X, b, perm, k, xb=Xb, n=n, p=p) 
#
#	# initialize weighted r 
#	update_residuals!(r, X, Y, b, indices, w, k, n=n, p=p, xb=Xb)
#
#	# initialize gradient
#	xty!(df, X, r)
#
#	# initialize loss value
#	next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
#
#	# guard against numerical instabilities
#	isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
#	isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
#
#	# formatted output to monitor algorithm progress
#	if !quiet
#		 println("\nBegin MM algorithm\n") 
#		 println("Iter\tHalves\tMu\t\tNorm\t\tObjective")
#		 println("0\t0\tInf\t\tInf\t\tInf")
#	end
#
#	# main loop
#	for mm_iter = 1:max_iter
# 
#		# notify and break if maximum iterations are reached.
#		if mm_iter >= max_iter
#
#			if !quiet
#				print_with_color(:red, "MM algorithm has hit maximum iterations $(max_iter)!\n") 
#				print_with_color(:red, "Current Objective: $(current_obj)\n") 
#			end
#
#			# send elements below tol to zero
#			threshold!(b, tol, n=p)
#
#			# update X*b for calculating loglikelihood 
##			update_xb!(Xb, X, b, indices, k, p=p, n=n)
#			xb!(Xb,X,b,indices,k)
#
#			# calculate loglikelihood 
#			next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
#
#			# stop timer
#			mm_time = toq()
#
#			# these are output variables for function
#			# wrap them into a Dict and return
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
##			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)
#
#			return output
#		end
#		
#		# save values from previous iterate 
#		copy!(b0,b)				# b0 = b	
#		copy!(Xb0,Xb)			# Xb0 = Xb
#		current_obj = next_obj
#
#		# now perform IHT step
#		(mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortk=tempki, sortidx=indices, gk=idx, step_multiplier=1.0)
#
#		# recompute weights
#		update_weights!(w, X, b, perm, k, xb=Xb, n=n, p=p) 
#
#		# initialize weighted r 
#		update_residuals!(r, X, Y, b, indices, w, k, n=n, p=p, xb=Xb)
#
#		# initialize gradient
#		xty!(df, X, r)
#
#		# update loss, objective
#		next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
#		next_obj  = -next_loss
#
#
#		# ensure that objective is finite
#		# if not, throw error
#		isnan(next_obj) && throw(error("Objective function is NaN, aborting..."))
#		isinf(next_obj) && throw(error("Objective function is Inf, aborting..."))
#
#		# track convergence
#		the_norm    = chebyshev(b,b0)
#		scaled_norm = the_norm / ( norm(b0,Inf) + 1)
#		converged   = scaled_norm < tol
#		
#		# output algorithm progress 
#		quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_obj)
#
#		# check for convergence
#		# if converged and in feasible set, then algorithm converged before maximum iteration
#		# perform final computations and output return variables 
#		if converged
#			
#			# send elements below tol to zero
#			threshold!(b, tol, n=p)
#
#			# update X*b for loglikelihood 
##			update_xb!(Xb, X, b, indices, k, p=p, n=n)
#			xb!(Xb,X,b,indices,k)
#
#			# calculate loglikelihood 
#			next_loss = compute_loglik(Y,X,b,perm,k, p=p, n=n, xb=Xb)
#			
#			# stop time
#			mm_time = toq()
#
#			if !quiet
#				println("\nMM algorithm has converged successfully.")
#				println("MM Results:\nIterations: $(mm_iter)") 
#				println("Final Loss: $(next_loss)") 
#				println("Total Compute Time: $(mm_time)") 
#			end
#
#
#			# these are output variables for function
#			# wrap them into a Dict and return
#			output = {"time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b}
##			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)
#
#			return output
#		end
#
#		# algorithm is unconverged at this point.
#		# if algorithm is in feasible set, then rho should not be changing
#		# check descent property in that case
#		# if rho is not changing but objective increases, then abort
#		if next_obj > current_obj + tol
#			if !quiet
#				print_with_color(:red, "\nMM algorithm fails to descend!\n")
#				print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
#				print_with_color(:red, "Current Objective: $(current_obj)\n") 
#				print_with_color(:red, "Next Objective: $(next_obj)\n") 
#				print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
#			end
#
#			output = {"time" => -1, "loss" => -Inf, "iter" => -1, "beta" => fill!(b, Inf)}
##			output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))
#
#			return output
#		end
#	end # end main loop
#end # end function





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
# -- max_iter caps the number of iterations for the algorithm. Defaults to 100.
# -- max_step caps the number of backtracking steps in the IHT kernel. Defaults to 50.
# -- quiet is a Boolean that controls the output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function iht_path_log(
	x        :: DenseMatrix{Float64}, 
	y        :: DenseVector{Float64}, 
	path     :: DenseVector{Int}; 
	tol      :: Float64 = 1e-4,
	max_iter :: Int     = 100, 
	max_step :: Int     = 50, 
	quiet    :: Bool    = true
)

	# size of problem?
	const (n,p) = size(x)

	# preallocate intermediate variables for IHT 
	mxty      = BLAS.gemv('T', -one(Float64), x, y) # - x * y
	b         = zeros(Float64, p)	                # model 
	b0        = zeros(Float64, p)	                # previous iterate beta0 
	df        = zeros(Float64, p)	                # (negative) gradient 
	Xb        = zeros(Float64, n)	                # X*beta 
	Xb0       = zeros(Float64, n)	                # X*beta0 
	indices   = collect(1:p)	                    # indices that sort beta 
	support   = falses(p)			                # indicates nonzero components of beta
	support0  = copy(support)		                # store previous nonzero indicators
	lambda    = sqrt(log(p) / n)                    # Tikhonov regularization parameter

	# preallocate space for path computation
	num_models = length(path)			# how many models will we compute?
	betas      = spzeros(p,num_models)	# a matrix to store calculated models

	# compute the path
	for i = 1:num_models
	
		# model size?
		q = path[i]

		# other variables change in size, so allocate them at every new model
		bk = zeros(q) # largest k nonzeroes of beta in magnitude 

		# store projection of beta onto largest k nonzeroes in magnitude 
		project_k!(b,bk,indices,q, p=p)

		# now compute current model
		output = L0_log(x, y, k, n=n, p=p,
mxty=mxty,
b=b,
b0=b0,
df=df,
xlxb=xlxb,
Xb=Xb,
Xb0=Xb0,
lxb=lxb,
bk=bk,
indices=indices,
support=support,
support0=support0,
tol=tol,
lambda=lambda,
max_iter=max_iter,
max_step=max_step,
quiet=quiet
)

		# extract and save model
		copy!(b, output["beta"])

		betas[:,i] = sparsevec(b)
	end

	# return a sparsified copy of the models
	return betas
end	# end iht_path_log	
