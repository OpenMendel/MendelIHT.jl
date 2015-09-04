##################################
### LOGISTIC STEP CALCULATIONS ###
##################################

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
		sortk = RegressionTools.selectperm!(sortidx,lg,k, p=p)
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
	sortk = RegressionTools.selectperm!(sortidx,b,k, p=p)
	fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
	fill!(b,0.0)
	b[sortk] = bk
#	RegressionTools.project_k!(b, bk, sortk, sortidx, k)

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
		sortk = RegressionTools.selectperm!(sortidx, b,k, p=p)
		fill_perm!(bk, b, sortk, k=k)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		RegressionTools.project_k!(b, bk, sortk, sortidx, k)

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
#function L0_log(X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0), tolerance::Float64 = 1e-4, max_iter::Int = 10000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = SharedArray(Float64, n,k), r::DenseArray{Float64,1} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0), Xb::DenseArray{Float64,1} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0), Xb0::DenseArray{Float64,1} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0), w::DenseArray{Float64,1} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 1.0), b0::DenseArray{Float64,1} = copy(b), df::DenseArray{Float64,1} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0), tempkf::DenseArray{Float64,1} = SharedArray(Float64, k, init = S -> S[localindexes(S)] = 0.0), idx::DenseArray{Float64,1} = SharedArray(Float64, k, init = S -> S[localindexes(S)] = 0.0), tempn::DenseArray{Float64,1}= SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0), indices::DenseArray{Int,1} = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S)), tempki::DenseArray{Int,1} = SharedArray(Int,k, init = S -> S[localindexes(S)] = 0), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p))
function L0_log(X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, k::Int; n::Int = length(Y), p::Int = size(X,2), b::DenseArray{Float64,1} = zeros(p), tolerance::Float64 = 1e-4, max_iter::Int = 10000, max_step::Int = 50,  quiet::Bool = true, Xk::DenseArray{Float64,2} = zeros(n,k), r::DenseArray{Float64,1} = zeros(n), Xb::DenseArray{Float64,1} = zeros(n), Xb0::DenseArray{Float64,1} = zeros(n), w::DenseArray{Float64,1} = ones(n), b0::DenseArray{Float64,1} = copy(b), df::DenseArray{Float64,1} = zeros(p), tempkf::DenseArray{Float64,1} = zeros(k), idx::DenseArray{Float64,1} = zeros(k), tempn::DenseArray{Float64,1} = zeros(n), indices::DenseArray{Int,1} = collect(1:p), tempki::DenseArray{Int,1} = collect(1:k), support::BitArray{1} = falses(p), support0::BitArray{1} = falses(p))

	# start timer
	tic()

	# first handle errors
	k            >= 0     || throw(error("Value of k must be nonnegative!\n"))
	max_iter     >= 0     || throw(error("Value of max_iter must be nonnegative!\n"))
	max_step     >= 0     || throw(error("Value of max_step must be nonnegative!\n"))
	tolerance    >  eps() || throw(error("Value of global tolerance must exceed machine precision!\n"))
    all(isfinite(b))      || throw(error("Argument b has nonfinite values"))

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

			return output
		end
		
		# save values from previous iterate 
		copy!(b0,b)				# b0 = b	
		copy!(Xb0,Xb)			# Xb0 = Xb
		current_obj = next_obj


#		println("all b finite?", all(isfinite(b)))
#		println("all b0 finite?", all(isfinite(b0)))
#		println("all Xb finite?", all(isfinite(Xb)))
#		println("all Xb0 finite?", all(isfinite(Xb0)))
#		println("all w finite?", all(isfinite(w)))
#		println("all df finite?", all(isfinite(df)))
#		println("all Xk finite?", all(isfinite(Xk)))
#		println("all tempki finite?", all(isfinite(tempki)))
#		println("all tempkf finite?", all(isfinite(tempkf)))
#		println("all gk finite?", all(isfinite(idx)))

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

#		println("all b finite?", all(isfinite(b)))
#		println("all b0 finite?", all(isfinite(b0)))
#		println("all Xb finite?", all(isfinite(Xb)))
#		println("all Xb0 finite?", all(isfinite(Xb0)))
#		println("all w finite?", all(isfinite(w)))
#		println("all df finite?", all(isfinite(df)))
#		println("all Xk finite?", all(isfinite(Xk)))
#		println("all tempki finite?", all(isfinite(tempki)))
#		println("all tempkf finite?", all(isfinite(tempkf)))
#		println("all gk finite?", all(isfinite(idx)))

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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
#			output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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
#			output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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
#			output = Dict{ASCIIString, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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
#			output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))

			return output
		end
	end # end main loop
end # end function


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
#		sortk    = zeros(Int,q)
		sortk    = RegressionTools.selectperm!(indices, b,q, p=p) 
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		RegressionTools.project_k!(b, bk, sortk, indices, q)

		# other variables change in size, so allocate them at every new model
		Xk     = zeros(Float64,n,q)  # store q columns of X
		tempkf = zeros(Float64,q)    # temporary array of q floats 
		idx    = zeros(Float64,q)    # another temporary array of q floats 
		tempki = collect(1:q)        # temporary array of q integers 

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
#		sortk    = zeros(Int,q)
		sortk    = RegressionTools.selectperm!(indices, b,q, p=p)
		fill_perm!(bk, b, sortk, k=q)	# bk = b[sortk]
		fill!(b,0.0)
		b[sortk] = bk
#		RegressionTools.project_k!(b, bk, sortk, indices, q)

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
