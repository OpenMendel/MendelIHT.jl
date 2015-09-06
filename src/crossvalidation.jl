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
# -- tol is the convergence tol to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- max_step caps the number of permissible backtracking steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
# -- logreg is a switch to activate logistic regression. Defaults to false (perform linear regression).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function one_fold(
	x        :: DenseArray{Float64,2}, 
	y        :: DenseArray{Float64,1}, 
	path     :: DenseArray{Int,1}, 
	folds    :: DenseArray{Int,1}, 
	fold     :: Int; 
	max_iter :: Int  = 1000, 
	max_step :: Int  = 50, 
	quiet    :: Bool = true, 
	logreg   :: Bool = false
) 

	# make vector of indices for folds
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
	train_idx = !test_idx

	# allocate the arrays for the training set
	x_train   = x[train_idx,:]
	y_train   = y[train_idx] 

	if logreg
		# compute the regularization path on the training set
		betas    = iht_path_log(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step) 

		# compute the mean out-of-sample error for the TEST set 
		myerrors = vec(sumabs2(broadcast(-, round(y[test_idx]), round(logistic(x[test_idx,:] * betas))), 1)) ./ length(test_idx)
	else
		# compute the regularization path on the training set
		betas    = iht_path(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step) 

		# compute the mean out-of-sample error for the TEST set 
		myerrors = vec(sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ length(test_idx)
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
	x             :: DenseArray{Float64,2}, 
	y             :: DenseArray{Float64,1}, 
	path          :: DenseArray{Int,1}, 
	numfolds      :: Int; 
	folds         :: DenseArray{Int,1} = cv_get_folds(sdata(y),numfolds), 
	tol           :: Float64           = 1e-4, 
	n             :: Int               = length(y), 
	p             :: Int               = size(x,2), 
	max_iter      :: Int               = 1000, 
	max_step      :: Int               = 50, 
	quiet         :: Bool              = true, 
	logreg        :: Bool              = false, 
	compute_model :: Bool              = true
) 

	# how many elements are in the path?
	const num_models = length(path)

	# preallocate vectors used in xval	
	errors  = zeros(Float64, num_models)	# vector to save mean squared errors
	my_refs = cell(numfolds)				# cell array to store RemoteRefs

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
		my_refs[i] = @spawn(one_fold(x, y, path, folds, i, max_iter=max_iter, max_step=max_step, quiet=quiet, logreg=logreg))
	end
	
	# recover MSEs on each worker
	@inbounds for i = 1:numfolds
		errors += fetch(my_refs[i])
	end

	# average the mses
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

		# initialize parameter vector
		b = zeros(Float64, p)

		if logreg
			
			# can preallocate some of the temporary arrays for use in both model selection and fitting
			# notice that they all depend on n, which is fixed,
			# as opposed to p, which changes depending on the number of nonzeroes in b
			xb   = zeros(Float64, n)      # xb = x*b 
			lxb  = zeros(Float64, n)      # logistic(xb), which we call pi 
			l2xb = zeros(Float64, n)      # logistic(xb) [ 1 - logistic(xb) ], or pi(1 - pi)

			# first use L0_reg to extract model
			output = L0_log(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, Xb=xb, Xb0=lxb, r=l2xb)
			copy!(b, output["beta"])

			# which components of beta are nonzero?
			bidx = find( x -> x .!= 0.0, b) 

			# allocate the submatrix of x corresponding to the inferred model
			x_inferred = x[:,bidx]

			# compute logistic fit
			b2 = fit_logistic(x_inferred, y, xb=xb, lxb=lxb, l2xb=l2xb)	
		else

			# first use L0_reg to extract model
			output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol)
			copy!(b, output["beta"])

			# which components of beta are nonzero?
			bidx = find( x -> x .!= 0.0, b) 

			# allocate the submatrix of x corresponding to the inferred model
			x_inferred = x[:,bidx]

			# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
			xty = BLAS.gemv('T', 1.0, x_inferred, y)	
			xtx = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
			b2 = xtx \ xty
		end
		return errors, b2, bidx 
	end
	return errors
end
