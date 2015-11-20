"""
If used with a BEDFile x, then the additional optional arguments are
-- pids, a vector of process IDs. Defaults to procs().
-- means, a vector of SNP means. Defaults to mean(Float64, x, shared=true, pids=procs().
-- invstds, a vector of SNP precisions. Defaults to mean(Float64, x, shared=true, pids=procs().
In addition, the temporary arrays are SharedArrays.
"""
function iht(
    b        :: DenseVector{Float64}, 
    x        :: BEDFile, 
    y        :: DenseVector{Float64}, 
    k        :: Int, 
    g        :: DenseVector{Float64}; 
    n        :: Int                  = length(y), 
    p        :: Int                  = length(b), 
    pids     :: DenseVector{Int}     = procs(),
    means    :: DenseVector{Float64} = mean(Float64, x, shared=true, pids=pids), 
    invstds  :: DenseVector{Float64} = invstd(x, means, shared=true, pids=pids),
    b0       :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = b[localindexes(S)], pids=pids), 
    Xb       :: DenseVector{Float64} = xb(x,b,IDX,k, means=means, invstds=invstds, pids=pids), 
    Xb0      :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = Xb[localindexes(S)], pids=pids), 
    sortidx  :: DenseVector{Int}     = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids), 
    xk       :: DenseMatrix{Float64} = zeros(Float64,n,k), 
    xgk      :: DenseVector{Float64} = zeros(Float64,n), 
    gk       :: DenseVector{Float64} = zeros(Float64,k), 
    bk       :: DenseVector{Float64} = zeros(Float64,k), 
    IDX      :: BitArray{1}          = falses(p), 
    IDX0     :: BitArray{1}          = copy(IDX), 
    iter     :: Int                  = 1,
    max_step :: Int                  = 50, 
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
        decompress_genotypes!(xk, x, IDX, means=means, invstds=invstds) 
    end

    # store relevant components of gradient
    fill_perm!(sdata(gk), sdata(g), IDX, k=k, p=p)  # gk = g[IDX]

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
    xb!(Xb,x,b,IDX,k, means=means, invstds=invstds, pids=pids)

    # calculate omega
    omega_top = sqeuclidean(sdata(b),sdata(b0))
    omega_bot = sqeuclidean(sdata(Xb),sdata(Xb0))

    # backtrack until mu sits below omega and support stabilizes
    mu_step = 0
    while mu*omega_bot > 0.99*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

        # stephalving
        mu *= 0.5

        # warn if mu falls below machine epsilon 
        mu <= eps(typeof(mu)) && warn("Step size equals zero, algorithm may not converge correctly")

        # recompute gradient step
        copy!(b,b0)
        BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

        # recompute projection onto top k components of b
        project_k!(b, bk, sortidx, k)

        # which indices of new beta are nonzero?
        update_indices!(IDX, b, p=p) 

        # recompute xb
        xb!(Xb,x,b,IDX,k, means=means, invstds=invstds, pids=pids)

        # calculate omega
        omega_top = sqeuclidean(sdata(b),sdata(b0))
        omega_bot = sqeuclidean(sdata(Xb),sdata(Xb0))

        # increment the counter
        mu_step += 1
    end

    return mu, mu_step
end

"""
If called with a BEDFile object X, then the additional optional arguments are
-- pids, a vector of process IDs. Defaults to procs().
-- means, a vector of SNP means. Defaults to mean(Float64, x, shared=true, pids=procs().
-- invstds, a vector of SNP precisions. Defaults to mean(Float64, x, shared=true, pids=procs().
"""
function L0_reg(
    X        :: BEDFile, 
    Y        :: DenseVector{Float64}, 
    k        :: Int; 
    n        :: Int                  = length(Y), 
    p        :: Int                  = size(X,2), 
    pids     :: DenseVector{Int}     = procs(),
    Xk       :: DenseMatrix{Float64} = SharedArray(Float64, (n,k), init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    means    :: DenseVector{Float64} = mean(Float64,X, shared=true, pids=pids), 
    invstds  :: DenseVector{Float64} = invstd(X,means, shared=true, pids=pids), 
    b        :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    b0       :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    df       :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    r        :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    Xb       :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    Xb0      :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    tempn    :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    tempkf   :: DenseVector{Float64} = SharedArray(Float64, k, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    idx      :: DenseVector{Float64} = SharedArray(Float64, k, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids), 
    indices  :: DenseVector{Int}     = SharedArray(Int,     p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids), 
    support  :: BitArray{1}          = falses(p), 
    support0 :: BitArray{1}          = falses(p), 
    tol      :: Float64              = 1e-4, 
    max_iter :: Int                  = 100, 
    max_step :: Int                  = 50, 
    quiet    :: Bool                 = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0            || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0            || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0            || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps(Float64) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    # initialize return values
    mm_iter   = 0                           # number of iterations of L0_reg
    mm_time   = zero(Float64)               # compute time *within* L0_reg
    next_loss = oftype(zero(Float64),Inf)   # loss function value  

    # initialize floats 
    current_obj = oftype(zero(Float64),Inf) # tracks previous objective function value
    the_norm    = zero(Float64)             # norm(b - b0)
    scaled_norm = zero(Float64)             # the_norm / (norm(b0) + 1)
    mu          = zero(Float64)             # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0                             # used for iterations in loops
    mu_step = 0                             # counts number of backtracking steps for mu

    # initialize booleans
    converged = false                       # scaled_norm < tol?
   
    # update Xb, r, and gradient 
    if sum(support) == 0
        fill!(Xb,zero(Float64))
        copy!(r,sdata(Y))
    else
        xb!(Xb,X,b,support,k, means=means, invstds=invstds, pids=pids)
        difference!(r, Y, Xb)
    end
    xty!(df, X, r, means=means, invstds=invstds, p=p, pids=pids) 

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
        copy!(b0,b)             # b0 = b    
        copy!(Xb0,Xb)           # Xb0 = Xb
        current_obj = next_loss

        # now perform IHT step
        (mu, mu_step) = iht(b,X,Y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, Xb=Xb, Xb0=Xb0, xgk=tempn, xk=Xk, bk=tempkf, sortidx=indices, gk=idx, means=means, invstds=invstds, iter=mm_iter, pids=pids) 

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient 
        difference!(r,Y,Xb)
        xty!(df, X, r, means=means, invstds=invstds, pids=pids) 

        # update loss, objective, and gradient 
        next_loss = 0.5 * sumabs2(sdata(r))

        # guard against numerical instabilities
        # ensure that objective is finite
        # if not, throw error
        isnan(next_loss) && throw(error("Objective function is NaN, aborting..."))
        isinf(next_loss) && throw(error("Objective function is Inf32, aborting..."))

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

        # algorithm is unconverged at this point.
        # if algorithm is in feasible set, then rho should not be changing
        # check descent property in that case
        # if rho is not changing but objective increases, then abort
        if next_loss > current_obj + tol
            if !quiet
                print_with_color(:red, "\nMM algorithm fails to descend!\n")
                print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
                print_with_color(:red, "Current Objective: $(current_obj)\n") 
                print_with_color(:red, "Next Objective: $(next_loss)\n") 
                print_with_color(:red, "Difference in objectives: $(abs(next_loss - current_loss))\n")
            end
            throw(ErrorException("Descent failure!"))
#           output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf32))

            return output
        end
    end # end main loop
end # end function



"""
If called with a BEDFile object X, then the additional optional arguments are
-- pids, a vector of process IDs. Defaults to procs().
-- means, a vector of SNP means. Defaults to mean(Float64, x, shared=true, pids=procs().
-- invstds, a vector of SNP precisions. Defaults to mean(Float64, x, shared=true, pids=procs().
"""
function iht_path(
    x        :: BEDFile, 
    y        :: DenseVector{Float64}, 
    path     :: DenseVector{Int}; 
    pids     :: DenseVector{Int}     = procs(),
    means    :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
    invstds  :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
    tol      :: Float64              = 1e-4, 
    max_iter :: Int                  = 100, 
    max_step :: Int                  = 50, 
    quiet    :: Bool                 = true
)

    # size of problem?
    n = length(y)
    p = size(x,2)

    # how many models will we compute?
    num_models = length(path)         

    # preallocate SharedArrays for intermediate steps of algorithm calculations 
    b       = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # for || Y - XB ||_2^2
    b0      = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # previous iterate beta0 
    df      = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # (negative) gradient 
    r       = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # for || Y - XB ||_2^2
    Xb      = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # X*beta 
    Xb0     = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # X*beta0 
    tempn   = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)        # temporary array of n floats 

    # index vector for b has more complicated initialization
    indices = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)

    # allocate the BitArrays for indexing in IHT
    # also preallocate matrix to store betas 
    support    = falses(p)                      # indicates nonzero components of beta
    support0   = copy(support)                  # store previous nonzero indicators
    betas      = spzeros(Float64,p,num_models)  # a matrix to store calculated models

    # compute the path
    @inbounds for i = 1:num_models
    
        # model size?
        q = path[i]

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        Xk     = zeros(Float64,n,q)     # store q columns of X
        tempkf = zeros(Float64,q)       # temporary array of q floats 
        idx    = zeros(Float64,q)       # another temporary array of q floats 

        # store projection of beta onto largest k nonzeroes in magnitude 
        project_k!(b, tempkf, indices, q)

        # now compute current model
        output = L0_reg(x,y,q, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, Xk=Xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, idx=idx, tempn=tempn, indices=indices, support=support, support0=support0, means=means, invstds=invstds, pids=pids) 

        # extract and save model
        copy!(sdata(b), output["beta"])
        
        # ensure that we correctly index the nonzeroes in b
        update_indices!(support, b, p=p)    
        fill!(support0, false)

        # put model into sparse matrix of betas
        betas[:,i] = sparsevec(sdata(b))
    end

    # return a sparsified copy of the models
    return betas
end 


"""
If called with a BEDFile object X, then the additional optional arguments are
-- pids, a vector of process IDs. Defaults to procs().
-- means, a vector of SNP means. Defaults to mean(Float64, x, shared=true, pids=procs().
-- invstds, a vector of SNP precisions. Defaults to mean(Float64, x, shared=true, pids=procs().
"""
function one_fold(
    x        :: BEDFile, 
    y        :: DenseVector{Float64}, 
    path     :: DenseVector{Int}, 
    folds    :: DenseVector{Int}, 
    fold     :: Int; 
    pids     :: DenseVector{Int}     = procs(),
    means    :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
    invstds  :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids), 
    max_iter :: Int                  = 100, 
    max_step :: Int                  = 50, 
    quiet    :: Bool                 = true
)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # preallocate vector for output
    myerrors = zeros(Float64, test_size)

    # train_idx is the vector that indexes the Float64RAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx] 
    Xb      = SharedArray(Float64, test_size, init = S -> S[localindexes(S)] = zero(Float64), pids=pids) 
    b       = SharedArray(Float64, size(x,2), init = S -> S[localindexes(S)] = zero(Float64), pids=pids) 
    r       = SharedArray(Float64, test_size, init = S -> S[localindexes(S)] = zero(Float64), pids=pids) 

    # compute the regularization path on the training set
    betas = iht_path(x_train,y_train,path, max_iter=max_iter, quiet=quiet, max_step=max_step, means=means, invstds=invstds, pids=pids) 

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
        xb!(Xb,x_test,b,indices,path[i],test_idx, means=means, invstds=invstds, pids=pids)

        # compute residuals
        difference!(r,y,Xb)

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size
    end

    return myerrors
end



"""
If called with a BEDFile object X, then the additional optional arguments are
-- pids, a vector of process IDs. Defaults to procs().
-- means, a vector of SNP means. Defaults to mean(Float64, x, shared=true, pids=procs().
-- invstds, a vector of SNP precisions. Defaults to mean(Float64, x, shared=true, pids=procs().
"""
function cv_iht(
    x             :: BEDFile, 
    y             :: DenseVector{Float64}, 
    path          :: DenseVector{Int}, 
    numfolds      :: Int; 
    folds         :: DenseVector{Int}     = cv_get_folds(sdata(y),numfolds), 
    pids          :: DenseVector{Int}     = procs(),
    means         :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
    invstds       :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
    tol           :: Float64              = 1e-4, 
    n             :: Int                  = length(y), 
    p             :: Int                  = size(x,2), 
    max_iter      :: Int                  = 100, 
    max_step      :: Int                  = 50, 
    quiet         :: Bool                 = true, 
    compute_model :: Bool                 = false
) 

    # how many elements are in the path?
    num_models = length(path)

    # preallocate vectors used in xval  
    errors  = zeros(Float64, num_models)    # vector to save mean squared errors
    my_refs = cell(numfolds)                # cell array to store RemoteRefs

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding 
    @sync for i = 1:numfolds
        # one_fold returns a vector of out-of-sample errors (MSE for linear regression) 
        # @fetch(one_fold(...)) spawns one_fold() on a CPU and returns the MSE 
        errors[i] = @fetch(one_fold(x, y, path, folds, i, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, pids=pids)) 
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

        # initialize model
        b = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)

        # first use L0_reg to extract model
        output = L0_reg(x,y,k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, b=b, pids=pids)

        # which components of beta are nonzero?
        inferred_model = output["beta"] .!= zero(Float64) 
        bidx = find( x -> x .!= zero(Float64), b) 

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = zeros(Float64,n,sum(inferred_model))
        decompress_genotypes!(x_inferred,x)

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
        xty = BLAS.gemv('T', one(Float64), x_inferred, y)   
        xtx = BLAS.gemm('T', 'N', one(Float64), x_inferred, x_inferred)
        b   = xtx \ xty
        return errors, b, bidx 
    end
    return errors
end
