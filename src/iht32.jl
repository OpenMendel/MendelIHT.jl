function iht(
    b         :: DenseVector{Float32}, 
    x         :: DenseMatrix{Float32}, 
    y         :: DenseVector{Float32}, 
    k         :: Int, 
    g         :: DenseVector{Float32}; 
    n         :: Int                  = length(y), 
    p         :: Int                  = length(b), 
    b0        :: DenseVector{Float32} = copy(b), 
    xb        :: DenseVector{Float32} = BLAS.gemv('N', one(Float32), x, b), 
    xb0       :: DenseVector{Float32} = copy(xb), 
    xk        :: DenseMatrix{Float32} = zeros(Float32,n,k), 
    xgk       :: DenseVector{Float32} = zeros(Float32,n), 
    gk        :: DenseVector{Float32} = zeros(Float32,k), 
    bk        :: DenseVector{Float32} = zeros(Float32,k), 
    sortidx   :: DenseVector{Int}     = collect(1:p), 
    IDX       :: BitArray{1}          = falses(p), 
    IDX0      :: BitArray{1}          = copy(IDX), 
    iter      :: Int                  = 1,
    max_step  :: Int                  = 50
) 

    # which components of beta are nonzero? 
    update_indices!(IDX, b, p=p)

    # if current vector is 0,
    # then take largest elements of d as nonzero components for b
    if sum(IDX) == 0
        selectpermk!(sortidx,g,k, p=p) 
        IDX[sortidx[1:k]] = true;
    end

    # store relevant columns of x
    # need to do this on 1st iteration
    # afterwards, only do if support changes
    if !isequal(IDX,IDX0) || iter < 2
        update_xk!(xk, x, IDX, k=k, p=p, n=n)   # xk = x[:,IDX]
    end

    # store relevant components of gradient
    fill_perm!(gk, g, IDX, k=k, p=p)    # gk = g[IDX]

    # now compute subset of x*g
    BLAS.gemv!('N', one(Float32), xk, gk, zero(Float32), xgk)

    # compute step size
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
    while mu*omega_bot > 0.99f0*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

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

function L0_reg(
    x         :: DenseMatrix{Float32}, 
    y         :: DenseVector{Float32}, 
    k         :: Int; 
    n         :: Int                  = length(y),
    p         :: Int                  = size(x,2),
    xk        :: DenseMatrix{Float32} = zeros(Float32,n,k),
    b         :: DenseVector{Float32} = zeros(Float32,p),
    b0        :: DenseVector{Float32} = zeros(Float32,p),
    df        :: DenseVector{Float32} = zeros(Float32,p),
    r         :: DenseVector{Float32} = zeros(Float32,n),
    Xb        :: DenseVector{Float32} = zeros(Float32,n),
    Xb0       :: DenseVector{Float32} = zeros(Float32,n),
    tempn     :: DenseVector{Float32} = zeros(Float32,n),
    tempkf    :: DenseVector{Float32} = zeros(Float32,k),
    gk        :: DenseVector{Float32} = zeros(Float32,k),
    indices   :: DenseVector{Int}     = collect(1:p),
    support   :: BitArray{1}          = falses(p),
    support0  :: BitArray{1}          = falses(p),
    tol       :: Float32              = 1f-4,
    max_iter  :: Int                  = 100,
    max_step  :: Int                  = 50,
    quiet     :: Bool                 = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0     || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0     || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0     || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps() || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    # initialize return values
    mm_iter   = 0                           # number of iterations of L0_reg
    mm_time   = zero(Float32)               # compute time *within* L0_reg
    next_obj  = zero(Float32)               # objective value
    next_loss = zero(Float32)               # loss function value 

    # initialize floats 
    current_obj = oftype(zero(Float32),Inf) # tracks previous objective function value
    the_norm    = zero(Float32)             # norm(b - b0)
    scaled_norm = zero(Float32)             # the_norm / (norm(b0) + 1)
    mu          = zero(Float32)             # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0                             # used for iterations in loops
    mu_step = 0                             # counts number of backtracking steps for mu

    # initialize booleans
    converged = false                       # scaled_norm < tol?
   
    # update X*beta
    update_xb!(Xb, x, b, indices, k, p=p, n=n)

    # update r and gradient 
    difference!(r,y,Xb, n=n)
    BLAS.gemv!('T', one(Float32), x, r, zero(Float32), df)

    # update loss and objective
    next_loss = oftype(zero(Float32),Inf) 

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
        current_obj = next_obj

        # now perform IHT step
        (mu, mu_step) = iht(b,x,y,k,df, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=xk, bk=tempkf, sortidx=indices, gk=gk, iter=mm_iter)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient 
        difference!(r,y,Xb, n=n)
        BLAS.gemv!('T', zero(Float32), x, r, zero(Float32), df)

        # update loss, objective, and gradient 
        next_loss = 0.5f0 * sumabs2(r)

        # guard against numerical instabilities
        isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
        isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

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
        if next_obj > current_obj + tol
            if !quiet
                print_with_color(:red, "\nMM algorithm fails to descend!\n")
                print_with_color(:red, "MM Iteration: $(mm_iter)\n") 
                print_with_color(:red, "Current Objective: $(current_obj)\n") 
                print_with_color(:red, "Next Objective: $(next_obj)\n") 
                print_with_color(:red, "Difference in objectives: $(abs(next_obj - current_obj))\n")
            end

            throw(error("Descent failure!"))
#            output = Dict{ASCIIString, Any}("time" => -1.0, "loss" => -1.0, "iter" => -1, "beta" => fill!(b,Inf))
#            return output
        end
    end # end main loop
end # end function

function iht_path(
    x        :: DenseMatrix{Float32}, 
    y        :: DenseVector{Float32}, 
    path     :: DenseVector{Int}; 
    b        :: DenseVector{Float32} = zeros(Float32,size(x,2)), 
    tol      :: Float32               = 1f-4,
    max_iter :: Int                   = 1000, 
    max_step :: Int                   = 50, 
    quiet    :: Bool                  = true 
)

    # size of problem?
    (n,p) = size(x)

    # how many models will we compute?
    num_models = length(path)         

    # preallocate space for intermediate steps of algorithm calculations 
    b0       = zeros(Float32,p)               # previous iterate beta0 
    df       = zeros(Float32,p)               # (negative) gradient 
    r        = zeros(Float32,n)               # for || Y - XB ||_2^2
    Xb       = zeros(Float32,n)               # X*beta 
    Xb0      = zeros(Float32,n)               # X*beta0 
    tempn    = zeros(Float32,n)               # temporary array of n floats 
    indices  = collect(1:p)                   # indices that sort beta 
    support  = falses(p)                      # indicates nonzero components of beta
    support0 = copy(support)                  # store previous nonzero indicators
    betas    = spzeros(Float32,p,num_models)  # a matrix to store calculated models

    # compute the path
    for i = 1:num_models
    
        # model size?
        q = path[i]

        # store projection of beta onto largest k nonzeroes in magnitude 
        bk     = zeros(Float32,q)
        project_k!(b, bk, indices, q)

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        xk     = zeros(Float32,n,q)           # store q columns of X
        tempkf = zeros(Float32,q)             # temporary array of q floats 
        gk     = zeros(Float32,q)             # another temporary array of q floats 

        # now compute current model
        output = L0_reg(x,y,q, n=n, p=p, b=b, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, xk=xk, r=r, Xb=Xb, Xb=Xb0, b0=b0, df=df, tempkf=tempkf, gk=gk, tempn=tempn, indices=indices, support=support, support0=support0) 

        # extract and save model
        copy!(b, output["beta"])
        update_indices!(support, b, p=p)    
        fill!(support0, false)
#        update_col!(betas, b, i, n=p, p=num_models, a=1.0) 
        betas[:,i] = sparsevec(b)
    end

    # return a sparsified copy of the models
#    return sparse(betas)
    return betas
end 
