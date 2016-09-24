### 10 August 2016: currently this is not used. it has not proven to be faster than IHT itself
### left here just in case it proves useful later
export L0_reg_aiht

# DOUBLE OVERRELAXATION SCHEME TO ACCELERATE IHT
#
# This subroutine computes the (D)ouble (O)ver(R)elaxation for accelerating IHT steps.
function dor{T <: Float}(
    x       :: DenseArray{T,2},
    y       :: DenseArray{T,1},
    b       :: DenseArray{T,1},
    b0      :: DenseArray{T,1},
    b00     :: DenseArray{T,1},
    xb      :: DenseArray{T,1},
    xb0     :: DenseArray{T,1},
    xb00    :: DenseArray{T,1},
#    sortidx :: DenseArray{Int,1},
    IDX     :: BitArray{1},
    z1      :: DenseArray{T,1},
    z2      :: DenseArray{T,1},
    xz1     :: DenseArray{T,1},
    xz2     :: DenseArray{T,1},
    dif     :: DenseArray{T,1},
    r       :: DenseArray{T,1},
    r2      :: DenseArray{T,1},
    obj     :: Float,
    k       :: Int;
    n       :: Int = length(y),
    p       :: Int = length(b)
)

    # calculate first overrelaxation
    difference!(dif, xb, xb0, n=n)                  # dif = xb - xb0
    a1  = dot(dif,r) / sumabs2(dif)                 # a1  = dif'*r / || dif ||
    ypatzmw!(z1, b, a1, b, b0, n=p)                 # z1  = b + a1 * (b - b0)
    difference!(xz1, xb, xb0, n=n, a=1.0+a1, b=a1)  # xz1 = (1+a1)*xb - a1*xb0
    difference!(r2,y,xz1, n=n)                      # r2  = y - xz1

    # calculate second overrelaxation
    difference!(dif, xz1, xb00, n=n)                # dif = xz1 - xb00
    a2 = dot(dif,r2) / sumabs2(dif)                 # a2  = dif'*r2 / || dif ||
    ypatzmw!(z2, z1, a2, z1, b00)                   # z2 = z1 + a2 * (z1 - b00)

    # project z2 onto sparsity set
    project_k!(z2, k)

    # update residual information about z2
#    update_xb!(xz2, x, z2, sortidx, k, n=n, p=p)
    update_xb!(xz2, x, z2, IDX, k, n=n, p=p)
    difference!(r2,y,xz2)

    # if z2 is better than b0, then overwrite b with z2
    if sumabs(r2) < obj * 2
        println("Successful acceleration")
        copy!(b,z2)
        copy!(r,r2)
        copy!(xb,xz2)
    end

    return nothing
end


# ACCELERATED ITERATIVE HARD THRESHOLDING
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
# This variant of IHT uses the double overrelaxation scheme described by Blumensath (2012) to accelerate convergence.
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
# -- b00 is the previous b0. Defaults to b.
# -- xb = x*b.
# -- xb0 = x*b0.
# -- xb00 = x*b00.
# -- xk is a temporary array to store the k columns of x corresponding to the support of b.
# -- gk is a temporary array of k floats used to subset the k components of the gradient g with the support of b.
# -- xgk = x*gk.
# -- max_step is the maximum number of backtracking steps to take. Defaults to 50.
# -- sortidx is a vector to store the indices that would sort beta. Defaults to p zeros of type Int.
# -- betak is a vector to store the largest k values of beta. Defaults to k zeros of type T.
# -- IDX and IDX0 are BitArrays indicating the nonzero status of components of beta. They default to falses.
# -- r and r2 store the overall residual and partial accelerated residual, respectively. They default to zeroes.
# -- z1 and z2 store intermediate steps of the acceleration calculations. They default to zeroes.
# -- xz1 and xz2 store x*z1 and x*z2. They default to zeroes.
# -- dif is a temporary array to store the difference of two vectors
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
# based on the HardLab demonstration code written in MATLAB by Thomas Blumensath
# http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html
function aiht{T <: Float}(
    x         :: DenseArray{T,2},
    y         :: DenseArray{T,1},
    b         :: DenseArray{T,1},
    g         :: DenseArray{T,1},
    obj       :: Float,
    k         :: Int,
    iter      :: Int;
    n         :: Int               = length(y),
    p         :: Int               = length(b),
    xk        :: DenseArray{T,2}   = zeros(T,n,k),
    b0        :: DenseArray{T,1}   = copy(b),
    b00       :: DenseArray{T,1}   = copy(b),
    xb        :: DenseArray{T,1}   = BLAS.gemv('N', one(T), x, b),
    xb0       :: DenseArray{T,1}   = copy(xb),
    xb00      :: DenseArray{T,1}   = copy(xb),
    gk        :: DenseArray{T,1}   = zeros(T,k),
    xgk       :: DenseArray{T,1}   = zeros(T,n),
    r         :: DenseArray{T,1}   = zeros(T,n),
    r2        :: DenseArray{T,1}   = zeros(T,n),
    z1        :: DenseArray{T,1}   = zeros(T,p),
    z2        :: DenseArray{T,1}   = zeros(T,p),
    dif       :: DenseArray{T,1}   = zeros(T,n),
    xz1       :: DenseArray{T,1}   = zeros(T,n),
    xz2       :: DenseArray{T,1}   = zeros(T,n),
#    sortidx   :: DenseArray{Int,1} = collect(1:p),
    IDX       :: BitArray{1}       = falses(p),
    IDX0      :: BitArray{1}       = copy(IDX),
    step_mult :: Float             = one(T),
    max_step  :: Int               = 50
)

    # which components of beta are nonzero?
    update_indices!(IDX, b, p=p)

    # if current vector is 0,
    # then take largest elements of d as nonzero components for b
    if sum(IDX) == 0
#        selectpermk!(sortidx,g,k, p=p)
#        IDX[sortidx[1:k]] = true
        a = select(g, k, by=abs, rev=true)
        threshold!(IDX, g, abs(a), n=p)
    end

    # store relevant columns of x
    # do so only if support has changed after first iteration
    if !isequal(IDX, IDX0) || iter < 2
        update_xk!(xk, x, IDX, k=k, p=p, n=n)   # xk = x[:,IDX]
    end

    # store relevant components of gradient
    fill_perm!(gk, g, IDX, k=k, p=p)    # gk = g[IDX]

    # now compute subset of x*g
    BLAS.gemv!('N', one(T), xk, gk, zero(T), xgk)

    # compute step size
    mu = step_mult * sumabs2(sdata(gk)) / sumabs2(sdata(xgk))
    isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))

    # take gradient step
    BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

    # preserve top k components of b
    project_k!(b, k)

    # which indices of new beta are nonzero?
    copy!(IDX0, IDX)
    update_indices!(IDX, b, p=p)

    # must correct for equal entries at kth pivot of b
    # this is a total hack! but matching magnitudes are very rare
    # should not drastically affect performance, esp. with big data
    # hack randomly permutes indices of duplicates and retains one 
    if sum(IDX) > k 
        a = select(b, k, by=abs, rev=true)          # compute kth pivot
        duples = find(x -> abs(x) .== abs(a), b)    # find duplicates
        c = randperm(length(duples))                # shuffle 
        d = duples[c[2:end]]                        # permute, clipping top 
        b[d] = zero(T)                             # zero out duplicates
        IDX[d] = false                              # set corresponding indices to false
    end 

    # update xb
#    update_xb!(xb, x, b, sortidx, k)
    update_xb!(xb, x, b, IDX, k)

    # update residuals
    difference!(r,y,xb, n=n)

    if iter > 2
#        dor(x, y, b, b0, b00, xb, xb0, xb00, sortidx, z1, z2, xz1, xz2, dif, r, r2, obj, k, n=n, p=p)
        dor(x, y, b, b0, b00, xb, xb0, xb00, IDX, z1, z2, xz1, xz2, dif, r, r2, obj, k, n=n, p=p)
    end

    # calculate omega
    omega_top = sqeuclidean(sdata(b),(b0))
    omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

    # backtrack until mu sits below omega and support stabilizes
    mu_step = 0
    while mu*omega_bot > 0.99*omega_top && sum(IDX) != 0 && sum(IDX $ IDX0) != 0 && mu_step < max_step

        # stephalving
        mu /= 2

        # recompute gradient step
        copy!(b,b0)
        BLAS.axpy!(p, mu, sdata(g), 1, sdata(b), 1)

        # recompute projection onto top k components of b
        project_k!(b, k)

        # which indices of new beta are nonzero?
        update_indices!(IDX, b, p=p)

        # must correct for equal entries at kth pivot of b
        # this is a total hack! but matching magnitudes are very rare
        # should not drastically affect performance, esp. with big data
        # hack randomly permutes indices of duplicates and retains one 
        if sum(IDX) > k 
            a = select(b, k, by=abs, rev=true)          # compute kth pivot
            duples = find(x -> abs(x) .== abs(a), b)    # find duplicates
            c = randperm(length(duples))                # shuffle 
            d = duples[c[2:end]]                        # permute, clipping top 
            b[d] = zero(T)                             # zero out duplicates
            IDX[d] = false                              # set corresponding indices to false
        end 

        # recompute xb
#        update_xb!(xb, x, b, sortidx, k)
        update_xb!(xb, x, b, IDX, k)

        # update residuals
        difference!(r,y,xb, n=n)

        if iter > 2
#            dor(x, y, b, b0, b00, xb, xb0, xb00, sortidx, z1, z2, xz1, xz2, dif, r, r2, obj, k, n=n, p=p)
            dor(x, y, b, b0, b00, xb, xb0, xb00, IDX, z1, z2, xz1, xz2, dif, r, r2, obj, k, n=n, p=p)
        end

        # calculate omega
        omega_top = sqeuclidean(sdata(b),sdata(b0))
        omega_bot = sqeuclidean(sdata(xb),sdata(xb0))

        # increment the counter
        mu_step += 1
    end

    return mu, mu_step
end



# ACCELERATED L0 PENALIZED LEAST SQUARES REGRESSION
#
# This routine solves the optimization problem
#
#     min 0.5*|| Y - XB ||_2^2
#
# subject to
#
#     B in S_k = { x in R^p : || x ||_0 <= k }.
#
# It uses Thomas Blumensath's accelerated iterative hard thresholding framework to keep B feasible.
# Note that the acceleration need not kick in, and this function could run slower than its IHT counterpart.
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
#       Xk        = zeros(n,k)      # store k columns of X
#       r         = zeros(n)        # for || Y - XB ||_2^2
#       Xb        = zeros(n)        # X*beta
#       Xb0       = zeros(n)        # X*beta0
#       Xb00      = zeros(n)        # X*beta00
#       b0        = zeros(p)        # previous iterate beta0
#       b00       = zeros(p)        # previous beta0
#       df        = zeros(p)        # (negative) gradient
#       idx       = zeros(k)        # another temporary array of k floats
#       tempn     = zeros(n)        # temporary array of n floats
#       indices   = collect(1:p)    # indices that sort beta
#       support   = falses(p)       # indicates nonzero components of beta
#       support0  = copy(support)   # store previous nonzero indicators
#       r2 stores the overall residual and partial accelerated residual, respectively. They default to zeroes.
#       z1 and z2 store intermediate steps of the acceleration calculations. They default to zeroes.
#       xz1 and xz2 store x*z1 and x*z2. They default to zeroes.
#       dif is a temporary array to store the difference of two vectors
#
# Outputs are wrapped into a Dict with the following fields:
# -- time is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults
# -- iter is the number of iterations that the algorithm took
# -- loss is the optimal loss (residual sum of squares divided by sqrt of RSS with previous iterate)
# -- beta is the final iterate
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function L0_reg_aiht{T <: Float}(
    X        :: DenseArray{T,2},
    Y        :: DenseArray{T,1},
    k        :: Int;
    n        :: Int               = length(Y),
    p        :: Int               = size(X,2),
    Xk       :: DenseArray{T,2}   = zeros(T,n,k),
    b        :: DenseArray{T,1}   = zeros(T,p),
    b0       :: DenseArray{T,1}   = zeros(T,p),
    b00      :: DenseArray{T,1}   = zeros(T,p),
    df       :: DenseArray{T,1}   = zeros(T,p),
    z1       :: DenseArray{T,1}   = zeros(T,p),
    z2       :: DenseArray{T,1}   = zeros(T,p),
    Xb       :: DenseArray{T,1}   = zeros(T,n),
    Xb0      :: DenseArray{T,1}   = zeros(T,n),
    Xb00     :: DenseArray{T,1}   = zeros(T,n),
    tempn    :: DenseArray{T,1}   = zeros(T,n),
    r        :: DenseArray{T,1}   = zeros(T,n),
    r2       :: DenseArray{T,1}   = zeros(T,n),
    dif      :: DenseArray{T,1}   = zeros(T,n),
    xz1      :: DenseArray{T,1}   = zeros(T,n),
    xz2      :: DenseArray{T,1}   = zeros(T,n),
    idx      :: DenseArray{T,1}   = zeros(T,k),
    indices  :: DenseArray{Int,1} = collect(1:p),
    support  :: BitArray{1}       = falses(p),
    support0 :: BitArray{1}       = falses(p),
    tol      :: Float             = convert(T, 1e-4),
    max_iter :: Int               = 1000,
    max_step :: Int               = 50,
    quiet    :: Bool              = true
)

    # start timer
    tic()

    # first handle errors
    k            >= 0      || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter     >= 0      || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step     >= 0      || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol          >  eps(T) || throw(ArgumentError("Value of global tolerance must exceed machine precision!\n"))

    # initialize return values
    mm_iter   = 0                   # number of iterations of L0_reg
    mm_time   = zero(T)             # compute time *within* L0_reg
    next_obj  = oftype(tol, Inf)    # objective value
    next_loss = oftype(tol, Inf)    # loss function value

    # initialize floats
    current_obj = oftype(tol, Inf)   # tracks previous objective function value
    the_norm    = zero(T)            # norm(b - b0)
    scaled_norm = zero(T)            # the_norm / (norm(b0) + 1)
    mu          = zero(T)            # IHT step size

    # initialize integers
    i       = 0          # used for iterations in loops
    mu_step = 0          # counts number of backtracking steps for mu

    # initialize booleans
    converged = false    # scaled_norm < tol?

    # update X*beta
    update_xb!(Xb, X, b, indices, k, p=p, n=n)

    # update r and gradient
    difference!(r, Y, Xb, n=n)
    BLAS.gemv!('T', one(T), X, r, zero(T), df)

    # update loss and objective
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
            difference!(r, Y, Xb, n=n)

            # calculate loss and objective
            next_loss = sumabs2(r) / 2

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            output = Dict{String, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

            return output
        end

        # save values from previous iterate
        copy!(b00,b0)           # b00  = b
        copy!(b0,b)             # b0   = b
        copy!(Xb00,Xb0)         # Xb00 = Xb0
        copy!(Xb0,Xb)           # Xb0  = Xb
        current_obj = next_obj

        # now perform AIHT step
#        (mu, mu_step) = aiht(X,Y,b,df,current_obj,k,mm_iter, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, sortidx=indices, gk=idx, step_mult=one(T), b00=b00, r=r, r2=r2, z1=z1, z2=z2, dif=dif, xz1=xz1, xz2=xz2)
        (mu, mu_step) = aiht(X,Y,b,df,current_obj,k,mm_iter, n=n, p=p, max_step=max_step, IDX=support, IDX0=support0, b0=b0, xb=Xb, xb0=Xb0, xgk=tempn, xk=Xk, gk=idx, step_mult=one(T), b00=b00, r=r, r2=r2, z1=z1, z2=z2, dif=dif, xz1=xz1, xz2=xz2)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals
        difference!(r, Y, Xb, n=n)

        # finally, recompute the gradient
        BLAS.gemv!('T', one(T), X, r, zero(T), df)

        # update loss, objective, and gradient
        next_loss = sumabs2(r) / 2
        next_obj  = next_loss

        # guard against numerical instabilities
        isnan(next_loss) && throw(error("Loss function is NaN, something went wrong..."))
        isinf(next_loss) && throw(error("Loss function is NaN, something went wrong..."))

        # track convergence
        the_norm    = chebyshev(b,b0)
        scaled_norm = the_norm / ( norm(b0,Inf) + one(T))
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
            output = Dict{String, Any}("time" => mm_time, "loss" => next_loss, "iter" => mm_iter, "beta" => b)

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

            throw(error("Descent failure"))
            output = Dict{String, Any}("time" => -one(T), "loss" => -one(T), "iter" => -1, "beta" => fill!(b,Inf))

            return output
        end
    end # end main loop
end # end function
