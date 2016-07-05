# subroutine to update residuals and gradient from data
function update_r_grad!{T}(
    v :: IHTVariables{T},
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    difference!(v.r, y, v.xb)
    BLAS.gemv!('T', one(T), x, v.r, zero(T), v.df)
    return nothing
end

function initialize_xb_r_grad!{T <: Float}(
    temp :: IHTVariables{T},
    x    :: DenseMatrix{T},
    y    :: DenseVector{T},
    k    :: Int
)
    # update x*beta
    if sum(temp.idx) == 0
        fill!(temp.xb, zero(T))
    else
        update_indices!(temp.idx, temp.b)
        update_xb!(temp.xb, x, temp.b, temp.idx, k)
    end

    # update r and gradient
    update_r_grad!(temp, x, y)
end

"""
ITERATIVE HARD THRESHOLDING

    iht!(v,x,y,k) -> (Float, Int)

This function computes a hard threshold update

    b = P_Sk(b0 + mu*x'*(y - x*b0))

where `mu` is the step size (or learning rate) and P_Sk denotes the projection onto the set S_k defined by

S_k = { x in R^p : || x ||_0 <= k }.

The projection in question preserves the largest `k` components of `b` in magnitude, and it sends the remaining
`p - k` components to zero. This update is intimately related to a projected gradient step used in Landweber iteration.
Unlike the Landweber method, this function performs a line search on `mu` whenever the step size exceeds a specified
threshold `omega` given by

    omega = sumabs2(b - b0) / sumabs2(x*(b - b0))

By backtracking on `mu`, this function guarantees a stable estimation of a sparse `b`.
This function is based on the [HardLab](http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html/) demonstration code written in MATLAB by Thomas Blumensath.

Arguments:

- `v` is the `IHTVariables` object of temporary arrays 
- `x` is the `n` x `p` design matrix.
- `y` is the vector of `n` responses.
- `k` is the model size.

Optional Arguments:

- iter is the current iteration count in the IHT algorithm. Defaults to `1`.
- max_step is the maximum permissible number of backtracking steps. Defaults to `50`.

Output:

- `mu` is the step size used to update `b`, after backtracking.`
- `mu_step` is the number of backtracking steps used on `mu`.
"""
function iht!{T <: Float}(
    v     :: IHTVariables{T},
    x     :: DenseMatrix{T},
    y     :: DenseVector{T},
    k     :: Int;
    iter  :: Int = 1,
    nstep :: Int = 50,
)

    # compute indices of nonzeroes in beta
    _iht_indices(v, k)

    # store relevant columns of x
    # need to do this on 1st iteration
    # afterwards, only do if support changes
    if !isequal(v.idx, v.idx0) || iter < 2
        update_xk!(v.xk, x, v.idx)   # xk = x[:,v.idx]
    end

    # store relevant components of gradient
    fill_perm!(v.gk, v.df, v.idx)    # gk = g[v.idx]

    # now compute subset of x*g
    BLAS.gemv!('N', one(T), v.xk, v.gk, zero(T), v.xgk)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size
    mu = (sumabs2(v.gk) / sumabs2(v.xgk)) :: T 

#    # compute step size
#    mu = _iht_stepsize(v, k) :: T # does not yield conformable arrays?! 
    isfinite(mu) || throw(error("Step size is not finite, is active set all zero?"))

    # compute gradient step
    _iht_gradstep(v, mu, k)

    # update xb
    update_xb!(v.xb, x, v.b, v.idx, k)

    # calculate omega
    omega_top, omega_bot = _iht_omega(v)

    # backtrack until mu sits below omega and support stabilizes
    mu_step = 0
    while _iht_backtrack(v, omega_top, omega_bot, mu, mu_step, nstep) 

        # stephalving
        mu /= 2

        # recompute gradient step
        copy!(v.b,v.b0)
        _iht_gradstep(v, mu, k)

        # recompute xb
        update_xb!(v.xb, x, v.b, v.idx, k)

        # calculate omega
        omega_top, omega_bot = _iht_omega(v)

        # increment the counter
        mu_step += 1
    end

    return mu::T, mu_step::Int
end

"""
L0 PENALIZED LEAST SQUARES REGRESSION

    L0_reg(x,y,k) -> Dict{ASCIIString,Any}

This routine minimizes the loss function

    0.5*sumabs2( y - x*b )

subject to `b` lying in the set S_k = { x in R^p : || x ||_0 <= k }.

It uses Thomas Blumensath's iterative hard thresholding framework to keep `b` feasible.

Arguments:

- `x` is the `n` x `p` data matrix.
- `y` is the `n`-dimensional continuous response vector.
- `k` is the desired model size (support).

Optional Arguments:

- `max_iter` is the maximum number of iterations for the algorithm. Defaults to `1000`.
- `max_step` is the maximum number of backtracking steps for the step size calculation. Defaults to `50`.
- `tol` is the global tolerance. Defaults to `1e-4`.
- `quiet` is a `Bool` that controls algorithm output. Defaults to `true` (no output).
- `temp` is an `IHTVariables` structure used to house temporary arrays. Used primarily in `iht_path` for memory efficiency. 

Outputs are wrapped into an `IHTResults` structure with the following fields:

- 'time' is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults.
- 'iter' is the number of iterations that the algorithm took before converging.
- 'loss' is the optimal loss (half of residual sum of squares) at convergence.
- 'beta' is the final estimate of `b`.
"""
function L0_reg{T <: Float}(
    x         :: DenseMatrix{T},
    y         :: DenseVector{T},
    k         :: Int;
    temp      :: IHTVariables{T}  = IHTVariables(x, y, k),
    tol       :: Float            = convert(T, 1e-4),
    max_iter  :: Int              = 100,
    max_step  :: Int              = 50,
    quiet     :: Bool             = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0     || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0     || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0     || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps() || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    # initialize return values
    mm_iter   = 0                 # number of iterations of L0_reg
    mm_time   = zero(T)           # compute time *within* L0_reg
    next_obj  = zero(T)           # objective value
    next_loss = zero(T)           # loss function value

    # initialize floats
    current_obj = oftype(tol,Inf) # tracks previous objective function value
    the_norm    = zero(T)         # norm(b - b0)
    scaled_norm = zero(T)         # the_norm / (norm(b0) + 1)
    mu          = zero(T)         # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i       = 0                   # used for iterations in loops
    mu_step = 0                   # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

#    # update xb, r, gradient 
    initialize_xb_r_grad!(temp, x, y, k)

    # update loss and objective
    next_loss = oftype(tol,Inf)

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    for mm_iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if mm_iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(temp.b, tol)

            # stop timer
            mm_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(temp.b))

            return output
        end

        # save values from previous iterate
        copy!(temp.b0, temp.b)             # b0 = b
        copy!(temp.xb0, temp.xb)           # Xb0 = Xb
        current_obj = next_obj

        # now perform IHT step
        (mu, mu_step) = iht!(temp, x, y, k, nstep=max_step, iter=mm_iter)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient
        update_r_grad!(temp, x, y)

        # update loss, objective, and gradient
        next_loss = sumabs2(temp.r) / 2

        # guard against numerical instabilities
        check_finiteness(next_loss)

        # track convergence
        the_norm    = chebyshev(temp.b, temp.b0)
        scaled_norm = the_norm / ( norm(temp.b0,Inf) + 1)
        converged   = scaled_norm < tol

        # output algorithm progress
        quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", mm_iter, mu_step, mu, the_norm, next_loss)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(temp.b, tol)

            # stop time
            mm_time = toq()

            # announce convergence 
            !quiet && print_convergence(mm_iter, next_loss, mm_time)

            # these are output variables for function
            # wrap them into a Dict and return
            return IHTResults(mm_time, next_loss, mm_iter, copy(temp.b))
        end

        # algorithm is unconverged at this point.
        # if algorithm is in feasible set, then rho should not be changing
        # check descent property in that case
        # if rho is not changing but objective increases, then abort
        if next_obj > current_obj + tol
            !quiet && print_descent_error(mm_iter, loss, next_loss)
            throw(ErrorException("Descent failure!"))
        end
    end # end main loop
end # end function

"""
COMPUTE AN IHT REGULARIZATION PATH FOR LEAST SQUARES REGRESSION

    iht_path(x,y,path) -> SparseCSCMatrix

This subroutine computes best linear models for matrix `x` and response `y` by calling `L0_reg` for each model over a regularization path denoted by `path`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path` is an `Int` vector that contains the model sizes to test.

Optional Arguments:

- `b` is the `p`-vector of effect sizes. This argument permits warmstarts to the path computation. Defaults to `zeros(p)`.
- `tol` is the global convergence tolerance for `L0_reg`. Defaults to `1e-4`.
- `max_iter` caps the number of iterations for the algorithm. Defaults to `1000`.
- `max_step` caps the number of backtracking steps in the IHT kernel. Defaults to `50`.
- `quiet` is a Boolean that controls the output. Defaults to `true` (no output).

Output:

- a sparse `p` x `length(path)` matrix where each column contains the computed model for each component of `path`.
"""
function iht_path{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    path     :: DenseVector{Int};
    tol      :: Float          = convert(T, 1e-4),
    max_iter :: Int            = 1000,
    max_step :: Int            = 50,
    quiet    :: Bool           = true
)

    # size of problem?
    (n,p) = size(x)

    # how many models will we compute?
    num_models = length(path)

    # preallocate space for intermediate steps of algorithm calculations
    temp  = IHTVariables(x, y, 1)
    betas = spzeros(T, p, num_models)  # a matrix to store calculated models

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(temp.b, q)

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(temp, x, q)

        # now compute current model
        output = L0_reg(x, y, q, temp=temp, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

        # extract and save model
        betas[:,i] = sparsevec(output.beta)
    end

    # return a sparsified copy of the models
    return betas
end
