"""
ITERATIVE HARD THRESHOLDING

    iht!(v, x, y, k [, iter=1, nstep=50]) -> (Float, Int)

This function computes a hard threshold update

    β = P_Sk(β0 + μ*x'*(y - x*β0))

where `μ` is the step size (or learning rate) and P_Sk denotes the projection onto the set S_k defined by

S_k = { x in R^p : || x ||_0 <= k }.

The projection in question preserves the largest `k` components of `β` in magnitude, and it sends the remaining
`p - k` components to zero. This update is intimately related to a projected gradient step used in Landweber iteration.
Unlike the Landweber method, this function performs a line search on `μ` whenever the step size exceeds a specified
threshold `ω` given by

    ω = sumabs2(β - β0) / sumabs2(x*(β - β0))

By backtracking on `μ`, this function guarantees a stable estimation of a sparse `β`.
This function is based on the [HardLab](http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html/) demonstration code written in MATLAB by Thomas Blumensath.

Arguments:

- `v` is the `IHTVariables` object of temporary arrays
- `x` is the `n` x `p` design matrix.
- `y` is the vector of `n` responses.
- `k` is the model size.
- `iter` is the current iteration count in the IHT algorithm. Defaults to `1`.
- `nstep` is the maximum permissible number of backtracking steps. Defaults to `50`.

Output:

- `μ` is the step size used to update `β`, after backtracking.`
- `μ_step` is the number of backtracking steps used on `μ`.
"""
function iht!{T <: Float}(
    v     :: IHTVariables{T},
    x     :: DenseMatrix{T},
    y     :: DenseVector{T},
    k     :: Int,
    iter  :: Int = 1,
    nstep :: Int = 50
)

    # compute indices of nonzeroes in beta
    _iht_indices(v, k)

    # store relevant columns of x
    # need to do this on 1st iteration
    # afterwards, only do if support changes
    if !isequal(v.idx, v.idx0) || iter < 2
        #update_xk!(v.xk, x, v.idx)   # xk = x[:,v.idx]
        copy!(v.xk, view(x, :, v.idx))
    end

    # store relevant components of gradient
    fill_perm!(v.gk, v.df, v.idx)    # gk = g[v.idx]

    # now compute subset of x*g
    A_mul_B!(v.xgk, v.xk, v.gk)

    # warn if xgk only contains zeros
    all(v.xgk .== zero(T)) && warn("Entire active set has values equal to 0")

    # compute step size
#    mu = _iht_stepsize(v, k) :: T # does not yield conformable arrays?!
    μ = (sumabs2(v.gk) / sumabs2(v.xgk)) :: T

    # check for finite stepsize
    isfinite(μ) || throw(error("Step size is not finite, is active set all zero?"))

    # compute gradient step
    _iht_gradstep(v, μ, k)

    # update xb
    update_xb!(v.xb, x, v.b, v.idx, k)
    #A_mul_B!(v.xb, view(x, :, v.idx), view(v.b, v.idx) )

    # calculate omega
    ω_top, ω_bot = _iht_omega(v)

    # backtrack until mu < omega and until support stabilizes
    μ_step = 0
    while _iht_backtrack(v, ω_top, ω_bot, μ, μ_step, nstep)

        # stephalving
        μ /= 2

        # recompute gradient step
        copy!(v.b,v.b0)
        _iht_gradstep(v, μ, k)

        # recompute xb
        update_xb!(v.xb, x, v.b, v.idx, k)
        #A_mul_B!(v.xb, view(x, :, v.idx), view(v.b, v.idx) )

        # calculate omega
        ω_top, ω_bot = _iht_omega(v)

        # increment the counter
        μ_step += 1
    end

    return μ::T, μ_step::Int
end

"""
L0 PENALIZED LEAST SQUARES REGRESSION

    L0_reg(x,y,k) -> IHTResults

This routine minimizes the loss function

    0.5*sumabs2( y - x*β )

subject to `β` lying in the set S_k = { x in R^p : || x ||_0 <= k }.

It uses Thomas Blumensath's iterative hard thresholding framework to keep `β` feasible.

Arguments:

- `x` is the `n` x `p` data matrix.
- `y` is the `n`-dimensional continuous response vector.
- `k` is the desired model size (support).

Optional Arguments:

- `max_iter` is the maximum number of iterations for the algorithm. Defaults to `1000`.
- `max_step` is the maximum number of backtracking steps for the step size calculation. Defaults to `50`.
- `tol` is the global tolerance. Defaults to `1e-4`.
- `quiet` is a `Bool` that controls algorithm output. Defaults to `true` (no output).
- `v` is an `IHTVariables` structure used to house temporary arrays. Used primarily in `iht_path` for memory efficiency.

Outputs are wrapped into an `IHTResults` structure with the following fields:

- 'time' is the compute time for the algorithm. Note that this does not account for time spent initializing optional argument defaults.
- 'iter' is the number of iterations that the algorithm took before converging.
- 'loss' is the optimal loss (half of residual sum of squares) at convergence.
- 'beta' is the final estimate of `b`.
"""
function L0_reg{T <: Float, V <: DenseVector}(
    x        :: DenseMatrix{T},
    y        :: V,
    k        :: Int;
    v        :: IHTVariables{T, V} = IHTVariables(x, y, k),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
)

    # start timer
    tic()

    # first handle errors
    k        >= 0      || throw(ArgumentError("Value of k must be nonnegative!\n"))
    max_iter >= 0      || throw(ArgumentError("Value of max_iter must be nonnegative!\n"))
    max_step >= 0      || throw(ArgumentError("Value of max_step must be nonnegative!\n"))
    tol      >  eps(T) || throw(ArgumentError("Value of global tol must exceed machine precision!\n"))

    # initialize return values
    iter   = 0                    # number of iterations of L0_reg
    exec_time   = zero(T)           # compute time *within* L0_reg
    next_obj  = zero(T)           # objective value
    next_loss = zero(T)           # loss function value

    # initialize floats
    current_obj = convert(T, Inf) # tracks previous objective function value
    the_norm    = zero(T)         # norm(b - b0)
    scaled_norm = zero(T)         # the_norm / (norm(b0) + 1)
    μ           = zero(T)          # Landweber step size, 0 < tau < 2/rho_max^2

    # initialize integers
    i      = 0                    # used for iterations in loops
    μ_step = 0                    # counts number of backtracking steps for mu

    # initialize booleans
    converged = false             # scaled_norm < tol?

    # update xb, r, gradient
    initialize_xb_r_grad!(v, x, y, k)

    # update loss and objective
    next_loss = convert(T, Inf)

    # formatted output to monitor algorithm progress
    !quiet && print_header()

    # main loop
    for iter = 1:max_iter

        # notify and break if maximum iterations are reached.
        if iter >= max_iter

            # alert about hitting maximum iterations
            !quiet && print_maxiter(max_iter, loss)

            # send elements below tol to zero
            threshold!(v.b, tol)

            # stop timer
            exec_time = toq()

            # these are output variables for function
            # wrap them into a Dict and return
            #return IHTResults(exec_time, next_loss, iter, copy(v.b))
            return IHTResults(exec_time, next_loss, iter, v.b)
        end

        # save values from previous iterate
        copy!(v.b0, v.b)             # b0 = b
        copy!(v.xb0, v.xb)           # Xb0 = Xb
        current_obj = next_obj

        # now perform IHT step
        (μ, μ_step) = iht!(v, x, y, k, max_step, iter)

        # the IHT kernel gives us an updated x*b
        # use it to recompute residuals and gradient
        update_r_grad!(v, x, y)

        # update loss, objective, and gradient
        next_loss = sumabs2(v.r) / 2

        # guard against numerical instabilities
        check_finiteness(next_loss)

        # track convergence
        the_norm    = chebyshev(v.b, v.b0)
        scaled_norm = (the_norm / ( norm(v.b0,Inf) + 1)) :: T
        converged   = (scaled_norm < tol) :: Bool

        # output algorithm progress
        quiet || @printf("%d\t%d\t%3.7f\t%3.7f\t%3.7f\n", iter, μ_step, μ, the_norm, next_loss)

        # check for convergence
        # if converged and in feasible set, then algorithm converged before maximum iteration
        # perform final computations and output return variables
        if converged

            # send elements below tol to zero
            threshold!(v.b, tol)

            # stop time
            exec_time = toq()

            # announce convergence
            !quiet && print_convergence(iter, next_loss, exec_time)

            # these are output variables for function
            # wrap them into a Dict and return
            #return IHTResults(exec_time, next_loss, iter, copy(v.b))
            return IHTResults(exec_time, next_loss, iter, v.b)
        end

        # algorithm is unconverged at this point.
        # if algorithm is in feasible set, then rho should not be changing
        # check descent property in that case
        # if rho is not changing but objective increases, then abort
        if next_obj > current_obj + tol
            !quiet && print_descent_error(iter, loss, next_loss)
            throw(ErrorException("Descent failure!"))
            #return IHTResults(-one(T), convert(T, -Inf), -1, [convert(T, -Inf)])
        end
    end # end main loop

    # return null result
    return IHTResults(zero(T), zero(T), 0, [zero(T)])
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
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
)

    # size of problem?
    (n,p) = size(x)

    # how many models will we compute?
    num_models = length(path)
    #nk = sum(path) # total number of elements in models

    # preallocate space for intermediate steps of algorithm calculations
    v  = IHTVariables(x, y, 1)
    betas = spzeros(T, p, num_models)  # a matrix to store calculated models
    #I = zeros(Int, nk)
    #J = zeros(Int, nk)
    #K = zeros(T,   nk)
    #q0 = 0

    # compute the path
    for i = 1:num_models

        # model size?
        q = path[i]

        # monitor progress
        quiet || print_with_color(:blue, "Computing model size $q.\n\n")

        # store projection of beta onto largest k nonzeroes in magnitude
        project_k!(v.b, q)

        # these arrays change in size from iteration to iteration
        # we must allocate them for every new model size
        update_variables!(v, x, q)

        # now compute current model
        output = L0_reg(x, y, q, v=v, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

        # extract and save model
        # strangely more efficient to continuously reallocate sparse matrix
        # than to form it directly
        #I[(q0+1):(q0+q)] = find(v.idx)
        #J[(q0+1):(q0+q)] = i
        #K[(q0+1):(q0+q)] = v.b[v.idx]
        #q0 += q
        betas[:,i] = sparsevec(output.beta)
    end

    # return a sparsified copy of the models
    #betas = sparse(I, J, K, p, num_models, false)
    return betas
end
