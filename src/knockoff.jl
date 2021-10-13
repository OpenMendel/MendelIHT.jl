function cv_iht_knockoff(
    y        :: AbstractVecOrMat{T},
    x        :: AbstractMatrix{T},
    z        :: AbstractVecOrMat{T},
    original :: AbstractVector{Int},
    knockoff :: AbstractVector{Int},
    fdr      :: Number;
    combine_beta::Bool = false,
    group_ko :: Union{Nothing, Vector{Int}} = nothing,
    d        :: Distribution = is_multivariate(y) ? MvNormal(T[]) : Normal(),
    l        :: Link = IdentityLink(),
    path     :: AbstractVector{<:Integer} = 1:20,
    q        :: Int64 = 5,
    est_r    :: Symbol = :None,
    group    :: AbstractVector{Int} = Int[],
    weight   :: AbstractVector{T} = T[],
    zkeep    :: BitVector = trues(size(y, 2) > 1 ? size(z, 1) : size(z, 2)),
    folds    :: AbstractVector{Int} = rand(1:q, is_multivariate(y) ? size(x, 2) : size(x, 1)),
    debias   :: Bool = false,
    verbose  :: Bool = true,
    max_iter :: Int = 100,
    min_iter :: Int = 5,
    init_beta :: Bool = false
    ) where T <: Float

    typeof(x) <: AbstractSnpArray && throw(ArgumentError("x is a SnpArray! Please convert it to a SnpLinAlg first!"))
    check_data_dim(y, x, z)
    verbose && print_iht_signature()

    # preallocated arrays for efficiency
    test_idx  = [falses(length(folds)) for i in 1:Threads.nthreads()]
    train_idx = [falses(length(folds)) for i in 1:Threads.nthreads()]
    V = [initialize(x, z, y, 1, 1, d, l, group, weight, est_r, false, zkeep) for i in 1:Threads.nthreads()]

    # for displaying cross validation progress
    pmeter = Progress(q * length(path), "Cross validating...")

    # cross validate. TODO: wrap pmap with batch_size keyword to enable distributed CV
    combinations = allocate_fold_and_k(q, path)
    mses = zeros(length(combinations))
    ThreadPools.@qthreads for i in 1:length(combinations)
        fold, sparsity = combinations[i]

        # assign train/test indices
        id = Threads.threadid()
        v = V[id]
        test_idx[id]  .= folds .== fold
        train_idx[id] .= folds .!= fold

        # run IHT on training data with current (fold, sparsity)
        v.k = sparsity
        init_iht_indices!(v, init_beta, train_idx[id])
        fit_iht!(v, debias=debias, verbose=false, max_iter=max_iter, min_iter=min_iter)

        # predict on validation data
        v.cv_wts[train_idx[id]] .= zero(T)
        v.cv_wts[test_idx[id]] .= one(T)
        knockoff_validate!(v, fdr, original, knockoff, combine_beta, group_ko)
        mses[i] = predict!(v)

        # update progres
        next!(pmeter)
    end

    # weight mses for each fold by their size before averaging
    mse = meanloss(mses, q, folds)

    # find best model size and print cross validation result
    k = path[argmin(mse)] :: Int
    verbose && print_cv_results(mse, path, k)

    return mse
end

function knockoff_validate!(v::IHTVariable, fdr::Number, original::AbstractVector{Int}, 
    knockoff::AbstractVector{Int}, combine_beta::Bool,
    group_ko::Union{Nothing, AbstractVector{Int}})
    β_new = zeros(length(v.b))
    if isnothing(group_ko)
        β_new[original] .= extract_beta(v.b, fdr, original, knockoff,
            :knockoff, combine_beta)
    else
        β_new[original] .= extract_beta(v.b, fdr, group_ko, original, knockoff,
            :knockoff)
    end
    v.b .= β_new
    # also update support index
    v.idx .= v.b .!= 0
    check_covariate_supp!(v)
    copyto!(v.xk, @view(v.x[:, v.idx]))
end
