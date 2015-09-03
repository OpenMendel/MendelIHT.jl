# COMPUTE WEIGHTED RESIDUALS (Y - 1/2 - diag(W)XB) IN LOGISTIC REGRESSION
# This subroutine, in contrast to the previous update_residuals!() function, 
# will compute WEIGHTED residuals in ONE pass.
# For optimal performance, feed it a precomputed vector of x*b.
# This variant accepts a BEDFile object for the argument x.
# 
# Arguments:
# -- r is the preallocated vector of n residuals to overwrite.
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- y is the n-vector of responses.
# -- b is the p-vector of effect sizes.
# -- perm is the p-vector that indexes b.
# -- w is the n-vector of residual weights.
#
# Optional Arguments:
# -- Xb is the n-vector x*b of predicted responses.
#    If X*b is precomputed, then this function will compute the residuals much more quickly.
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_residuals!(r::DenseArray{Float64,1}, x::BEDFile, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, w::DenseArray{Float64,1}, k::Int; Xb::DenseArray{Float64,1} = xb(x,b,perm,k), n::Int = length(y), p::Int = length(b))
    (n,p) == size(x) || throw(DimensionMismatch("update_residuals!: nonconformable arguments!"))

    @sync @inbounds @parallel for i = 1:n 
        r[i] = y[i] - 0.5 - w[i] * Xb[i] 
    end 

    return r
end


# UPDATE WEIGHTS FOR SURROGATE FUNCTION IN LOGISTIC REGRESSION FOR ENTIRE GWAS
#
# This function calculates a vector of weights
#
#     w = 0.5*diag( tanh(0.5 * x * b) ./ x*b )
#
# for the logistic loglikelihood surrogate function. 
# Note that w is actually defined as 0.25 for each component of x*b that equals zero,
# even though the formula above would yield an undefined quantity.
#
# Arguments:
# -- w is the n-vector of weights for the predicted responses.
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- xb is the n-vector x*b of predicted responses. 
#    If x*b is precomputed, then this function will compute the weights much more quickly.
# -- n is the number of samples. Defaults to length(w).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_weights!(w::DenseArray{Float64,1}, x::BEDFile, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, k::Int; Xb::DenseArray{Float64,1} = xb(x,b,perm,k), n::Int = length(w), p::Int = length(b))
    (n,p) == size(x) || throw(DimensionMismatch("update_weights!: nonconformable arguments!"))

    @sync @inbounds @parallel for i = 1:n 
        w[i] = ifelse(xb[i] == 0.0, 0.25, 0.5*tanh(0.5*xb[i]) / xb[i]) 
    end 

    return w
end

# COMPUTE THE LOGISTIC LOGLIKELIHOOD (Y - 0.5)'XB - LOG(COSH(0.5*XB)) FOR GWAS DATA
# This subroutine computes the logistic likelihood in one pass.
# For optimal performance, supply this function with a precomputed x*b. 
# 
# Arguments:
# -- y is the n-vector of responses
# -- x is the BEDFile that contains the compressed n x p design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- xb is the n-vector x*b of predicted responses
#    If x*b is precomputed, then this function will compute the loglikelihood more quickly. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function compute_loglik(y::DenseArray{Float64,1}, x::BEDFile, b::DenseArray{Float64,1}, perm::DenseArray{Int,1}, k::Int; n::Int = length(y), Xb::DenseArray{Float64,1} = xb(x,b,perm,k), p::Int = length(b))
    n == length(xb) || throw(DimensionMismatch("compute_loglik: y and X*b must have same length!"))

    # each part accumulates sum s
    s = @sync @inbounds @parallel (+) for i = 1:n
        y[i]*xb[i] - softplus(xb[i])
    end

    return s
end
