using PLINK
using IHT

#function tutorial_simulation()

    # problem dimensions
    n = 5000    # number of cases
    p = 23999   # number of predictors
    k = 10      # number of true predictors
    s = 0.1     # standard deviation of noise
    T = Float64 # bits type used for SharedArrays

    # make simulated float data
    srand(2016)         # fix random seed, reproducible x_temp
    x_temp = randn(n,p) # simulated data

    # make simulated model
    b = zeros(p)      # vector to house statistical model
    b[1:k] = randn(k) # k random nonzero coefficients drawn from standard normal distribution
    shuffle!(b)       # random indices for nonzero coefficients
    bidx = find(b)    # save locations of nonzero coefficients

    # simulate a noisy response variable
    # this makes response vector with noise level drawn from N(0,0.01)
    y = x_temp*b + s*randn(n)

    # add grand mean to data
    X = zeros(n,p+1)             # array to house x_temp + grand mean
    setindex!(X, x_temp, :, 1:p) # (efficiently) assign x_temp to correct location in x
    X[:,end] = 1.0               # rightmost column (24,000th one) now contains grand mean

    # make noisier response variable
    s  = 5.0                   # very noisy!
    y2 = x_temp*b + s*randn(n) # y2 has noise distribution N(0,25), substantially noiser than y

    # simulate (or recover from PLINK.jl) some GWAS data
    fpath = expanduser("~/.julia/v0.6/PLINK/data/")         # path to simulated data from PLINK module
    genopath  = fpath * "x_test.bed"                        # path to original BED file
    tgenopath = fpath * "xt_test.bed"                       # path to transposed BED file
    covpath   = fpath * "covfile.txt"                       # path to covariate file
    xbed  = BEDFile(genopath, tgenopath, covpath, pids=[1]) # load the data to master process

    n,p    = size(xbed)        # dimensions of data
    fill!(xbed.covar.x, 1)     # this fills the grand mean with ones
    fill!(xbed.covar.xt, 1)    # this fills the *transposed* grand mean with ones; remember to always change both!
    mean!(xbed)                # compute means in-place
    prec!(xbed)                # compute precisions in-place
    xbed.means[end] = 0        # index "end" substitutes for position of grand mean in x.means!
    xbed.precs[end] = 1        # same as above

    # now simulate model, response with GWAS data
    bbed      = SharedArray{T}((p,), pids=[1])   # a model b to use with the BEDFile
    bbed[1:k] = randn(k)                         # random coefficients
    shuffle!(bbed)                               # random model
    bidxbed   = find(bbed)                       # store locations of nonzero coefficients
    idx       = bbed .!= 0                       # need BitArray indices of nonzeroes in b for A_mul_B
    xb        = A_mul_B(xbed, bbed, idx, k)      # compute x*b
    ybed2     = xb + 0.1*randn(n)                # yields a Vector, so we must convert it to SharedVector
    ybed      = SharedVector{T}((n,), pids=[1])  # our response variable with the BEDFile
    copy!(ybed, ybed2)
    ypath     = ENV["TMPDIR"] * "y.bin"          # ybed uses path to Julia TMP directory
    write(open(ypath, "w"), ybed)                # "w"rite ybed to file

    # crossvalidation parameters
    nfolds = 5     # number of crossvalidation folds
    srand(2016)    # reset seed before crossvalidation

#    return X, y, k, b, bidx, y2, xbed, ybed, bbed, bidxbed, nfolds, xpath, ypath, covpath
#end

#X, y, k, b, bidx, y2, xbed, ybed, bbed, bidxbed, nfolds, xpath, ypath, covpath = tutorial_simulation()
println("Simulation complete.")
