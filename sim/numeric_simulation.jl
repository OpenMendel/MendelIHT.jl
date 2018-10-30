using IHT, PLINK    # PLINK.jl handles PLINK files
n      = 5000       # number of cases
p      = 23999      # number of predictors
k      = 10         # number of true predictors
s      = 0.1        # standard deviation of noise
srand(2016)         # fix random seed, reproducible x_temp
x_temp = randn(n,p) # simulated data

#Since we want our simulation to be reproducible, configure $\boldsymbol{\beta}$ with a fixed random seed:
b      = zeros(p) # vector to house statistical model
b[1:k] = randn(k) # k random nonzero coefficients drawn from standard normal distribution
shuffle!(b)       # random indices for nonzero coefficients
bidx   = find(b)  # save locations of nonzero coefficients

#Now we make a noisy response y:
y = x_temp*b + s*randn(n) # this makes response vector with noise level drawn from N(0,0.01)

#Next we configure a regression problem. In this case, we need a data matrix with a grand mean included:
x = zeros(n,p+1)             # array to house x_temp + grand mean
setindex!(x, x_temp, :, 1:p) # (efficiently) assign x_temp to correct location in x
x[:,end] = 1.0               # rightmost column (24,000th one) now contains grand mean
x                            # check that x is correct!

#run IHT
output = L0_reg(x,y,k)  # run IHT
bk = copy(output.beta)  # copy the beta for later use
[b[bidx] bk[bidx]]      # did we get the correct model and coefficient values?