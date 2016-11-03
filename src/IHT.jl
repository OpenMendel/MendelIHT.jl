module IHT


using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using RegressionTools
#using StatsBase
using StatsFuns: logistic, logit, softplus
using DataFrames
using Gadfly

### idea from Julia Hoffimann Mendes to conditionally load OpenCL module
# only load if Julia can find OpenCL module
# otherwise warn and set "cl" variable to Void
# will load GPU code based on value of "cl"
try
    using OpenCL
catch e
    warn("IHT.jl cannot find an OpenCL library and will not load GPU functions correctly.")
    global cl = nothing
end

# since PLINK module also loads OpenCL, the previous test can fail
# check that PLINK actually loaded GPU code
try
    PLINK.PlinkGPUVariables
catch e
    warn("PLINK.jl could not find an OpenCL library, so IHT.jl will not load GPU functions correctly.")
    global cl = nothing
end


# used for pretty printing of IHTResults, IHTCrossvalidationResults
import Base.show

# used to plot MSEs v. models from IHTCrossvalidationResults
import Gadfly.plot

export L0_reg
export L0_log
export iht_path
export iht_path_log
export cv_iht
export cv_get_folds
export cv_log

# IHT will only work on single/double precision floats!
typealias Float Union{Float64, Float32}

include("common.jl")
if cl != nothing 
    include("gpu.jl") # conditional load of GPU code
end
include("gwas.jl")
include("cv.jl")
include("hardthreshold.jl")
include("aiht.jl")
include("log.jl")

end # end module IHT
