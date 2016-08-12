module IHT

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using RegressionTools
using OpenCL
using StatsBase
using DataFrames
#using Plots
#using UnicodePlots
using Gadfly

# used for pretty printing of IHTResults, IHTCrossvalidationResults
import Base.show

# used to plot MSEs v. models from IHTCrossvalidationResults
#import Plots.plot
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
include("gpu.jl")
include("gwas.jl")
include("cv.jl")
include("hardthreshold.jl")
include("aiht.jl")
include("log.jl")

end # end module IHT
