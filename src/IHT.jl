module IHT 

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using RegressionTools
using OpenCL
using StatsBase

export L0_reg
export L0_log
export iht_path
export iht_path_log
export cv_iht
export cv_get_folds
export cv_log

typealias Float Union{Float64, Float32}

include("gpu.jl")
include("gwas.jl")
include("cv.jl")
include("iht.jl")
include("aiht.jl")
include("log.jl")

end # end module IHT
