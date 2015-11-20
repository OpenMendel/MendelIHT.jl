module IHT

using Distances: euclidean, chebyshev, sqeuclidean
using PLINK
using RegressionTools
using OpenCL

export L0_reg
export L0_log
export iht_path
export iht_path_log
export cv_iht
export cv_get_folds

include("aiht64.jl")
include("aiht32.jl")
include("cv64.jl")
include("cv32.jl")
include("gwas64.jl")
include("gwas32.jl")
include("iht64.jl")
include("iht32.jl")
include("logistic64.jl")
include("logistic32.jl")
include("gpu64.jl")
include("gpu32.jl")

end # end module IHT
