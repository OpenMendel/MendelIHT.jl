"""
This function will create a random SnpArray in the current directory without missing value, 
where each SNP has at least 5 minor alleles. 

n = number of samples
p = number of SNPs
s = name of the simulated SnpArray
min_ma = the minimum number of minor alleles that must be present for each SNP (defaults to 5)
"""
function simulate_random_snparray(n::Int64, p::Int64, s::Union{String, UndefInitializer}; 
                                  min_ma::Int = 5)
    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    A1 = BitArray(undef, n, p) 
    A2 = BitArray(undef, n, p) 
    mafs = zeros(Float64, p)
    for j in 1:p
        minor_alleles = 0
        maf = 0
        while minor_alleles <= min_ma
            maf = 0.5rand()
            for i in 1:n
                A1[i, j] = rand(Bernoulli(maf))
                A2[i, j] = rand(Bernoulli(maf))
            end
            minor_alleles = sum(view(A1, :, j)) + sum(view(A2, :, j))
        end
        mafs[j] = maf
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(A1, A2, s), mafs
end

"""
This function requires a vector of minor alleles frequencies to perform simulation. 
It will create a random SnpArray in the current directory without missing value, 
where each SNP has at least 5 minor alleles. 

n = number of samples
p = number of SNPs
s = name of the simulated SnpArray
mafs = vector of minor allele freuqencies
min_ma = the minimum number of minor alleles that must be present for each SNP (defaults to 5)


Note: if supplied minor allele frequency is extremely small, it could take a long time for 
the simulation to generate samples where at least `min_ma` (defaults to 5) are present. 
"""
function simulate_random_snparray(n::Int64, p::Int64, s::Union{String, UndefInitializer}, 
                                  mafs::Vector{Float64}; min_ma::Int = 5)
    all(0.0 .<= mafs .<= 0.5) || throw(ArgumentError("vector of minor allele frequencies must be in (0, 0.5)"))
    any(mafs .<= 0.0005) && @warn("Provided minor allele frequencies contain entries smaller than 0.0005, simulation may take long if sample size is small and min_ma = $min_ma is large")

    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    A1 = BitArray(undef, n, p) 
    A2 = BitArray(undef, n, p) 
    for j in 1:p
        minor_alleles = 0
        maf = mafs[j]
        while minor_alleles <= min_ma
            for i in 1:n
                A1[i, j] = rand(Bernoulli(maf))
                A2[i, j] = rand(Bernoulli(maf))
            end
            minor_alleles = sum(view(A1, :, j)) + sum(view(A2, :, j))
        end
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(A1, A2, s)
end

"""
Make a SnpArray from 2 BitArrays.
"""
function _make_snparray(A1::BitArray, A2::BitArray, s::Union{String, UndefInitializer})
    n, p = size(A1)
    x = SnpArray(s, n, p)
    for i in 1:(n*p)
        c = A1[i] + A2[i]
        if c == 0
            x[i] = 0x00
        elseif c == 1
            x[i] = 0x02
        elseif c == 2
            x[i] = 0x03
        else
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end

"""
This function simulates a random response vector `y` based on provided x, β, distirbution,
and link function. `k` denotes the true number of predictors. When the distribution is from 
Poisson, Gamma, or Negative Binomial, then the true model is drawn from N(0, 0.3) to roughly 
ensure the mean of response `y` doesn't become too large. 

Note: for negative binomial and gamma, the link function must be LogLink. For Bernoulli,
the probit link seems to work better than logitlink.

`nn` and `α` are optional input argument needed for gamma and negative binomial simulations. 
For Negative binomial, it is the number of success until stopping. For gamma distribution, 
it is the shape parameter. 
"""
function simulate_random_response(x::SnpArray, xbm::SnpBitMatrix, k::Int, 
                                  d::UnionAll, l::Link; nn = 10,
                                  α = 1) where {T <: Float}
    n, p = size(x)
    if (typeof(d) <: NegativeBinomial) || (typeof(d) <: Gamma)
        l == LogLink() || throw(ArgumentError("Distribution $d must use LogLink!"))
    end

    #simulate a random model β from a normal distribution
    true_b = zeros(p)
    if d == Poisson || d == Gamma || d == NegativeBinomial
        true_b[1:k] = rand(Normal(0, 0.3), k)
    else
        true_b[1:k] = randn(k)
    end
    shuffle!(true_b)
    correct_position = findall(x -> x != 0, true_b)

    #simulate phenotypes (e.g. vector y)
    if d == Normal || d == Poisson || d == Bernoulli
        prob = linkinv.(l, xbm * true_b)
        y = [rand(d(i)) for i in prob]
    elseif d == NegativeBinomial
        μ = linkinv.(l, xbm * true_b)
        prob = 1 ./ (1 .+ μ ./ nn)
        y = [rand(d(nn, i)) for i in prob] #number of failtures before nn success occurs
    elseif d == Gamma
        μ = linkinv.(l, xbm * true_b)
        β = 1 ./ μ # here β is the rate parameter for gamma distribution
        y = [rand(d(α, i)) for i in β] # α is the shape parameter for gamma
    end
    y = Float64.(y)

    return y, true_b, correct_position
end

"""
This function makes some columns of x correlated with a specific one, in some rather ad-hoc way.
We check whether the correlation is actually met in the end by printing the correlation 
coefficients afterwards. 

- `x` the snparray
- `ρ` correlation coefficient
- `pos` the position of the target SNP that everything would be correlated to
- `num` even integer: how many columns of `x` would be made correlated
- `increment` how far should each correlated SNP be located than the one specified in `pos`.
"""
function adhoc_add_correlation(x::SnpArray, ρ::Float64, pos::Int64, location::Vector{Int})
    @assert 0 <= ρ <= 1 "correlation coefficient must be in (0, 1) but was $ρ"

    for loc in location 
        for i in 1:size(x, 1)
            prob = rand(Bernoulli(ρ))
            prob == 1 && (x[i, loc] = x[i, pos]) #make 2nd column the same as 1st ~90% of the time
        end
        corr = cor(x[:, loc], x[:, pos])
        println("for SNP $loc the simulated correlation is $corr")
    end
end

"""
This function simulates case controls based on `v` variants that can independently cause a 
disease. We typically end up with a handful of causal variants and a lot of rare variants. 

This is Algorithm 1 of "Association screening of common and rare genetic variants by
penalized regression" here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3025646/
"""
function simulate_rare_variants(mafs::Vector{Float64}, penetrance::Vector{Float64}, 
                                cases::Int, controls::Int)
    return nothing # not implemented yet
end

"""
This function creates .bim and .bed files from a SnpArray (which can potentially be memory 
mapped to a .bed file).
"""
function make_bim_fam_files(x::SnpArray, y, name::String)
    ly = length(y)
    n, p = size(x)
    @assert n == ly "dimension mismatch: phenotype data has length $ly but SnpArray has $n samples"

    #create .bim file structure: https://www.cog-genomics.org/plink2/formats#bim
    open(name * ".bim", "w") do f
        for i in 1:p
            write(f, "1 \t $i \t 0 \t 1 \t 1 \t 2 \n")
        end
    end

    #create .fam file structure: https://www.cog-genomics.org/plink2/formats#bim
    open(name * ".fam", "w") do f
        for i in 1:n
            yi = y[i]
            write(f, "$i \t 1 \t 0 \t 0 \t 1 \t $yi \n")
        end
    end
end