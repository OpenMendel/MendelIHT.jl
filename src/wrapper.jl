# TODO: VCF and BGEN read

"""
    iht(plinkfile, k, d, phenotypes=6, covariates="", summaryfile="iht.summary.txt",
        betafile="iht.beta.txt", kwargs...)

Runs IHT with sparsity level `k`. 

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `k`: An `Int` for sparsity parameter = number of none-zero coefficients
- `d`: Distribution of phenotypes. Specify `Normal` for quantitative traits,
    `Bernoulli` for binary traits, `Poisson` or `NegativeBinomial` for
    count traits, and `MvNormal` for multiple quantitative traits. 

# Optional Arguments
- `phenotypes`: Phenotype file name (`String`), an integer, or vector of integer. Integer(s)
    coresponds to the column(s) of `.fam` file that stores phenotypes (default `phenotypes=6`). 
    Enter multiple integers for multivariate analysis (e.g. `phenotypes=[6, 7]`).
    We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
    (univariate or multivariate), missing phenotypes are imputed with the mean. Binary
    and count phenotypes cannot be imputed. Phenotype files are read using `readdlm` function
    in Julia base. We require each subject's phenotype to occupy a different row. The file
    should not include a header line. Each row should be listed in the same order as in
    the PLINK and (for multivariate analysis) be comma separated. 
- `covariates`: Covariate file name. Default `covariates=""` (in which case an intercept
    term will be automatically included). If `covariates` file specified, it will be 
    read using `readdlm` function in Julia base. We require the covariate file to be
    comma separated, and not include a header line. Each row should be listed in the
    same order as in the PLINK. The first column should be all 1s to indicate an
    intercept. All other columns not specified in `exclude_std_idx` will be standardized
    to mean 0 variance 1. 
- `summaryfile`: Output file name for saving IHT's summary statistics. Default
    `summaryfile="iht.summary.txt"`.
- `betafile`: Output file name for saving IHT's estimated genotype effect sizes. 
    Default `betafile="iht.beta.txt"`. 
- `covariancefile`: Output file name for saving IHT's estimated trait covariance
    matrix for multivariate analysis. Default `covariancefile="iht.cov.txt"`. 
- `exclude_std_idx`: Indices of non-genetic covariates that should be excluded from
    standardization. 
- `dosage`: Currently only guaranteed to work for VCF files. If `true`, will read
    genotypes dosages (i.e. `X[i, j] ∈ [0, 2]` before standardizing)
- All optional arguments available in [`fit_iht`](@ref)
"""
function iht(
    plinkfile::AbstractString,
    k::Int,
    d::UnionAll;
    phenotypes::Union{AbstractString, Int, AbstractVector{Int}} = 6,
    covariates::AbstractString = "",
    summaryfile::AbstractString = "iht.summary.txt",
    betafile::AbstractString = "iht.beta.txt",
    covariancefile::AbstractString = "iht.cov.txt",
    exclude_std_idx::AbstractVector{<:Integer} = Int[],
    dosage::Bool = false,
    kwargs...
    )
    # read genotypes
    X, X_sampleID, X_chr, X_pos, X_ids, X_ref, X_alt = parse_genotypes(plinkfile, dosage)
    if typeof(X) <: SnpData
        xla = SnpLinAlg{Float64}(X.snparray, model=ADDITIVE_MODEL, 
            center=true, scale=true, impute=true)
    else
        xla = X # numeric matrix from VCF or BGEN files
    end

    # read phenotypes
    y = parse_phenotypes(X, phenotypes, d())

    # read and standardize covariates 
    z = covariates == "" ? ones(size(xla, 1)) : 
        parse_covariates(covariates, exclude_std_idx, standardize=true)
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    # run IHT
    io = open(summaryfile, "w")
    if is_multivariate(y)
        result = fit_iht(y, Transpose(xla), z, k=k, io=io; kwargs...)
    else
        l = d == NegativeBinomial ? LogLink() : canonicallink(d()) # link function
        result = fit_iht(y, xla, z, k=k, d=d(), l=l, io=io; kwargs...)
    end

    show(io, result)

    if is_multivariate(y)
        writedlm(betafile, result.beta')
        writedlm(covariancefile, result.Σ)
    else
        writedlm(betafile, result.beta)
    end

    close(io)
    flush(io)

    return result
end

# adhoc constructor for empty MvNormal distribution
MvNormal() = MvNormal(Float64[])

"""
    parse_phenotypes(x, col::Union{Int, AbstractVector{Int}}, ::Distribution)

Reads phenotypes to numeric array. If `x` is a `SnpData`, columns `col` of the `.fam`
file will be parsed as phenotypes. Otherwise, will read `x` as comma-separated text
file where each sample occupies a row. We recognize missing phenotypes as `NA` or
`-9`. For quantitative traits (univariate or multivariate), missing phenotypes are
imputed with the mean. Binary and count phenotypes cannot be imputed. 
"""
function parse_phenotypes end

function parse_phenotypes(x::SnpData, col::AbstractVector{Int}, ::MvNormal)
    n = x.people
    r = length(col) # number of traits
    y = Matrix{Float64}(undef, r, n)
    offset = 5

    # impute missing phenotypes "-9" by mean of observed phenotypes
    missing_idx = Int[]
    for c in col
        empty!(missing_idx)
        s = 0.0
        for i in 1:n
            if phenotype_is_missing(x.person_info[i, c])
                y[c - offset, i] = 0.0
                push!(missing_idx, i)
            else
                y[c - offset, i] = parse(Float64, x.person_info[i, c])
                s += y[c - offset, i]
            end
        end
        avg = s / (n - length(missing_idx))
        for i in missing_idx
            y[c - offset, i] = avg
        end
    end
    return y
end

parse_phenotypes(::SnpData, ::Int, ::MvNormal) = 
    throw(ArgumentError("Multivariate analysis requires multiple phenotypes! Please specify " * 
        "e.g. phenotypes=[6, 7] or save each sample's phenotypes in a comma-" * 
        "separated file where each sample occupies a different row and each" * 
        " phenotype is separated by a single comma."))

function parse_phenotypes(x::SnpData, col::Int, ::Normal)
    n = x.people
    y = Vector{Float64}(undef, n)

    # impute missing phenotypes by mean of observed phenotypes
    missing_idx = Int[]
    s = 0.0
    for i in 1:n
        if phenotype_is_missing(x.person_info[i, col])
            y[i] = 0.0
            push!(missing_idx, i)
        else
            y[i] = parse(Float64, x.person_info[i, col])
            s += y[i]
        end
    end
    avg = s / (n - length(missing_idx))
    for i in missing_idx
        y[i] = avg
    end
    return y
end

function parse_phenotypes(x::SnpData, col::Int, ::UnivariateDistribution)
    n = x.people
    y = Vector{Float64}(undef, n)

    # missing phenotypes NOT allowed for binary/count phenotypes
    for i in 1:n
        if phenotype_is_missing(x.person_info[i, col])
            throw(MissingException("Missing phenotype detected for sample $i. Automatic " * 
                "phenotype imputation are only possible for quantitative traits. " * 
                "Please exclude missing phenotypes or impute them first."))
        else
            y[i] = parse(Float64, x.person_info[i, col])
        end
    end
    return y
end

function parse_phenotypes(::Any, pheno_filename::AbstractString, d)
    y = readdlm(pheno_filename, ',', Float64)
    if is_multivariate(y)
        y = convert(Matrix{Float64}, Transpose(y))
    else
        y = dropdims(y, dims=2)
    end
    return y
end

"""
    parse_covariates(filename, exclude_std_idx; standardize::Bool=true)

Reads a comma separated text file `filename`. Each row should be a sample ordered the 
same as in the plink file. The first column should be array of 1 (representing
intercept). Each covariate should be comma separated. If `standardize=true`, 
all columns except those in `exclude_std_idx` will be standardized. 
"""
function parse_covariates(filename::AbstractString, exclude_std_idx::AbstractVector{<:Integer};
    standardize::Bool=true)
    z = readdlm(filename, ',', Float64)

    if eltype(exclude_std_idx) == Bool
        std_idx = .!exclude_std_idx
    else
        std_idx = trues(size(z, 2))
        std_idx[exclude_std_idx] .= false
    end

    if all(x == 1 for x in @view(z[:, 1]))
        std_idx[1] = false # don't standardize intercept
    else
        @warn("Covariate file provided but did not detect an intercept. An intercept will NOT be included in IHT!")
    end

    standardize && standardize!(@view(z[:, std_idx]))
    return z
end

function phenotype_is_missing(s::AbstractString)
    return s == "-9" || s == "NA"
end

"""
    cross_validate(plinkfile, d, path=1:20, phenotypes=6, covariates="", 
        cv_summaryfile="cviht.summary.txt", q=5, kwargs...)

Runs cross-validation to determinal optimal sparsity level `k`. Different
sparsity levels are specified in `path`. 

# Arguments
- `plinkfile`: A `String` for input PLINK file name (without `.bim/.bed/.fam` suffixes)
- `d`: Distribution of phenotypes. Specify `Normal` for quantitative traits,
    `Bernoulli` for binary traits, `Poisson` or `NegativeBinomial` for
    count traits, and `MvNormal` for multiple quantitative traits. 

# Optional Arguments
- `path`: Different values of `k` that should be tested. One can input a vector of 
    `Int` (e.g. `path=[5, 10, 15, 20]`) or a range (default `path=1:20`).
- `phenotypes`: Phenotype file name (`String`), an integer, or vector of integer. Integer(s)
    coresponds to the column(s) of `.fam` file that stores phenotypes (default 6). 
    We recognize missing phenotypes as `NA` or `-9`. For quantitative traits
    (univariate or multivariate), missing phenotypes are imputed with the mean. Binary
    and count phenotypes cannot be imputed. Phenotype files are read using `readdlm` function
    in Julia base. We require each subject's phenotype to occupy a different row. The file
    should not include a header line. Each row should be listed in the same order as in
    the PLINK. 
- `covariates`: Covariate file name. Default `covariates=""` (in which case an intercept
    term will be automatically included). If `covariates` file specified, it will be 
    read using `readdlm` function in Julia base. We require the covariate file to be
    comma separated, and not include a header line. Each row should be listed in the
    same order as in the PLINK. The first column should be all 1s to indicate an
    intercept. All other columns not specified in `exclude_std_idx` will be standardized
    to mean 0 variance 1
- `cv_summaryfile`: Output file name for saving IHT's cross validation summary statistics.
    Default `cv_summaryfile="cviht.summary.txt"`.
- `q`: Number of cross validation folds. Larger means more accurate and more computationally
    intensive. Should be larger 2 and smaller than 10. Default `q=5`. 
- `dosage`: Currently only guaranteed to work for VCF files. If `true`, will read
    genotypes dosages (i.e. `X[i, j] ∈ [0, 2]` before standardizing)
- All optional arguments available in [`cv_iht`](@ref)
"""
function cross_validate(
    plinkfile::AbstractString,
    d::UnionAll;
    path::AbstractVector{<:Integer} = 1:20,
    phenotypes::Union{AbstractString, Int, AbstractVector{Int}} = 6,
    covariates::AbstractString = "",
    cv_summaryfile::AbstractString = "cviht.summary.txt",
    q::Int = 5,
    exclude_std_idx::AbstractVector{<:Integer} = Int[],
    dosage::Bool = false,
    kwargs...
    )
    start_time = time()

    # read genotypes
    X, X_sampleID, X_chr, X_pos, X_ids, X_ref, X_alt = parse_genotypes(plinkfile, dosage)
    if typeof(X) <: SnpData
        x = SnpLinAlg{Float64}(X.snparray, model=ADDITIVE_MODEL, 
            center=true, scale=true, impute=true)
    else
        x = X # numeric matrix from VCF or BGEN files
    end

    # read phenotypes
    y = parse_phenotypes(X, phenotypes, d())

    # read and standardize covariates
    z = covariates == "" ? ones(size(x, 1)) : 
        parse_covariates(covariates, exclude_std_idx, standardize=true)
    is_multivariate(y) && (z = convert(Matrix{Float64}, Transpose(z)))

    # run cross validation
    if is_multivariate(y)
        mse = cv_iht(y, Transpose(x), z, path=path, q=q; kwargs...)
    else
        l = d == NegativeBinomial ? LogLink() : canonicallink(d()) # link function
        mse = cv_iht(y, x, z, path=path, q=q, d=d(), l=l; kwargs...)
    end

    # save results
    open(cv_summaryfile, "w") do io
        k = path[argmin(mse)]
        print_cv_results(io, mse, path, k)
        end_time = time() - start_time
        println(io, "Total cross validation time = $end_time seconds")
    end

    return mse
end

"""
    convert_gt(t::Type{T}, b::Bgen, trans::Bool)

Imports BGEN genotypes and chr/sampleID/pos/snpID/ref/alt into numeric arrays.
Genotypes are centered and scaled to mean 0 variance 1. Missing genotypes will
be replaced with the mean. Assumes every variant is biallelic (ie only 1 alt allele). 

# Input
- `b`: a `Bgen` object
- `T`: Type for genotype array
- `trans`: a `Bool`. If `trans==true`, output `G` will be `p × n`, otherwise `G` will be `n × p`.

# Output
- `G`: matrix of genotypes with type `T`. 
"""
function convert_gt(t::Type{T}, b::Bgen) where T <: Real
    n = n_samples(b)
    p = n_variants(b)

    # return arrays
    G = Matrix{t}(undef, n, p)
    Gchr = Vector{String}(undef, p)
    Gpos = Vector{Int}(undef, p)
    GsnpID = [String[] for _ in 1:p] # each variant can have >1 rsid, although we don't presently allow this
    Gref = Vector{String}(undef, p)
    Galt = [String[] for _ in 1:p] # each variant can have >1 alt allele, although we don't presently allow this

    # import each variant
    i = 1
    for v in iterator(b; from_bgen_start=true)
        dose = ref_allele_dosage!(b, v; T=t) # this reads REF allele as 1
        BGEN.alt_dosage!(dose, v.genotypes.preamble) # switch 2 and 0 (ie treat ALT as 1)
        copyto!(@view(G[:, i]), dose)
        # store chr/pos/snpID/ref/alt info
        Gchr[i], Gpos[i] = chrom(v), pos(v)
        push!(GsnpID[i], rsid(v))
        ref_alt_alleles = alleles(v)
        length(ref_alt_alleles) > 2 && error("Marker $i of BGEN is not biallelic!")
        Gref[i] = ref_alt_alleles[1]
        push!(Galt[i], ref_alt_alleles[2])
        i += 1
        clear!(v)
    end

    # center/scale/impute
    standardize_genotypes!(G)

    return G, b.samples, Gchr, Gpos, GsnpID, Gref, Galt
end

"""
    standardize_genotypes(G::AbstractMatrix)

Centers and scales each column (SNP) of `G` to mean 0 variance 1. Also each 
missing entry will be imputed as mean. 
"""
function standardize_genotypes!(G::AbstractMatrix)
    T = eltype(G)
    @inbounds for snp in eachcol(G)
        μi, mi = zero(T), 0
        @simd for i in eachindex(snp)
            μi += isnan(snp[i]) ? zero(t) : snp[i]
            mi += isnan(snp[i]) ? 0 : 1
        end
        μi /= mi
        σi = sqrt(μi * (1 - μi / 2))
        @simd for i in eachindex(snp)
            isnan(snp[i]) && (snp[i] = μi) # impute
            snp[i] -= μi # center
            σi > 0 && (snp[i] /= σi) # scale
        end
    end
    return nothing
end

"""
    parse_genotypes(tgtfile::AbstractString, dosage=false)

Imports genotype data from `tgtfile`. If binary PLINK files are supplied, genotypes
will not be decompressed to numeric matrices (~2 bit per entry). VCF or BGEN genotypes
will be stored in single precision matrices (32 bit per entry). 

# Inputs 
- `tgtfile`: VCF, binary PLINK, or BGEN file. VCF files should end in `.vcf` or
    `.vcf.gz`. Binary PLINK files should exclude `.bim/.bed/.fam` trailings but
    the trio must all be present in the same directory. BGEN files should end in
    `.bgen`

# Optional Inputs
- `dosage`: Currently only guaranteed to work for VCF files. If `true`, will read
    genotypes dosages (i.e. `X[i, j] ∈ [0, 2]` before standardizing)

# Output
- `X`: a `n × p` genotype matrix of type `Float32` (VCF or BGEN inputs) or `SnpData`
    (binary PLINK inputs)
- `Gchr`: Vector of `String`s holding chromosome number for each variant
- `Gpos`: Vector of `Int` holding each variant's position
- `GsnpID`: Vector of `String`s holding variant ID for each variant
- `Gref`: Vector of `String`s holding reference allele for each variant
- `Galt`: Vector of `String`s holding alterante allele for each variant
"""
function parse_genotypes(tgtfile::AbstractString, dosage=false)
    if (endswith(tgtfile, ".vcf") || endswith(tgtfile, ".vcf.gz"))
        f = dosage ? VCFTools.convert_ds : VCFTools.convert_gt
        X, X_sampleID, X_chr, X_pos, X_ids, X_ref, X_alt = 
            f(Float32, tgtfile, trans=false, 
            save_snp_info=true, msg = "Importing from VCF file...")
        # convert missing to NaN
        replace!(X, missing => NaN32)
        X = convert(Matrix{Float32}, X) # drop Missing from Matrix type
        # center/scale/impute
        standardize_genotypes!(X)
    elseif endswith(tgtfile, ".bgen")
        # dosage && error("Currently don't support reading dosage data from BGEN files")
        samplefile = isfile(tgtfile[1:end-5] * ".sample") ? 
            tgtfile[1:end-5] * ".sample" : nothing
        indexfile = isfile(tgtfile * ".bgi") ? tgtfile * ".bgi" : nothing
        bgen = Bgen(tgtfile; sample_path=samplefile, idx_path=indexfile)
        X, X_sampleID, X_chr, X_pos, X_ids, X_ref, X_alt = MendelIHT.convert_gt(Float32, bgen)
    elseif isplink(tgtfile)
        dosage && error("PLINK files detected but dosage = true!")
        X = SnpArrays.SnpData(tgtfile)
        # get other relevant information
        X_sampleID = X.person_info[!, :iid]
        X_chr = X.snp_info[!, :chromosome]
        X_pos = X.snp_info[!, :position]
        X_ids = X.snp_info[!, :snpid]
        X_ref = X.snp_info[!, :allele1]
        X_alt = X.snp_info[!, :allele2]
    else
        error("Unrecognized target file format: target file can only be VCF" *
            " files (ends in .vcf or .vcf.gz), BGEN (ends in .bgen) or PLINK" *
            " (do not include.bim/bed/fam) and all trio must exist in 1 directory)")
    end
    return X, X_sampleID, X_chr, X_pos, X_ids, X_ref, X_alt
end

isplink(tgtfile::AbstractString) = isfile(tgtfile * ".bed") && 
                                   isfile(tgtfile * ".fam") && 
                                   isfile(tgtfile * ".bim")
