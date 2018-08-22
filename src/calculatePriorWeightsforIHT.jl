"""
Calculates the Prior Weighting for IHT.
Returns a weight array (my_snpweights) (1,10000)
    and a MAF array (my_snpMAF ) (1,10000).

This function updates: hopefully nothing
"""
function calculatePriorWeightsforIHT(
#    xxx        :: SnpData,
    x        :: SnpLike{2},
    y        :: Vector{Float64},
    k        :: Int,
    v        :: IHTVariable,
    keyword  :: Dict{AbstractString, Any}
)
    # get my_snpMAF from x
    ALLELE_MAX = 2 * size(x,1)
    maf, minor_allele, missings_per_snp, missings_per_person = summarize(x)
    people, snps = size(x)
    my_snpMAF = maf' # crashes line 308 npzwrite
    my_snpMAF = convert(Matrix{Float64},my_snpMAF)

    # GORDON - CALCUATE CONSTANT WEIGHTS - another weighting option
    my_snpweights_const = copy(my_snpMAF) # only to allocate my_snpweights_const
    # need to test for bad user input !!!
    for i = 1:size(my_snpweights_const,2)
        my_snpweights_const[1,i] = keyword["pw_algorithm_value"]
    end

    # GORDON - CALCULATE WEIGHTS BASED ON p=MAF, 1/(2√pq) SUGGESTED BY BEN AND HUA ZHOU
    my_snpweights_p = my_snpMAF      # p_hat
    my_snpweights = 2 * sqrt(my_snpweights_p .* (1 - my_snpweights_p))   # just verifying 2 * sqrtm(p .* q) == 1.0 OK!
    my_snpweights_huazhou = my_snpweights
    my_snpweights = my_snpweights .\ 1      # this works! to get reciprocal of each element
    my_snpweights_huazhou_reciprocal = my_snpweights

    # DECIDE NOW WHICH WEIGHTS TO APPLY !!!
    if true # to ensure an algorithm, do this regardless
        my_snpweights = copy(my_snpweights_const)    # Ben/Kevin this is currently at 1.0 for testing null effect
    end
    if keyword["pw_algorithm"] == "hua.zhou.maf"
        my_snpweights = copy(my_snpweights_huazhou_reciprocal)
    end

#=
    println("===============================================================")
    println("========== REMOVE OUTLIERS HERE ===============================")
    println("========== AND TRIM DATA POINTS ===============================")
    println("===============================================================")
=#
    NULLweight = 1 # was 0 or 1 (Ben says don't weight to zero, one is good enough)
    NULLweight = keyword["null_weight"]
    cutoff = 0.025
    cutoff = 0.01
    cutoff = 0.001
    cutoff = keyword["cut_off"]
    #  NOTE: EXTREME VALUES INTERFERE WITH CONVERGENCE ** ONLY SOMETIMES **, SO PUT THEM BACK IN IF YOU WANT
    found = find(my_snpMAF .< cutoff)    # below 2.5% causes me problems with convergence
    my_snpweights[found] = NULLweight

    # trim the top too ???
    TRIM_TOP = 0.1
    TRIM_TOP = 0.0
    TRIM_TOP = keyword["trim_top"]
    if TRIM_TOP > 0.0
        cutoff_top = TRIM_TOP
        found_top = find(my_snpMAF .> cutoff_top)
        my_snpweights[found_top] = 0.0
    end

#=
    println("===============================================================")
    println("===============================================================")
    println("====== APPLYING THE APRIORI WEIGHTS AS IF BY APRIORI ==========")
    println("===============================================================")
=#
    # APRIORI
    # DEVELOPERS NOTE: TO REALLY SEE THE EFFECT OF Weighting
    #   TURN APRIORI ON AND SET WEIGHT ALGO TO CONSTANT = 1.0
    pw_tmp = "none"    # test for none at end of this section
    if keyword["pw_pathway1"] > ""
        pw_tmp = eval(parse(keyword["pw_pathway1"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway1_constantvalue"]
            my_snpweights[m2] = m1
        end
    end
    if keyword["pw_pathway2"] > ""
        pw_tmp = eval(parse(keyword["pw_pathway2"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway2_constantvalue"]
            my_snpweights[m2] = m1
        end
    end
    if keyword["pw_pathway3"] > ""
        pw_tmp = eval(parse(keyword["pw_pathway3"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway3_constantvalue"]
            my_snpweights[m2] = m1
        end
    end

    println("===============================================================")
    println("============ MAKE A TABLE OF THE WEIGHT FUNCTION VARIABLES ====")
    println("===============================================================")
    println()
    snpmatrix = convert(Array{Float64,2}, x)
    # compute some summary statistics for our snpmatrix
    maf, minor_allele, missings_per_snp, missings_per_person = summarize(x)
    people, snps = size(x)

    #precompute mean and standard deviations for each snp. Note that (1) the mean is
    #given by 2 * maf, and (2) based on which allele is the minor allele, might need to do
    #2.0 - the maf for the mean vector.
    mean_vec = zeros(snps)
    for i in 1:snps
        minor_allele[i] ? mean_vec[i] = 2.0 - 2.0maf[i] : mean_vec[i] = 2.0maf[i]
    end
    std_vec = std_reciprocal(x, mean_vec)
    println("SNP\t0\t1\t2\tmean\tsd\tcount\tMAF\tp\tq\t2√(pq)\tweight\tcountmap")
    for i = 1:10

        (μ_snp, σ_snp) = mean_and_std(snpmatrix[:,i])
        @printf("snp[%1d]:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t", i%10,
            (0 - μ_snp)/σ_snp, (1 - μ_snp)/σ_snp, (2 - μ_snp)/σ_snp, μ_snp, σ_snp)

        @printf("%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", sum(snpmatrix[:,i]), my_snpMAF[1,i],
                                                    my_snpweights_p[i], my_snpweights_q[i],
                                                    (2 * sqrt(my_snpweights_p .* (1 - my_snpweights_p)))[i],
                                                    my_snpweights[i])
        #print(countmap(snpmatrix[:,i]))
        print()
    end
    return my_snpMAF, my_snpweights
end
