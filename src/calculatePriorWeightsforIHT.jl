#for Gordon
#using PyPlot
#using Plots
using StatPlots
#using BenchmarkTools
#using Distributions
using JLD
using NPZ
#export MendelIHT
#using RDatasets
#using Gadfly

if !isdefined(:RED)
    const RED     = "\e[1;31m" # BOLD
    const GREEN   = "\x1b[32m"
    const YELLOW  = "\x1b[33m"
    const BLUE    = "\e[1;34m" # BOLD
    const PURPLE  = "\x1b[35m"
    const BOLD    = "\x1b[1m"
    const DEFAULT = "\x1b[0m"
    const YELLOW_BG = "\e[43m"
    const CYAN_BG = "\e[46m"
    #= Background
    Value	Color
    \e[40m	Black
    \e[41m	Red
    \e[42m	Green
    \e[43m	Yellow
    \e[44m	Blue
    \e[45m	Purple
    \e[46m	Cyan
    \e[47m	White
    =#
end
#USE_INTERCEPT = true
#USE_INTERCEPT = false    # has not been used/tested for quite some time

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
    #convert bitarrays to Float64 genotype matrix, standardize each SNP, and add intercept
#    snpmatrix = convert(Array{Float64,2}, xxx.snpmatrix)
    println("===============================================================")
    println("============= BEGIN CODE FOR PRIOR WEIGHTS FUNCTION ===========")
    println("===============================================================")
    #myplot = Gadfly.plot(dataset("HistData", "ChestSizes"), x="Chest", y="Count", Geom.bar)
    #myplot = Gadfly.plot(x = 1:10, y = 2:11, Geom.line)
    #draw(PNG("first_gadfly_bar_chart.png", 3inch, 3inch), myplot)

    println(BLUE*"Gordon 124: new code for Prior Weights work"*DEFAULT)
    println("Note: numbers after my name should be close to line numbers in original code.")
    #USE_INTERCEPT = true  # its global now
    #println("Note: Set USE_INTERCEPT = false to drop intercept column.")
    #println("Note:      must adjust intercept in MendelIHT_utilities.jl ALSO!")
#=
    # this was working right here
    xxx = SnpArray("gwas 1 data")
    xxx = convert(Array{Float64,2}, xxx)
    println("Gordon 124: contents of snpmatrix from SnpArray()")
    println(xxx[1,1:10])
=#
#=
# get snpdata for Gordon, this was working in MendelIHT.jl
    (pedigree, person, nuclear_family, locus, snpdata,
    locus_frame, phenotype_frame, pedigree_frame, snp_definition_frame) =
    read_external_data_files(keyword)
    snpmatrix = convert(Array{Float64,2}, snpdata.snpmatrix)
    println(snpmatrix[1,1:10])
=#

    print(YELLOW_BG)
    println("===============================================================")
    println("========== PRIOR WEIGHTS - BEGIN CALCULATIONS =================")
    println("===============================================================")
    print(DEFAULT)

#=
    # Create my own snpmatrix for Prior Weights work
    my_snpmatrix = zeros(snpmatrix)
    copy!(my_snpmatrix, snpmatrix)
    my_snpmatrix = deepcopy(snpmatrix)  # ask Ben about copying matricies
    println(my_snpmatrix[1,1:10])       # all 0,1,2 as Ben said
=#
    # ALLELE_MAX is times 2 * SNPcount because each PERSON's SNP has 2 alleles
    ALLELE_MAX = 2 * size(x,1)
    println(size(x))
    println("ALLELE_MAX = $(ALLELE_MAX)")
    # MAF = Minor Allele Frequency
    # my_snpMAF is the sum the allele counts 0,1, or 2 for each SNP (column)
    #   in the snpmatrix / ALLELE_MAX
    # Note: Since it is a MINOR allele the max should be 0.500000, OK!
#=
    # Minor Allele Freq (MAF) = sum of alleles / (2 * SNPcount)
    myold_snpMAF = sum(my_snpmatrix, 1) ./ ALLELE_MAX
    my_snpMAF = myold_snpMAF
=#

    # compute some summary statistics for our snpmatrix
    maf, minor_allele, missings_per_snp, missings_per_person = summarize(x)
    people, snps = size(x)
    println("minor_allele")
#    println("size(myold_snpMAF) = $(size(myold_snpMAF))")
    println("size(maf) = $(size(maf))")
    #myold_snpMAF[1] = maf  # doesn't work
    #my_snpMAF = maf
    println()
    println("typeof(maf) = $(typeof(maf))")
    println("size(maf) = $(size(maf))")
#    println("size(myold_snpMAF) = $(size(myold_snpMAF))")
    println("size(maf) = $(size(maf))")
    println("size(transpose(maf)) = $(size(transpose(maf)))")
    println("size(maf') = $(size(maf'))")
#    println("typeof(my_snpMAF) = $(typeof(my_snpMAF))")
    my_snpMAF = maf' # crashes line 308 npzwrite
    my_snpMAF = convert(Matrix{Float64},my_snpMAF)
#    my_snpMAF = my_snpMAF[:,1:end-1]
    println("typeof(my_snpMAF) = $(typeof(my_snpMAF))")

    print("Gordon - size(my_snpMAF) = ")
    println(size(my_snpMAF))
    println("Gordon - my_snpMAF = $(my_snpMAF[1,1:10])")
    println("Gordon -       maf = $(maf[1:10])")
    println("Gordon - max(my_snpMAF) = $(maximum(my_snpMAF))")
    println("my_snpMAF[1,7022] = $(my_snpMAF[1,7022])")
    println("my_snpMAF[1,7023] = $(my_snpMAF[1,7023])")
    println("my_snpMAF[1,7024] = $(my_snpMAF[1,7024])")
    describe(my_snpMAF[1,:])


    # GORDON - CALCUATE CONSTANT WEIGHTS - another weighting option
    my_snpweights_const = copy(my_snpMAF) # only to allocate my_snpweights_const
    # need to test for bad user input !!!
    for i = 1:size(my_snpweights_const,2)
        my_snpweights_const[1,i] = keyword["pw_algorithm_value"]
    end
    println(my_snpweights_const[1,1:10])
    println("\ndescribe(my_snpweights_const)")
    describe(my_snpweights_const[1,:])

    # GORDON - CALCULATE WEIGHTS BASED ON p=MAF, 1/(2√pq) SUGGESTED BY BEN AND HUA ZHOU
    my_snpweights_p = my_snpMAF      # p_hat
    my_snpweights_q = 1 - my_snpweights_p   # q_hat
    my_snpweights = my_snpweights_p + my_snpweights_q   # just verifying p + q == 1 OK!
    my_snpweights_pq = my_snpweights_p .* my_snpweights_q   # just verifying P .* q max is 0.25 OK!
    # next line is 1/pq
    my_snpweights = my_snpweights .\ 1      # not what we want, but this works! to get invervse of each element
    my_snpweights = sqrt(my_snpweights_p .* my_snpweights_q)   # just verifying sqrtm(p .* q) == 0.5 OK!
    # next line is 2 * sqrt(p(1-p)) from Hua Zhou 2011 paper pg. 110
    my_snpweights = 2 * sqrt(my_snpweights_p .* (1 - my_snpweights_p))   # just verifying 2 * sqrtm(p .* q) == 1.0 OK!
    my_snpweights_huazhou = my_snpweights
    println("\ndescribe(my_snpweights_p)")
    describe(my_snpweights_p[1,:])

    println("\ndescribe(my_snpweights_q)")
    describe(my_snpweights_q[1,:])
    #println("describe(1 - my_snpweights_p)")   # SAME AS Q
    #describe(1 .- my_snpweights_p[1,:])
    println("\ndescribe(my_snpweights 2√pq)")
    describe(my_snpweights[1,:])
    my_snpweights = my_snpweights .\ 1      # this works! to get reciprocal of each element
    my_snpweights_huazhou_reciprocal = my_snpweights
    println("\ndescribe(my_snpweights 2√pq .\\ 1)")
    describe(my_snpweights[1,:])

    print(CYAN_BG)
    println("===============================================================")
    println("=========== SELECT WEIGHT FUNCTION HERE =======================")
    println("===============================================================")
    print(DEFAULT)
    # DECIDE NOW WHICH WEIGHTS TO APPLY !!!

    if true # to ensure an algorithm, do this regardless
        my_snpweights = copy(my_snpweights_const)    # Ben/Kevin this is currently at 1.0 for testing null effect
        println("Selected my_snpweights_const")
    end
    if keyword["pw_algorithm"] == "hua.zhou.maf"
        my_snpweights = copy(my_snpweights_huazhou_reciprocal)
        println("Selected my_snpweights_huazhou_reciprocal")
    end
    #my_snpweights = copy(my_snpMAF)
    println(RED*"SELECT WEIGHT FUNCTION HERE !!!"*DEFAULT)
    println("\ndescribe(my_snpweights)")
    describe(my_snpweights[1,:])

    println("===============================================================")
    println("========== REMOVE OUTLIERS HERE ===============================")
    println("========== AND TRIM DATA POINTS ===============================")
    println("===============================================================")
    #   DO I REALLY WANT TO KILL OUTLIERS HERE - I LIKE THEM, MUST BE A BETTER WAY TO KEEP THEM
    NULLweight = 1 # was 0 or 1 (Ben says don't weight to zero, one is good enough)
    NULLweight = keyword["null_weight"]
    cutoff = 0.025
    cutoff = 0.01
    cutoff = 0.001
    cutoff = keyword["cut_off"]
    #cutoff = 0.025
    #  NOTE: EXTREME VALUES INTERFERE WITH CONVERGENCE ** ONLY SOMETIMES **, SO PUT THEM BACK IN IF YOU WANT
    found = find(my_snpMAF .< cutoff)    # below 2.5% causes me problems with convergence
    println(found)
    println(my_snpMAF[found])
    println(RED*"Setting weight = $(NULLweight) for $(size(found)) outliers with MAF below $(cutoff) cutoff."*DEFAULT)
    my_snpweights[found] = NULLweight
    println(my_snpweights[found])

    # List the RARE KEEPERS
    found = find(my_snpMAF .< 0.05)    # below 2.5% causes me problems with convergence
    println("Here are the rest of the SNPs with MAF < .05 ($(size(found)) RARE KEEPERS)")
    println(found)
    println(my_snpMAF[found])
    println(RED*"NOT Setting weight = $(NULLweight) for $(size(found)) RARE KEEPERS with MAF below $(0.05) cutoff."*DEFAULT)
    if my_snpweights[found][1] != 1.0
        println(my_snpweights[found])
    end

    # trim the top too ???
    TRIM_TOP = 0.1
    TRIM_TOP = 0.0
    TRIM_TOP = keyword["trim_top"]
    if TRIM_TOP > 0.0
        cutoff_top = TRIM_TOP
        found_top = find(my_snpMAF .> cutoff_top)    # below 2.5% causes me problems with convergence
        #println(found_top)
        #println(my_snpMAF[found_top])
        println(RED*"Setting weight = 0 for $(size(found_top)) data points with MAF above $(cutoff_top) cutoff."*DEFAULT)
        my_snpweights[found_top] = 0.0
        #println(my_snpweights[found_top])
    end

    println("===============================================================")
    println("===============================================================")
    println("====== APPLYING THE APRIORI WEIGHTS AS IF BY APRIORI ==========")
    println("===============================================================")
    println(RED*"HERE I ASSIGN THE APRIORI WEIGHTS AS IF BY APRIORI !!!"*DEFAULT)
    # APRIORI
    # DEVELOPERS NOTE: TO REALLY SEE THE EFFECT OF Weighting
    #   TURN APRIORI ON AND SET WEIGHT ALGO TO CONSTANT = 1.0
    # [899, 1881, 3775, 3982, 4210, 4399, 5794, 6612, 6628, 7024] # 10 here, 12 below
    pw_tmp = "none"    # test for none at end of this section
    if keyword["pw_pathway1"] > ""
        #for m2 in [899, 1881, 3775, 3982, 4210, 4399, 5794, 6612, 6628, 7024, 8960, 9468]
        #for m2 in [898, 1880, 2111, 3774, 3981, 4398, 5793, 6611, 6627, 7023] # [898, 1880, 2111, 3774, 3981, 4398, 6611, 6627, 7023]
        pw_tmp = eval(parse(keyword["pw_pathway1"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway1_constantvalue"]
            println("Apply adjustment for APRIORI weight at m2 = $(m2), m1 = $(m1)")
            print("Before = $(my_snpweights[m2])")
            my_snpweights[m2] = m1
            println(", after = $(my_snpweights[m2])")
        end
    end
    if keyword["pw_pathway2"] > ""
        pw_tmp = eval(parse(keyword["pw_pathway2"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway2_constantvalue"]
            println("Apply adjustment for APRIORI weight at m2 = $(m2), m1 = $(m1)")
            print("Before = $(my_snpweights[m2])")
            my_snpweights[m2] = m1
            println(", after = $(my_snpweights[m2])")
        end
    end
    if keyword["pw_pathway3"] > ""
        pw_tmp = eval(parse(keyword["pw_pathway3"]))
        for m2 in pw_tmp
            m1 = keyword["pw_pathway3_constantvalue"]
            println("Apply adjustment for APRIORI weight at m2 = $(m2), m1 = $(m1)")
            print("Before = $(my_snpweights[m2])")
            my_snpweights[m2] = m1
            println(", after = $(my_snpweights[m2])")
        end
    end
    if pw_tmp == "none"
        println("None")
    end

    println("===============================================================")
    println("============ MAKE GRAPHS OF SELECTED WEIGHT FUNCTION ==========")
    println("===============================================================")

    println(BLUE*"THE SUMMARY STATS FOR THE WEIGHTS ARE PLOTTED IN histogram_my_snpweights.png"*DEFAULT)
    StatPlots.histogram(my_snpweights[1,:])
    Plots.savefig("ZZZ_histogram_my_snpweights.png")


    println(BLUE*"THE SUMMARY STATS FOR THE WEIGHTS ARE PLOTTED IN histogram_my_snpweights_p.png"*DEFAULT)
    StatPlots.histogram(my_snpweights_p[1,:])
    Plots.savefig("ZZZ_histogram_my_snpweights_p.png")


    println(BLUE*"THE SUMMARY STATS FOR THE WEIGHTS ARE PLOTTED IN bar_my_snpweights.png"*DEFAULT)
    StatPlots.bar(my_snpweights[1,:])
    Plots.savefig("ZZZ_bar_my_snpweights.png")

    println(BLUE*"THE SUMMARY STATS FOR THE WEIGHTS ARE PLOTTED IN boxplot_my_snpweights.png"*DEFAULT)
    StatPlots.boxplot(my_snpweights[1,:])
    Plots.savefig("ZZZ_boxplot_my_snpweights.png")

    println()

    # =================================================================
    # =================================================================
    println(BLUE*"HERE ARE THE SUMMARY STATS FOR THE WEIGHTS"*DEFAULT)
    print("Gordon - size(my_snpweights) = ")    # (1, 10000)
    println(size(my_snpweights))
    print("Gordon - my_snpweights = ")
    println(my_snpweights[1,1:10])
    print("Gordon - max(my_snpweights) = ")
    println(maximum(my_snpweights))
    describe(my_snpweights[1,:])

    # save the variables to disk for external analysis
    # specifically intended for making inline plots in Jupyter
    npzwrite("ZZZ_my_snpweights.npz",my_snpweights)
    npzwrite("ZZZ_my_snpweights_p.npz",my_snpweights_p)
    npzwrite("ZZZ_my_snpweights_q.npz",my_snpweights_q)
    println("test3test")
    npzwrite("ZZZ_my_snpweights_pq.npz",my_snpweights_pq)
    npzwrite("ZZZ_my_snpweights_huazhou.npz",my_snpweights_huazhou)
    npzwrite("ZZZ_my_snpweights_huazhou_reciprocal.npz",my_snpweights_huazhou_reciprocal)


    save("ZZZ_my_snpweights.jld", "data", my_snpweights)
    #load("data.jld")["data"]


    println("===============================================================")
    println("============ MAKE A TABLE OF THE WEIGHT FUNCTION VARIABLES ====")
    println("===============================================================")
    println()
#    println("SNP\t0\t1\t2\tmean\tsd\tcount\tMAF\tp\tq\t2√(pq)\tweight\tcountmap")
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
    println()
    println("===============================================================")
    println("===============================================================")
    println("========== PRIOR WEIGHTS - END CALCULATIONS ===================")
    println("===============================================================")
    println("===============================================================")
    println("===============================================================")
    println(typeof(my_snpMAF))
    println(typeof(my_snpweights))
#    println(typeof(my_snpmatrix))
    println(size(my_snpMAF))
    println(size(my_snpweights))
#    println(size(my_snpmatrix))
    return my_snpMAF, my_snpweights
end
