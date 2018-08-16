
"""
Prints a convergence report
    and makes some charts
    from the results returned by L0_reg

Returns nothing

This function updates: hopefully nothing???
"""
function printConvergenceReport(
    result          :: gIHTResults,
    my_snpMAF       :: Array{Float64,2},
    my_snpweights   :: Array{Float64,2},
    snpmatrix       :: SnpLike{2},                  # x in L0_reg
    y               :: Vector{Float64},             # phenotype in L0_reg
    v               :: IHTVariable,
    snp_definition_frame )
    # locals
    mm_time = result.time
    mm_loss = result.loss
    mm_iter = result.iter
    mm_beta = result.beta       # should be equal to v.b

    println("===============================================================")
    println("============= INFORMATION ABOUT SELECTED BETAS ================")
    println("============= IN L0_reg =======================================")
    println("===============================================================")

    println("IHT converged in " * string(mm_iter) * " iterations")
    println("It took " * string(mm_time) * " seconds to converge")
    println("The estimated model is stored in 'v.b'")
    println("typeof(mm_beta) = $(typeof(mm_beta))")
    println("typeof(v.b) = $(typeof(v.b))")
    println("There are " * string(countnz(v.b)) * " non-zero entries of β")
    found = find(v.b .!= 0.0)
    print("found: ")
    println(found)
    println("v.b, the betas(coefficients), are the result of the IHT")
    print("v.b[found]: ")
    println(v.b[found])
    println("v.b[1:10] $(size(v.b)) = $(v.b[1:10])")
    #describe(mm_beta) # same as v.b
    describe(v.b)

    println(BLUE*"THE SUMMARY STATS FOR THE Betas ARE PLOTTED IN bar_b.png"*DEFAULT)
    StatPlots.bar(v.b)
    Plots.savefig("ZZZ_bar_b1.png")

    println()
    #USE_INTERCEPT = true
    if true # && false # currently the intercept is at the end
        #my_snpweights_intercept = [ones(size(my_snpweights, 1)) my_snpweights]
        my_snpweights_intercept = my_snpweights
        println("my_snpweights_intercept for found:")
        println(size(my_snpweights_intercept))
        println(my_snpweights_intercept[1,found])

        println()
        my_snpMAF_intercept = [my_snpMAF ones(size(my_snpMAF, 1))]
        println("my_snpMAF_intercept for found:")
        println(size(my_snpMAF_intercept))
        println(my_snpMAF_intercept[1,found])
    else
        println("my_snpweights for found:")
        println(size(my_snpweights))
        println(my_snpweights[1,found])

        println()
        println("my_snpMAF for found:")
        println(size(my_snpMAF))
        println(my_snpMAF[1,found])
    end
    println()
    println("y, the phenotype for 2200 people, is given in the file 'gwas 1 data.fam'")
    print("y: ")
    print(size(y))
    println(y[1:10])
    #println(y[found])
    describe(y)
    println(BLUE*"THE Phenotype for 2200 people are plotted IN bar_phenotype.png"*DEFAULT)
    StatPlots.bar(y)
    Plots.savefig("ZZZ_bar_phenotype.png")


    println()
    println("snpmatrix holds allele counts, zscored, and weighted for 2200 people")
    println("here is a sample of the first 10 SNPs for the first 3 people")
    println("note: first SNP is likely 1 for the intercept")
    println("snpmatrix[1,1:10] = $(snpmatrix[1,1:10])")
    println("snpmatrix[2,1:10] = $(snpmatrix[2,1:10])")
    println("snpmatrix[3,1:10] = $(snpmatrix[3,1:10])")

    println()
    println("v.xb is x*b for 2200 people, where x is weighted_snpmatrix and b are the coefficients beta")
    print("v.xb: ")
    print(size(v.xb))
    println("v.xb[1:10] = $(v.xb[1:10])")
    println("type(v.xb) = $(typeof(v.xb))")
    describe(v.xb)

    #=
    println()
    print("(y .- v.xb): ")      # same as v.r
    println(size(y .- v.xb))
    describe((y .- v.xb))
    #println((y .- v.xb)[found])
    =#
    println()
    println("v.r are the residuals for 2200 data points (people)")
    println("we are trying to minimize a function of this number")
    println("v.r = y .- v.xb")
    println("the function is 'next_loss = sum(abs2, v.r) / 2'")
    print("v.r: ")
    println(size(v.r))
    describe(v.r)
    #println(v.r[found])
    println(BLUE*"THE SUMMARY STATS FOR THE Residuals ARE PLOTTED IN bar_r1.png"*DEFAULT)
    StatPlots.bar(v.r)
    Plots.savefig("ZZZ_bar_r1.png")

    println("THE REGRESSION RESULTS ARE SAVED IN bar_b.png")
    StatPlots.bar(found,v.b[found])
    Plots.savefig("ZZZ_bar_b2.png")

    println("===============================================================")
    println("============= MORE INFORMATION ABOUT SELECTED BETAS ===========")
    println("============= IN MENDELIHT ====================================")
    println("===============================================================")
    println("time returned is $(mm_time)")
    found = find(mm_beta .!= 0.0)
    print("found: ")
    println(found)
    println("mm_beta, the betas(coefficients), are the result of the IHT")
    print("mm_beta[found]: ")
    println(mm_beta[found])
    println("mm_beta[1:10] $(size(mm_beta)) = $(mm_beta[1:10])")
    describe(mm_beta)
    snp = 1
    println("SNP[$(snp)] SNP        = $(snp_definition_frame[snp,:SNP])")
    println("SNP[$(snp)] Chromosome = $(snp_definition_frame[snp,:Chromosome])")
    println("SNP[$(snp)] BasePairs  = $(snp_definition_frame[snp,:Basepairs])")
    println("SNP[$(snp)] Allele1    = $(snp_definition_frame[snp,:Allele1])")
    println("SNP[$(snp)] Allele2    = $(snp_definition_frame[snp,:Allele2])")
    println()
if true        # NO NEED FOR SNP NAMES SINCE THEY ARE FAKE? or are they BMI related
    #run(`powershell pwd`)   # works !!!
    rs = Nullable{Int64}[]
    for snp in found
#        if USE_INTERCEPT && false # intercept is at the end now
#            if snp_index == 1
#                continue
#            end
#            snp = snp_index - 1 # adjust for intercept
#        else
            if snp == 10001
                continue
            end
            #snp = snp_index
#        end

        #aaaa=readstring(`bash -c 'curl -X GET --header "Accept: application/json" "https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/268"'`);
        #println("back from readstring()")
        #run(`bash -c curl -X GET --header "Accept: application/json" "https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/$(snp)"` |> "$(snp).txt")
        r = tryparse(Int64, snp_definition_frame[snp,:SNP][3:end])
        push!(rs, r)
        println("SNP[$(snp)] = $(snp_definition_frame[snp,:SNP])")
        println("SNP[$(snp)] = $(snp_definition_frame[snp,:Chromosome])")
        println("SNP[$(snp)] = $(snp_definition_frame[snp,:Basepairs])")
        println("SNP[$(snp)] = $(snp_definition_frame[snp,:Allele1])")
        println("SNP[$(snp)] = $(snp_definition_frame[snp,:Allele2])")
        println()
    end
    println("SNP names to pass to NCBI:")
    println(rs)
end

    return nothing
end
