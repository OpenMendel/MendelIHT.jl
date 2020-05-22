# print all significant SNPs from logistic GWAS into latex table format
using CSV
df = CSV.read("ukb.final.logistic.pval.txt")
function run()
    tot = 0
    for i in 1:size(df, 1)
        if df[i, 6] < 5e-8
            tot += 1
            # snpid, chrom, pos, p-value
            # println(df[i, 3], " & ", df[i, 1], " & ", df[i, 2], " & ", 
            #     round(df[i, 6], sigdigits=3), "\\\\")
        end
    end
    tot
end
