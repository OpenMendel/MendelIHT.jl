using SnpArrays

# 
# filter NFBC data for 98% genotype success rate and SNPs with missing data < 2%
#
xdata = SnpData("NFBC_dbGaP_20091127")
rmask, cmask = SnpArrays.filter(xdata.snparray)

#
# save only chr1
#
chr1_snps = count(x -> x == "1", xdata.snp_info[!, 1])
cmask[chr1_snps + 1:end] .= 0
@show sum(cmask)
SnpArrays.filter("NFBC_dbGaP_20091127", rmask, cmask; des="NFBC.chr1.qc")

#
# Use PLINK to filter for SNPs in LD and compute GRM with GEMMA
# These actions can be done with SnpArrays.jl, but we use PLINK and GEMMA
# because we want to show the analysis is unbiased if we use external softwares
# 
# r2 = 0.25 gives 7594 SNPs
# r2 = 0.5 gives 13441 SNPs
# r2 = 0.75 gives 18580 SNPs
#
plink_exe = "/scratch/users/bbchu/NFBC_sim/plink"
plink_file = "/scratch/users/bbchu/NFBC_sim/data/NFBC.qc.imputeBy0.chr.1"
gemma_exe = "/scratch/users/bbchu/NFBC_sim/gemma"
for ld in [0.25, 0.5, 0.75]
    # filter for SNPs in LD
    run(`$plink_exe --bfile $plink_file --indep-pairwise 10000000 24523 $ld`)
    run(`$plink_exe --bfile $plink_file --extract plink.prune.in --make-bed --out NFBC.qc.imputeBy0.chr.1.LD$ld`)
    # compute GRM with gemma
    run(`$gemma_exe -bfile NFBC.qc.imputeBy0.chr.1.LD$ld -gk 1 -o NFBC.qc.imputeBy0.chr.1.LD$ld`)
end
