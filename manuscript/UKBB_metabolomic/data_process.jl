#
# These scripts were used to process the UKB data for the 18 metabolomic traits
# analysis featured in the multivariate IHT paper. This file was not intended
# to be distributed, hence is not written in the most streamlined/readable way.
# Nevertheless, we provide this script in the hope that it will be useful to others.
#

#
# After running the script, the final result are:
# Phenotype file (not standardized, with header and family-ID): phenotypes.reordered.csv 
# Final phenotype file (standardized, without header/family-ID): phenotypes.reordered.standardized.csv
# Final covariates file (with family ID, not standardized): covariates.reordered.csv
# Final covariates file (without family ID, standardized, extra intercept column): covariates.reordered.standardized.csv
# Final genotype file: ukb.merged.metabolic.subset.european.400K.QC.bed
#

using SnpArrays, CSV, DataFrames, DelimitedFiles, MendelIHT
global propca_exe = "/scratch/users/bbchu/ProPCA/build/propca"

# merge all chrom into 1 plink (note: this requires roughly 100GB of ram)
SnpArrays.merge_plink("ukb_chr", des="ukb.merged")

# keep only samples with phenotypes (we can match 121157 samples genotype to phenotypes)
merged = SnpData("ukb.merged")
y = CSV.read("phenotypes.csv", DataFrame)
phenotype_sampleID = y[:, 1]
genotype_sampleID = parse.(Int, merged.person_info[!, :fid])
keep_idx = indexin(phenotype_sampleID, genotype_sampleID)
filter!(x -> !isnothing(x), keep_idx)
keep_idx = sort!(convert(Vector{Int}, keep_idx))
rowidx = falses(1:size(merged.person_info, 1))
rowidx[keep_idx] .= true
SnpArrays.filter(merged, rowidx, 1:size(merged.snp_info, 1), 
    des="ukb.merged.metabolic.subset")

# Get sample ancestry
pop_code = String[]
sampleID = String[]
cnt = 1
for line in eachline("/scratch/users/bbchu/ukb_iht/ukb40055.txt")
    if cnt == 1
        cnt += 1
        continue
    end
    l = split(line, '\t')
    id, population = l[2], l[6151]
    if id == "" || population == ""
        cnt += 1
        continue
    end
    push!(sampleID, id)
    push!(pop_code, population)
    cnt += 1
end
writedlm("UKB_sample_population_code.txt", [sampleID pop_code])

# only keep samples of European descent
sample_ancestry = CSV.read("UKB_sample_population_code.txt", DataFrame, header=false)
europeans = [1001, 1002, 1003]
european_idx = findall(x -> x in europeans, sample_ancestry[!, 2]) # 472128 samples
subset_merged = SnpData("ukb.merged.metabolic.subset")
subset_fam_file = subset_merged.person_info
to_keep_sampleID = sample_ancestry[european_idx, 1] ∩ parse.(Int, subset_fam_file[!, 1]) # 114256 samples
to_keep_sample_idx = findall(x -> x in to_keep_sampleID, parse.(Int, subset_fam_file[!, 1])) # 114256 samples
SnpArrays.filter(subset_merged, to_keep_sample_idx, 1:size(subset_merged.snp_info, 1), 
    des="ukb.merged.metabolic.subset.european")

# filter for samples with 99% genotype success rate and only keep snps in blood pressure analysis
blood_pressure_bim = CSV.read("ukb.bloodpressure.filtered.bim", DataFrame, header=false)
euro_subset = SnpData("ukb.merged.metabolic.subset.european")
rmask, _ = SnpArrays.filter(euro_subset.snparray,
    min_success_rate_per_row = 0.99,  # 108352
    min_success_rate_per_col = 0.98)  # 580871
shared_snp_idx = indexin(euro_subset.snp_info[!, 2], blood_pressure_bim[!, 2])
cmask = trues(euro_subset.snps)
cmask[findall(isnothing, shared_snp_idx)] .= false # 470228
SnpArrays.filter(euro_subset, rmask, cmask,
    des="ukb.merged.metabolic.subset.european.400K")

# exclude 1st and 2nd relatives (i.e. GRM values > 0.125) 
# note this requires 400-500 GB of memory
x = SnpArray("ukb.merged.metabolic.subset.european.400K.bed")
Φ = grm(x, method=:Robust, minmaf=0.05, t=Float32)
idx = findall(x -> x > 0.125, Φ)
xidx = [idx[i][1] for i in 1:length(idx)];
yidx = [idx[i][2] for i in 1:length(idx)];
writedlm("related_idx", [xidx yidx])
idx = Int.(readdlm("related_idx"))
related_samples = Int[]
for i in 1:size(idx, 1)
    if idx[i, 1] != idx[i, 2] # delete both for simplicity
        push!(related_samples, idx[i, 1])
        push!(related_samples, idx[i, 2])
    end
end
unique!(related_samples) # 4088
xdata = SnpData("ukb.merged.metabolic.subset.european.400K")
n, p = size(xdata)
cmask = trues(p)
rmask = trues(n)
rmask[related_samples] .= false # 104264 samples left
SnpArrays.filter(xdata, rmask, cmask,
    des="ukb.merged.metabolic.subset.european.400K.QC")

# compute PCA (this was pretty fast, around 1 hour)
seed = 2022
plinkname = "ukb.merged.metabolic.subset.european.400K.QC"
outfile = "ukb.merged.metabolic.subset.european.400K.QC."
run(`$propca_exe -g $plinkname -k 10 -o $outfile -nt $(Threads.nthreads()) -seed $seed`)

# filter phenotypes and reorder them so they match genotypes
# after all filtering, there are 104264 samples remaining
xdata = SnpData("ukb.merged.metabolic.subset.european.400K.QC")
y = CSV.read("phenotypes.csv", DataFrame) #121733 samples
phenotype_sampleID = y[:, 1]
genotype_sampleID = parse.(Int, xdata.person_info[!, :fid])
reorder_idx = indexin(genotype_sampleID, phenotype_sampleID) |> Vector{Int} # 104264 samples
n = length(reorder_idx)
p = size(y, 2)
y_reordered = zeros(n, p)
for i in 1:n
    row = reorder_idx[i]
    for j in 1:p
        y_reordered[i, j] = y[row, j] 
    end
end
all(y_reordered[:, 1] .== genotype_sampleID) || error("bug in matching genotype ID to phenotype ID")
y_reordered_df = DataFrame(y_reordered, names(y))
y_reordered_df[!, 1] = y_reordered_df[!, 1] |> Vector{Int}
CSV.write("phenotypes.reordered.csv", y_reordered_df)

# log-transform and standardize phenotypes
y_df = CSV.read("phenotypes.reordered.csv", DataFrame)
y = Matrix(y_df[!, 2:end])
ylog = log.(y)
standardize!(ylog)
writedlm("phenotypes.reordered.standardized.csv", ylog, ',')

# Get sample age and BMI
sampleID, sample_age = String[], String[]
cnt = 1
for line in eachline("/scratch/users/bbchu/ukb_iht/ukb40055.txt")
    if cnt == 1
        cnt += 1
        continue
    end
    l = split(line, '\t')
    id, age = l[2], l[6162]
    if id == "" || age == ""
        cnt += 1
        continue
    end
    push!(sampleID, id)
    push!(sample_age, age)
    cnt += 1
end
writedlm("UKB_sample_age.txt", [sampleID sample_age])

# combine all non-genetic covariates (sex, bmi, age, age^2, PC1-PC10)
xdata = SnpData("ukb.merged.metabolic.subset.european.400K.QC")
z = readdlm("ukb.merged.metabolic.subset.european.400K.QC.projections.txt")
sex = parse.(Int, xdata.person_info[!, :sex]) # 1 = male, 2 = female
full_sample_age = readdlm("UKB_sample_age.txt")
cov_sampleID = full_sample_age[:, 1] |> Vector{Int}
genotype_sampleID = parse.(Int, xdata.person_info[!, :fid])
reorder_idx = indexin(genotype_sampleID, cov_sampleID) |> Vector{Int} # 104264 samples
n = length(reorder_idx)
age_reordered = zeros(n, 2)
for i in 1:n
    row = reorder_idx[i]
    for j in 1:2 # 1st column is sample ID, 2nd column is age
        age_reordered[i, j] = full_sample_age[row, j]
    end
end
all(age_reordered[:, 1] .== genotype_sampleID) || error("bug in matching genotype ID to age sample ID")
age = age_reordered[:, 2]
age_squared = age.^2
full_z = [genotype_sampleID sex age age_squared z]
column_name = ["genotype_sampleID", "sex", "age", "age_squared",
                "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8",
                "PC9", "PC10"]
z_full_df = DataFrame(full_z, column_name)
z_full_df[!, 1] = z_full_df[!, 1] |> Vector{Int} # convert Float64 columns to Int
z_full_df[!, 2] = z_full_df[!, 2] |> Vector{Int}
z_full_df[!, 3] = z_full_df[!, 3] |> Vector{Int}
z_full_df[!, 4] = z_full_df[!, 4] |> Vector{Int}
CSV.write("covariates.reordered.csv", z_full_df)

# standardize all covariates and include extra column of 1 as intercept
z_df = CSV.read("covariates.reordered.csv", DataFrame)
z = Matrix(z_df[!, 2:end])
standardize!(z)
z_full = [ones(size(z, 1)) z]
writedlm("covariates.reordered.standardized.csv", z_full, ',')

##
## Final result:
## Phenotype file (not standardized, with header and family-ID): phenotypes.reordered.csv 
## Final phenotype file (standardized, without header/family-ID): phenotypes.reordered.standardized.csv
## Final covariates file (with family ID, not standardized): covariates.reordered.csv
## Final covariates file (without family ID, standardized, extra intercept column): covariates.reordered.standardized.csv
## Final genotype file: ukb.merged.metabolic.subset.european.400K.QC.bed
##
