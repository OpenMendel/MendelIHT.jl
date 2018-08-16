result, my_snpMAF, my_snpweights, x, y, v  = L0_reg2(snpmatrix, snpdata, phenotype, J, k, groups, keyword)
printConvergenceReport(result, my_snpMAF, my_snpweights, x, y, v, snp_definition_frame)
return result
