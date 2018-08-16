println("Note: Set keyword[\"use_weights\"] = true to use weights.")
if keyword["use_weights"] == true
    my_snpMAF, my_snpweights, notused_snpmatrix = calculatePriorWeightsforIHT(snpdata,y,k,v,keyword)
    # NOTICE - WE ARE NOT USING MY snpmatrix, just my_snpweights and my_snpMAF
    hold_std_vec = deepcopy(std_vec)
    my_snpweights  = [my_snpweights ones(size(my_snpweights, 1))]
    println("sizeof(std_vec) = $(sizeof(std_vec))")
    println("sizeof(my_snpweights) = $(sizeof(my_snpweights))")
    Base.A_mul_B!(std_vec, diagm(hold_std_vec), my_snpweights[1,:])
end
