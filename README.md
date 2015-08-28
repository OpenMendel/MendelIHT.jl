# PLINK.jl

A module to handle PLINK binary genotype files in Julia.
This package furnishes decompression routines for PLINK `.bed` files.
The compressed genotype matrix `X` is stored as a string of `Int8` components,
each which stores four genotypes. 
PLINK.jl also provides linear algebra routines that decompress `X` on the fly,
including both `X * y` and `X' * y`. 
