function run_repeats()
    for set in 1:6, ld in 0.25:0.25:1
        # create .sh file to submit jobs
        filename = "submit.sh"
        open(filename, "w") do io
            println(io, "#!/bin/bash")
            println(io, "#")
            println(io, "#SBATCH --job-name=set$(set)LD$ld")
            println(io, "#")
            println(io, "#SBATCH --time=48:00:00")
            println(io, "#SBATCH --cpus-per-task=16")
            println(io, "#SBATCH --mem-per-cpu=3G")
            println(io, "#SBATCH --partition=owners,normal,candes")
            println(io, "")
            println(io, "#save job info on joblog:")
            println(io, "echo \"Job \$JOB_ID started on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID started on:   \" `date `")
            println(io, "")
            println(io, "# load the job environment:")
            println(io, "module load julia/1.7.2")
            println(io, "")
            println(io, "# run code")
            println(io, "export JULIA_NUM_THREADS=16")
            println(io, "echo 'julia NFBC_chr1_sim.jl set $set LD $ld'")
            println(io, "julia NFBC_chr1_sim.jl $set $ld")
            println(io, "")
            println(io, "#echo job info on joblog:")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `date `")
            println(io, "#echo \" \"")
        end
        # submit job
        run(`sbatch $filename`)
        println("submitted job set $set ld $ld")
        rm(filename, force=true)
    end
end
run_repeats()
