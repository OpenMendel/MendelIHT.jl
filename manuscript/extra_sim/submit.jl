function run_repeats()
    for seed in 1:100
        # create .sh file to submit jobs
        filename = "sim$seed.sh"
        open(filename, "w") do io
            println(io, "#!/bin/bash")
            println(io, "#")
            println(io, "#SBATCH --job-name=sim$seed")
            println(io, "#")
            println(io, "#SBATCH --time=1:00:00")
            println(io, "#SBATCH --cpus-per-task=8")
            println(io, "#SBATCH --mem-per-cpu=2G")
            println(io, "#SBATCH --partition=owners")
            println(io, "")
            println(io, "#save job info on joblog:")
            println(io, "echo \"Job \$JOB_ID started on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID started on:   \" `date `")
            println(io, "")
            println(io, "# load the job environment:")
            println(io, "module load julia")
            println(io, "")
            println(io, "# run code")
            println(io, "export JULIA_NUM_THREADS=8")
            println(io, "echo 'julia extra_sim.jl $seed'")
            println(io, "julia extra_sim.jl $seed")
            println(io, "")
            println(io, "#echo job info on joblog:")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `date `")
            println(io, "#echo \" \"")
        end

        # submit job
        run(`sbatch $filename`)
        println("submitted job $seed")
        rm(filename, force=true)
        # sleep(1)
    end
end
run_repeats()
