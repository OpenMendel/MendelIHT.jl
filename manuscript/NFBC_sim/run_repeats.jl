function run_repeats()
    for set in 1:9
        # create .sh file to submit jobs
        filename = "poly.set$set.sh"
        open(filename, "w") do io
            println(io, "#!/bin/bash")
            println(io, "#\$ -cwd")
            println(io, "# error = Merged with joblog")
            println(io, "#\$ -o joblog.\$JOB_ID")
            println(io, "#\$ -j y")
            println(io, "#\$ -pe shared 16")
            println(io, "#\$ -l h_rt=24:00:00,h_data=1G,arch=intel-gold-61*")
            println(io, "# Email address to notify")
            println(io, "## \$ -M \$USER@mal")
            println(io, "# Notify when:")
            println(io, "#\$ -m bea")
            println(io, "")
            println(io, "#save job info on joblog:")
            println(io, "echo \"Job \$JOB_ID started on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID started on:   \" `date `")
            println(io, "")
            println(io, "# load the job environment:")
            println(io, ". /u/local/Modules/default/init/modules.sh")
            println(io, "module load julia/1.5.4")
            println(io, "module li")
            println(io, "which julia")
            println(io, "")
            println(io, "# run code")
            println(io, "export JULIA_NUM_THREADS=16")
            println(io, "echo 'julia NFBC_chr1_sim.jl (run IHT/GEMMA/mvPLINK on NFBC data (polygenic model))'")
            println(io, "pwd; julia NFBC_chr1_sim.jl $set")
            println(io, "")
            println(io, "#echo job info on joblog:")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `date `")
            println(io, "#echo \" \"")
        end
        
        # submit job
        run(`qsub $filename`)
        rm(filename, force=true)
        sleep(2)
    end
end
run_repeats()
