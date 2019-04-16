#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -l arch=intel-E5-2670,exclusive,h_rt=1:00:00,h_data=3G
# Email address to notify
## $ -M $USER@mal
# Notify when:
#$ -m bea

#save job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `

# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load julia/1.0.1
module li
which julia

# run julia code
echo 'julia benchmark_l0_reg.jl normal 50000 1000000 10 true'
pwd; julia /u/home/b/biona001/benchmark/benchmark_l0_reg.jl normal 500 1000 10 true #glm, n, p, k, debias

#echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job #JOB_ID ended on:   " `date `
#echo " "
