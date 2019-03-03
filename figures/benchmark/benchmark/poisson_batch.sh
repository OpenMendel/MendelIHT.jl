#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -l arch=intel-E5-2670,exclusive,h_rt=24:00:00,h_data=62G
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

#  Job array indexes
#$ -t 1-5:1

# run julia code
echo "julia benchmark_poisson.jl, where debias = true, n = 110300, run = $SGE_TASK_ID"
pwd; julia /u/home/b/biona001/benchmark/benchmark_poisson.jl 110300 $SGE_TASK_ID

#echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job #JOB_ID ended on:   " `date `
#echo " "
