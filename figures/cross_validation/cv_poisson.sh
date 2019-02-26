#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -pe shared 8 
#$ -l arch=intel-E5-2670,exclusive,h_rt=24:00:00,h_data=6G
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
echo "julia poisson_cv.jl with 16 thread, debias=true"
pwd; export JULIA_NUM_THREADS=16; julia /u/home/b/biona001/cross_validation/poisson_cv.jl

#echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job #JOB_ID ended on:   " `date `
#echo " "
