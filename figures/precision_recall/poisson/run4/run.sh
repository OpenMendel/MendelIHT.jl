#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -l h_rt=24:00:00,h_data=10G
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
module load R/3.5.1
module li
which julia

# run julia code
echo "julia run.jl, run = 4"
pwd; julia /u/home/b/biona001/precision_recall/poisson/run4/run.jl

#echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job #JOB_ID ended on:   " `date `
#echo " "
