#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -pe shared 8
#$ -l h_rt=24:00:00,h_data=2G
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
echo 'julia negbin debias'
export JULIA_NUM_THREADS=16
pwd; julia /u/home/b/biona001/generate_repeats/negativebinomial_repeats.jl

#echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job #JOB_ID ended on:   " `date `
#echo " "
