#!/bin/bash

# This is the main code to prepare the input for the interruption detection experiments

### --------  SLURM  ----------- ###
#SBATCH --job-name=kernel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output="out/id_%A_%a_%j.out"
#SBATCH --error="error/id_%A_%a_%j.err"
#SBATCH --mem=32G
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

set -x
set -e

var=cos
#var=window
#var=kernel
wav=../data/wav/
txt=../data/n_transcripts/
xl=../data/Finalized_CPS_Annotations/
enroll=../data/enr_files/
odir_raw=results/${var}/raw
odir_stats=results/${var}/stats
mkdir -p $odir_raw
mkdir -p $odir_stats
config=ecapa_config
#config=config

# Extract the window size for the current $SLURM_ARRAY_TASK_ID
#window=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
#kernel=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
cos=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

#echo $window
#echo $kernel
echo $cos
echo $var

source /projects/shdu9019/miniconda3/etc/profile.d/conda.sh
conda init
conda activate interD

#python overlap_main.py\
#       --wav=$wav \
#       --txt=$txt \
#       --xl=$xl \
#       --enroll=$enroll \
#       --outputdir=$odir_raw \
#       --speaker_similarity_th=$cos
#       --window_size=$window
#       --kernel_size=$kernel

python process_res.py\
       --inputdir=$odir_raw \
       --outputdir=$odir_stats \
       --speaker_similarity_th=$cos
#       --window_size=$window
#       --kernel_size=$kernel
