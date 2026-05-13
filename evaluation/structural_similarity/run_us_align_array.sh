#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=1
#SBATCH -t 0-04:00:00
#SBATCH --array=98,1999
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

modulo=$((SLURM_ARRAY_TASK_MAX + 1))
remainder=$SLURM_ARRAY_TASK_ID

source /home/akubaney/projects/na_mpnn/.venv/bin/activate
python ./run_us_align_array.py "$modulo" "$remainder"