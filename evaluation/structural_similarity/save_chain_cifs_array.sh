#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=1
#SBATCH -t 0-01:00:00
#SBATCH --array=0-59
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

modulo=$((SLURM_ARRAY_TASK_MAX + 1))
remainder=$SLURM_ARRAY_TASK_ID

source /home/akubaney/projects/na_mpnn/.venv/bin/activate
python ./save_chain_cifs_array.py "$modulo" "$remainder"