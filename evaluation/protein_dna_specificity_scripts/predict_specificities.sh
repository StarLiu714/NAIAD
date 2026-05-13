#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=predict_specificities

source /home/akubaney/projects/na_mpnn/.venv/bin/activate

CSV_FILE=$1
OUTPUT_DIR=$2
METHOD=$3
NUM_SAMPLES=${4:-}
TEMPERATURE=${5:-}
NA_MPNN_MODEL_PATH=${6:-}
FAMILY_CSV=${7:-}
FAMILY=${8:-}

# 1) sanity check
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi

# If FAMILY is set, FAMILY_CSV must also be set and exist.
if [[ -n "$FAMILY" && ( -z "$FAMILY_CSV" || ! -f "$FAMILY_CSV" ) ]]; then
    echo "FAMILY '$FAMILY' specified but FAMILY_CSV '$FAMILY_CSV' not found!" >&2
    exit 1
fi

# 2) read all structure_path values via Python csv.DictReader, optionally
# subsetting to structures whose id appears in FAMILY_CSV with family == FAMILY.
mapfile -t STRUCTURE_PATHS < <(
    python - "$CSV_FILE" "$FAMILY_CSV" "$FAMILY" <<'PYCODE'
import sys, pandas as pd

structure_csv_path = sys.argv[1]
family_csv_path = sys.argv[2]
family = sys.argv[3]

df = pd.read_csv(structure_csv_path)

if family:
    family_df = pd.read_csv(family_csv_path)
    matching_ids = set(family_df.loc[family_df['family'] == family, 'id'])
    df = df[df['id'].isin(matching_ids)]

for p in df['structure_path']:
    print(p)
PYCODE
)

total=${#STRUCTURE_PATHS[@]}
if (( total == 0 )); then
    echo "No data rows found in CSV (after family filter, if any)." >&2
    exit 1
fi

# 3) compute chunking based on SLURM array
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

# 4) process this shard
for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    structure_path=${STRUCTURE_PATHS[idx]}

    cmd=(
        python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py
        --function_name "predict_nucleic_acid_ppm"
        --structure_path "$structure_path"
        --overall_output_directory "$OUTPUT_DIR"
        --method "$METHOD"
    )

    if [[ -n "$NUM_SAMPLES" ]]; then
        cmd+=(--num_samples "$NUM_SAMPLES")
    fi

    if [[ -n "$TEMPERATURE" ]]; then
        cmd+=(--temperature "$TEMPERATURE")
    fi

    if [[ -n "$NA_MPNN_MODEL_PATH" ]]; then
        cmd+=(--na_mpnn_model_path "$NA_MPNN_MODEL_PATH")
    fi

    # Execute the command
    "${cmd[@]}"
done
