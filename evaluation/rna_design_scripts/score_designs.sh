#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=score_designs

source /home/akubaney/projects/na_mpnn/.venv/bin/activate

PROCESSED_DIR=$1
PROCESSED_REF_DIR=$2
OUTPUT_DIR=$3

# --- 1) Sanity checks ---
if [[ ! -d "$PROCESSED_DIR" ]]; then
    echo "Processed designs directory '$PROCESSED_DIR' not found!" >&2
    exit 1
fi
if [[ ! -d "$PROCESSED_REF_DIR" ]]; then
    echo "Processed reference directory '$PROCESSED_REF_DIR' not found!" >&2
    exit 1
fi
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Score output directory '$OUTPUT_DIR' not found; creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

# 2) Collect all JSON files under the processed-designs directory
shopt -s nullglob
json_files=( "$PROCESSED_DIR"/*/processed_design_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $PROCESSED_DIR/*/processed_design_json/*.json." >&2
    exit 1
fi

# 3) Compute chunk boundaries for this SLURM array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total_json + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total_json )) && END_IDX=$(( total_json - 1 ))

# 4) Process the assigned slice of JSON files
for idx in $(seq "$START_IDX" "$END_IDX"); do
    if (( idx >= total_json )); then
        break
    fi
    json_path=${json_files[idx]}
    filename=$(basename "$json_path" .json)
    input_structure_name="${filename%_*}"

    ref_json="$PROCESSED_REF_DIR/$input_structure_name/reference_json/$input_structure_name.json"

    if [[ ! -f "$ref_json" ]]; then
        echo "Reference JSON not found for $input_structure_name: '$ref_json'. Skipping." >&2
        continue
    fi

    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name score_design \
        --reference_path "$ref_json" \
        --subject_path "$json_path" \
        --overall_output_directory "$OUTPUT_DIR"
done
