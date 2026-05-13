#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=48g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=process_designs

source /home/akubaney/projects/na_mpnn/.venv/bin/activate

SPECIFIED_DIRECTORY=$1
OUTPUT_DIR=$2
REFERENCE_DIR=$3

# 1) sanity checks
if [[ ! -d "$SPECIFIED_DIRECTORY" ]]; then
    echo "Directory '$SPECIFIED_DIRECTORY' not found!" >&2
    exit 1
fi
if [[ ! -d "$REFERENCE_DIR" ]]; then
    echo "Reference directory '$REFERENCE_DIR' not found!" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "Error: jq required but not on PATH." >&2
    exit 1
fi

# 2) Collect all design JSON files under the specified directory
shopt -s nullglob
json_files=( "$SPECIFIED_DIRECTORY"/*/design_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $SPECIFIED_DIRECTORY/*/design_json/." >&2
    exit 1
fi

# 3) Compute chunk boundaries for this SLURM array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total_json + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total_json )) && END_IDX=$(( total_json - 1 ))

# 4) Process the assigned chunk
for idx in $(seq "$START_IDX" "$END_IDX"); do
    if (( idx >= total_json )); then
        break
    fi
    json_file=${json_files[idx]}

    input_structure_name=$(jq -r '.input_structure_name // empty' "$json_file")
    if [[ -z "$input_structure_name" || "$input_structure_name" == "null" ]]; then
        echo "Design '$json_file' is missing input_structure_name. Skipping." >&2
        continue
    fi
    ref_json="$REFERENCE_DIR/$input_structure_name/reference_json/$input_structure_name.json"
    if [[ ! -f "$ref_json" ]]; then
        echo "Reference JSON not found for $input_structure_name: '$ref_json'. Skipping." >&2
        continue
    fi

    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name "process_design" \
        --subject_path "$json_file" \
        --overall_output_directory "$OUTPUT_DIR" \
        --reference_path "$ref_json"
done
