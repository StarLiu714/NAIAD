#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=48g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=process_natives

source /home/akubaney/projects/na_mpnn/.venv/bin/activate

CSV_FILE=$1
PROCESSED_REF_DIR=$2
STRUCTURE_FILTER=${3:-all}

# 1) sanity check
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi

# 2) Load filtered structure_path values from the CSV into a Bash array
mapfile -t PDB_PATHS < <(
    python - "$CSV_FILE" "$STRUCTURE_FILTER" <<'PYCODE'
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/akubaney/projects/na_mpnn/evaluation")

from na_eval_utils import extract_sequences_from_structure, prepare_complex_sequence_data

csv_file, structure_filter = sys.argv[1:3]

def classify_row(row):
    structure_path = os.path.abspath(row["structure_path"])
    na_sequence_data, protein_sequences = extract_sequences_from_structure(
        structure_path
    )
    complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = na_sequence_data,
        protein_sequences = protein_sequences
    )

    return {
        "has_protein": complex_sequence_data["has_protein"],
    }

def passes_structure_filter(metadata):
    if structure_filter in ("", "all"):
        return True
    if structure_filter in ("na_only",):
        return not metadata["has_protein"]
    if structure_filter in (
        "with_protein",
    ):
        return metadata["has_protein"]

    raise ValueError(
        f"Unsupported structure filter: {structure_filter}"
    )

df = pd.read_csv(csv_file)

for _, row in df.iterrows():
    metadata = classify_row(row)
    if passes_structure_filter(metadata):
        print(row["structure_path"])
PYCODE
)

# 3) Compute chunk boundaries for this SLURM array task
total=${#PDB_PATHS[@]}
if (( total == 0 )); then
    echo "No data rows found in CSV for filter '$STRUCTURE_FILTER'." >&2
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

# 4) Process the assigned slice of structure paths
for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    pdb_path=${PDB_PATHS[idx]}
    echo "$pdb_path"
    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name process_reference \
        --reference_structure_path "$pdb_path" \
        --overall_output_directory "$PROCESSED_REF_DIR"
done
