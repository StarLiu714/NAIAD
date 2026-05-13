#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=design_sequences

source /home/akubaney/projects/na_mpnn/.venv/bin/activate

CSV_FILE=$1
OUTPUT_DIR=$2
METHOD=$3
NUM_SAMPLES=$4
TEMPERATURE=${5:-}
NA_MPNN_MODEL_PATH=${6:-}
WITH_PROTEIN=${7:-1}
STRUCTURE_FILTER=${8:-all}

# 1) sanity check
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi

# 2) read structure_path values via Python with optional filtering
mapfile -t STRUCTURE_PATHS < <(
    python - "$CSV_FILE" "$METHOD" "$WITH_PROTEIN" "$STRUCTURE_FILTER" <<'PYCODE'
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/akubaney/projects/na_mpnn/evaluation")

from na_eval_utils import extract_sequences_from_structure, prepare_complex_sequence_data

csv_file, method, with_protein_raw, structure_filter = sys.argv[1:5]
with_protein = bool(int(with_protein_raw))

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
        "has_dna": complex_sequence_data["has_dna"],
        "is_single_rna_chain": complex_sequence_data["is_single_rna_chain"],
        "is_monomer_rna": complex_sequence_data["is_monomer_rna"],
        "is_protein_monomer_rna": (
            complex_sequence_data["is_single_rna_chain"] and
            complex_sequence_data["has_protein"]
        ),
    }

def passes_structure_filter(metadata):
    if structure_filter in ("", "all"):
        return True
    if structure_filter == "monomer_rna":
        return metadata["is_monomer_rna"]
    if structure_filter == "protein_monomer_rna":
        return metadata["is_protein_monomer_rna"]
    if structure_filter == "has_protein":
        return metadata["has_protein"]

    raise ValueError(
        f"Unsupported structure filter: {structure_filter}"
    )

df = pd.read_csv(csv_file)

for _, row in df.iterrows():
    metadata = classify_row(row)
    if not passes_structure_filter(metadata):
        continue

    print(row["structure_path"])
PYCODE
)

total=${#STRUCTURE_PATHS[@]}
if (( total == 0 )); then
    echo "No data rows found in CSV." >&2
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
        --function_name "design_nucleic_acid_sequence"
        --structure_path "$structure_path"
        --overall_output_directory "$OUTPUT_DIR"
        --num_samples "$NUM_SAMPLES"
        --method "$METHOD"
    )

    if [[ -n "$TEMPERATURE" ]]; then
        cmd+=(--temperature "$TEMPERATURE")
    fi

    if [[ -n "$NA_MPNN_MODEL_PATH" ]]; then
        cmd+=(--na_mpnn_model_path "$NA_MPNN_MODEL_PATH")
    fi

    cmd+=(--with_protein "$WITH_PROTEIN")
    # Execute the command
    "${cmd[@]}"
done
