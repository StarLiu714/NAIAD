#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append("/home/akubaney/projects/na_mpnn/evaluation")
from na_eval_utils import save_nucleic_acid_chains_from_structure

# Paths to the csv file containing the dataset entries for all.
CSV_PATH = Path("/home/akubaney/projects/na_mpnn/data/datasets/design_dataset_v2/all.csv")

# Directory containing legacy CIF files for structures that are no longer
# available at their original paths.
LEGACY_CIF_DIR = Path("/home/akubaney/projects/na_mpnn/evaluation/structural_similarity/legacy_cifs")

# Output directory for the extracted nucleic acid chain CIF files.
CHAIN_OUT_DIR = Path("/home/akubaney/projects/na_mpnn/evaluation/structural_similarity/chain_cifs")

def main():
    modulo = int(sys.argv[1])
    remainder = int(sys.argv[2])

    CHAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(CSV_PATH)
    for i, row in enumerate(df_all.itertuples(index=False)):
        if (i % modulo) != remainder:
            continue
        
        if os.path.exists(row.structure_path):
            structure_path = row.structure_path
        else:
            structure_path = str(LEGACY_CIF_DIR / f"{row.id}.cif.gz")

        save_nucleic_acid_chains_from_structure(
            structure_path, 
            str(CHAIN_OUT_DIR)
        )

if __name__ == "__main__":
    main()