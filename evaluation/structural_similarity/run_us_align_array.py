import sys
from pathlib import Path
import pandas as pd

sys.path.append("/home/akubaney/projects/na_mpnn/evaluation")
from na_eval_utils import run_us_align

# Paths to dataset splits.
TRAIN_CSV = Path("/home/akubaney/projects/na_mpnn/data/datasets/design_dataset_v2/train.csv")
VALID_CSV = Path("/home/akubaney/projects/na_mpnn/data/datasets/design_dataset_v2/valid.csv")
TEST_CSV  = Path("/home/akubaney/projects/na_mpnn/data/datasets/design_dataset_v2/test.csv")

# Chain CIFs directory (expects xxxx_yy...yy.cif.gz)
CHAIN_DIR = Path("/home/akubaney/projects/na_mpnn/evaluation/structural_similarity/chain_cifs")

# Output directory (writes one CSV per subject chain)
OUT_DIR = Path("/home/akubaney/projects/na_mpnn/evaluation/structural_similarity/usalign_per_chain")

def main():
    modulo = int(sys.argv[1])
    remainder = int(sys.argv[2])

    # Load splits; assume ids are already lowercase, but normalize anyway.
    train_ids = set(pd.read_csv(TRAIN_CSV)["id"].astype(str).str.lower())
    valid_ids = set(pd.read_csv(VALID_CSV)["id"].astype(str).str.lower())
    test_ids  = set(pd.read_csv(TEST_CSV)["id"].astype(str).str.lower())

    # Scan all chain CIFs once and bucket them by split membership.
    # Tuples are (pdb_id, chain_id, chain_structure_path).
    train_chains = []
    valid_chains = []
    test_chains = []

    for chain_structure_path in sorted(CHAIN_DIR.glob("**/*.cif")):
        pdb_and_chain_id = chain_structure_path.name.split(".", 1)[0]
        pdb_id, chain_id = pdb_and_chain_id.split("_", 1)
        chain_path_str = str(chain_structure_path)

        if pdb_id in train_ids:
            train_chains.append((pdb_id, chain_id, chain_path_str))
        elif pdb_id in valid_ids:
            valid_chains.append((pdb_id, chain_id, chain_path_str))
        elif pdb_id in test_ids:
            test_chains.append((pdb_id, chain_id, chain_path_str))
        else:
            raise ValueError(
                f"Chain CIF {chain_structure_path} has pdb_id {pdb_id} not found in any split!"
            )

    # References are test only.
    reference_chains = sorted(test_chains)

    # Subjects are train + valid.
    subject_chains = sorted(train_chains + valid_chains)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Each array task handles a subset of subject chains.
    for subject_index, (subject_pdb_id, subject_chain_id, subject_chain_structure_path) in enumerate(subject_chains):
        if (subject_index % modulo) != remainder:
            continue
        
        # Create output sub-directory.
        subject_out_dir = OUT_DIR / f"{subject_pdb_id[1:3]}"
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        out_csv_path = subject_out_dir / f"{subject_pdb_id}_{subject_chain_id}.csv"

        # If output CSV already exists, check if it's complete.
        # If complete, skip. If incomplete or unreadable, recompute.
        if out_csv_path.exists():
            try:
                existing_df = pd.read_csv(out_csv_path)
                if len(existing_df) == len(reference_chains):
                    continue
            except Exception as e:
                pass

        rows = []
        for (
            reference_pdb_id, 
            reference_chain_id, 
            reference_chain_structure_path
        ) in reference_chains:
            # Reference is test chain, subject is train/valid chain.
            result = run_us_align(
                reference_chain_structure_path, 
                subject_chain_structure_path
            )
            rows.append({
                "reference_pdb_id": reference_pdb_id,
                "reference_chain_id": reference_chain_id,
                "subject_pdb_id": subject_pdb_id,
                "subject_chain_id": subject_chain_id,
                **result,
            })

        pd.DataFrame(rows).to_csv(out_csv_path, index=False)

if __name__ == "__main__":
    main()