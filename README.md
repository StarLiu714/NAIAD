**N**ucleic **A**cid **I**nverse-folding with **A**bsorbing **D**iffusion
======================================

NAIAD is a diffusion model for inverse design of DNA and RNA, with support for protein-nucleic-acid complexes.

The structure dataset is not stored in this GitHub repository. It is published separately on Hugging Face:

https://huggingface.co/datasets/StarLiu714/NAIAD

See [Data](#mmcif-data-for-training) for HF dataset download and local path setup.

## Repository Layout

- `na_run_diffusion.py`: diffusion training entry point.
- `inference/na_sample_diffusion.py`: diffusion inference and sequence sampling entry point.
- `inference/examples/`: example PDB inputs.
- `na_model_utils.py`, `na_diffusion_utils.py`, `na_data_utils.py`: model, diffusion, and data-loading code.
- `configs/irm_enhanced_diffusion_training.json`: current diffusion training config.
- `evaluation/`: evaluation helpers and downstream analysis utilities.
- `splits/`: train/valid/test design split IDs.
- `models/s1836.pt`: current NAIAD checkpoint used for training warm-starts and design inference.


## Install

Use conda for OpenBabel and the scientific stack. 
This repository currently does not include a locked environment file, so create an environment with the core dependencies used by the codebase:
```bash
conda create -n naiad python=3.10 -y
conda activate naiad
conda install -c conda-forge openbabel numpy pandas scipy networkx tqdm -y
```

Install PyTorch for your CUDA or CPU setup from the official PyTorch
instructions, then install any remaining Python packages required by your run such as `wandb` if logging is enabled.

Initialize the `pdbx` submodule after cloning:
```bash
git submodule update --init --recursive
export PYTHONPATH="$PWD/pdbx${PYTHONPATH:+:$PYTHONPATH}"
```

The code imports `pdbx.reader` from the submodule, so `pdbx/` must be on `PYTHONPATH`.



## Inference

Generate sequences for a single PDB or mmCIF structure with the diffusion sampler:
```bash
python inference/na_sample_diffusion.py \
  --checkpoint models/s1836.pt \
  --config configs/irm_enhanced_diffusion_training.json \
  --pdb_path inference/examples/4oqu.pdb \
  --output_dir outputs/diffusion_4oqu \
  --num_samples 10 \
  --num_steps 30 \
  --temperature 1.0 \
  --seed 37
```

The same command accepts `.cif` input by passing the mmCIF path to `--pdb_path`.
By default, the sampler designs nucleic-acid residues and keeps protein residues fixed. Add `--mask_all` only if you want to redesign all parsed polymer residues.

Outputs:
```text
outputs/diffusion_4oqu/generated_sequences.fasta
outputs/diffusion_4oqu/results.json
```

Use `inference/na_sample_diffusion.py` for the current diffusion checkpoint `models/s1836.pt`. The remaining legacy autoregressive helpers under `inference/` are not used by the diffusion sampler and should not be used with `models/s1836.pt`.


## mmCIF Data (for training)

Download the Hugging Face dataset snapshot into a local folder:
```bash
huggingface-cli download StarLiu714/NAIAD --repo-type dataset --local-dir data/naiad_dataset
```

The current training config expects CSV indexes at:
```text
data/naiad_dataset/train_dna_enhanced.csv
data/naiad_dataset/valid_dna_enhanced.csv
```

If the CSV `structure_path` entries are relative paths, set the dataset root:
```bash
export NAIAD_DATASET_ROOT=data/naiad_dataset
```

The CIF parser also needs an RCSB chemical component cache:
```bash
export NAIAD_RCSB_CIF_DIR=data/datasets/rcsb_cif
```


## Training

Run the current IRM-enhanced diffusion training config:
```bash
python na_run_diffusion.py configs/irm_enhanced_diffusion_training.json
```

The current checkpoint in this repository is:
```text
models/s1836.pt
```

To warm-start from it, set the config value:
```json
"PREV_CHECKPOINT": "models/s1836.pt"
```

Checkpoints are written under `BASE_FOLDER`, which is currently `models`.
