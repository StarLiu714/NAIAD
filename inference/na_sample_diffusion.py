"""
Inference script for NAIAD.

This implements iterative denoising for sequence generation:
- Start from all-masked sequence
- Iteratively unmask positions based on model confidence
- Optional rewalk mechanism for refinement

Usage:
    python inference/na_sample_diffusion.py \
        --checkpoint model.pt \
        --pdb_path structure.pdb \
        --output_dir results/ \
        --num_steps 50 \
        --num_samples 10

Inspired by:
- ProRefiner: Iterative refinement
- MaskGIT: Confidence-based unmasking
- DPLM: Discrete diffusion for proteins
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from typing import Optional, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDBX_ROOT = os.path.join(REPO_ROOT, "pdbx")
for path in (PDBX_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.cvar.obErrorLog.StopLogging()

import cifutils
import pdbutils
from na_data_utils import PDBDataset
from na_model_utils import featurize, ProteinMPNNDiffusion
from na_diffusion_utils import (
    iterative_denoise, get_na_mask_from_polymer_masks,
    get_denoising_schedule, select_positions_to_unmask, rewalk_positions
)


def load_model_and_dataset(checkpoint_path: str, config_path: Optional[str] = None):
    """Load trained diffusion model and create dataset."""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load or infer config
    if config_path:
        params = json.load(open(config_path))
    else:
        # Use default parameters
        params = {
            "ATOMS_TO_LOAD": "backbone",
            "PARSE_PROTEIN": 1,
            "PARSE_DNA": 1,
            "PARSE_RNA": 1,
            "PARSE_RNA_AS_DNA": 0,
            "NA_SHARED_TOKENS": 1,
            "PROTEIN_BACKBONE_OCC_CUTOFF": 0.8,
            "PROTEIN_SIDE_CHAIN_OCC_CUTOFF": 0.5,
            "DNA_BACKBONE_OCC_CUTOFF": 0.8,
            "DNA_SIDE_CHAIN_OCC_CUTOFF": 0.5,
            "RNA_BACKBONE_OCC_CUTOFF": 0.8,
            "RNA_SIDE_CHAIN_OCC_CUTOFF": 0.5,
            "CROP_LARGE_STRUCTURES": 0,
            "BATCH_TOKENS": 6000,
            "NA_REF_ATOM": "C1'",
            "PARSE_PPMS": 0,
            "MIN_OVERLAP_LENGTH": 5,
            "DROP_PROTEIN_PROBABILITY": 0,
            "NA_ONLY_AS_UNIFORM_PPM": 0,
            "PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY": 0,
            "MUTATE_BASE_PAIR_TOGETHER": 0,
            "MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY": 0,
            "NA_NON_INTERFACE_AS_UNIFORM_PPM": 0,
            "EXCLUDE_RES": [],
            "RANDOMIZE_NMR_MODEL": 0,
            "HIDDEN_DIM": 128,
            "NUM_ENCODER_LAYERS": 3,
            "NUM_DECODER_LAYERS": 3,
            "NUM_NEIGHBORS": 32,
            "DROPOUT": 0.1,
            "PROTEIN_BACKBONE_NOISE": 0.0,
            "DNA_BACKBONE_NOISE": 0.0,
            "RNA_BACKBONE_NOISE": 0.0,
            "INCLUDE_PRED_NA_N": 1,
            "USE_SEQUENCE_CONTEXT": True,
            "VOCAB_SIZE": 33,
            "NUM_LETTERS": 33,
        }
    
    # Atom list
    if params["ATOMS_TO_LOAD"] == "backbone":
        atom_list_to_save = [
            'N', 'CA', 'C', 'O',
            'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
            "C2'", "O2'", "C1'"
        ]
    else:
        atom_list_to_save = [
            'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 
            'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 
            'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 
            'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
            'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
            "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 
            'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1'
        ]
    
    # Create parsers and dataset
    cif_parser = cifutils.CIFParser(
        skip_res=params.get("EXCLUDE_RES", []),
        randomize_nmr_model=params.get("RANDOMIZE_NMR_MODEL", 0)
    )
    pdb_parser = pdbutils.PDBParser()
    
    pdb_dataset = PDBDataset(
        cif_parser=cif_parser,
        pdb_parser=pdb_parser,
        atom_list_to_save=atom_list_to_save,
        parse_protein=params["PARSE_PROTEIN"],
        parse_dna=params["PARSE_DNA"],
        parse_rna=params["PARSE_RNA"],
        parse_rna_as_dna=params["PARSE_RNA_AS_DNA"],
        na_shared_tokens=params["NA_SHARED_TOKENS"],
        protein_backbone_occ_cutoff=params["PROTEIN_BACKBONE_OCC_CUTOFF"],
        protein_side_chain_occ_cutoff=params["PROTEIN_SIDE_CHAIN_OCC_CUTOFF"],
        dna_backbone_occ_cutoff=params["DNA_BACKBONE_OCC_CUTOFF"],
        dna_side_chain_occ_cutoff=params["DNA_SIDE_CHAIN_OCC_CUTOFF"],
        rna_backbone_occ_cutoff=params["RNA_BACKBONE_OCC_CUTOFF"],
        rna_side_chain_occ_cutoff=params["RNA_SIDE_CHAIN_OCC_CUTOFF"],
        crop_large_structures=params["CROP_LARGE_STRUCTURES"],
        batch_tokens=params["BATCH_TOKENS"],
        na_ref_atom=params["NA_REF_ATOM"],
        parse_ppms=params["PARSE_PPMS"],
        min_overlap_length=params["MIN_OVERLAP_LENGTH"],
        drop_protein_probability=0,  # No dropout during inference
        na_only_as_uniform_ppm=params["NA_ONLY_AS_UNIFORM_PPM"],
        protein_interface_residue_mutation_probability=0,
        mutate_base_pair_together=0,
        mutate_entire_side_chain_interface_probability=0,
        na_non_interface_as_uniform_ppm=params["NA_NON_INTERFACE_AS_UNIFORM_PPM"]
    )
    
    # Create model
    model = ProteinMPNNDiffusion(
        node_features=params["HIDDEN_DIM"],
        edge_features=params["HIDDEN_DIM"],
        hidden_dim=params["HIDDEN_DIM"],
        num_encoder_layers=params["NUM_ENCODER_LAYERS"],
        num_decoder_layers=params["NUM_DECODER_LAYERS"],
        k_neighbors=params["NUM_NEIGHBORS"],
        dropout=0.0,  # No dropout during inference
        atom_dict=pdb_dataset.atom_dict,
        restype_to_int=pdb_dataset.restype_to_int,
        polytype_to_int=pdb_dataset.polytype_to_int,
        protein_augment_eps=0.0,
        dna_augment_eps=0.0,
        rna_augment_eps=0.0,
        na_ref_atom=params["NA_REF_ATOM"],
        include_pred_na_N=params["INCLUDE_PRED_NA_N"],
        use_sequence_context=params.get("USE_SEQUENCE_CONTEXT", True),
        device=device,
        vocab=params["VOCAB_SIZE"],
        num_letters=params["NUM_LETTERS"]
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, pdb_dataset, params, device


def load_structure(pdb_path: str, pdb_dataset: PDBDataset, device: torch.device):
    """Load and featurize a structure for inference."""
    
    # Parse structure
    if pdb_path.endswith('.pdb') or pdb_path.endswith('.pdb.gz'):
        chains, asmb, covalei, meta = pdb_dataset.pdb_parser.parse(pdb_path)
    else:
        chains, asmb, covalei, meta = pdb_dataset.cif_parser.parse(pdb_path)
    
    # Load chains
    macromolecule_chain_dict = pdb_dataset.load_chains(chains)
    
    # Get first assembly
    assembly_id = list(asmb.keys())[0]
    
    # Load assembly
    out_dict = pdb_dataset.load_assembly(
        macromolecule_chain_dict, asmb, assembly_id, ppms=[]
    )
    
    # Create dummy preprocessed data (zeros for inference)
    L = out_dict["macromolecule_L"]
    out_dict["interface_mask"] = np.zeros(L, dtype=np.int32)
    out_dict["side_chain_interface_mask"] = np.zeros(L, dtype=np.int32)
    out_dict["nearest_protein_side_chain_index"] = np.zeros(L, dtype=np.int64)
    out_dict["base_pair_mask"] = np.zeros(L, dtype=np.int32)
    out_dict["base_pair_index"] = np.zeros(L, dtype=np.int64)
    out_dict["canonical_base_pair_mask"] = np.zeros(L, dtype=np.int32)
    out_dict["canonical_base_pair_index"] = np.zeros(L, dtype=np.int64)
    
    out_dict["structure_path"] = pdb_path
    out_dict["assembly_id"] = assembly_id
    out_dict["ppm_paths"] = "[]"
    out_dict["ppm_paths_chosen"] = []

    for key, value in list(out_dict.items()):
        if isinstance(value, np.ndarray):
            out_dict[key] = torch.from_numpy(value)
    
    # Create batch
    batch = [[(out_dict, torch.tensor(L, dtype=torch.long))]]
    
    # Featurize
    feature_dict = featurize(
        batch[0], 
        pdb_dataset.polytype_to_int, 
        pdb_dataset.restype_to_int, 
        pdb_dataset.atom_dict, 
        device
    )
    
    return feature_dict, out_dict


def sequence_to_string(
    seq_tensor: torch.Tensor, 
    pdb_dataset: PDBDataset,
    mask: Optional[torch.Tensor] = None
) -> str:
    """Convert sequence tensor to string representation."""
    seq = seq_tensor.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()
    
    result = []
    for i, token_idx in enumerate(seq):
        if mask is not None and mask[i] == 0:
            continue
        restype = pdb_dataset.int_to_restype.get(token_idx, 'X')
        char = pdb_dataset.restype_3_to_1.get(restype, 'X')
        result.append(char)
    
    return ''.join(result)


@torch.no_grad()
def sample_sequences(
    model: ProteinMPNNDiffusion,
    feature_dict: dict,
    pdb_dataset: PDBDataset,
    num_samples: int = 10,
    num_steps: int = 50,
    schedule_type: str = "cosine",
    selection_strategy: str = "confidence",
    temperature: float = 1.0,
    rewalk_enabled: bool = False,
    rewalk_ratio: float = 0.1,
    rewalk_threshold: float = 0.3,
    mask_na_only: bool = True,
    use_argmax: bool = True,
    verbose: bool = True,
) -> List[Tuple[str, torch.Tensor, list]]:
    """
    Generate multiple sequence samples using iterative denoising.
    
    Args:
        model: Trained diffusion model
        feature_dict: Featurized structure
        pdb_dataset: Dataset for token conversion
        num_samples: Number of sequences to generate
        num_steps: Number of denoising steps
        schedule_type: Denoising schedule (linear, cosine, sqrt)
        selection_strategy: How to select positions to unmask
        temperature: Sampling temperature
        rewalk_enabled: Enable rewalk mechanism
        rewalk_ratio: Proportion of low-conf positions to re-mask
        rewalk_threshold: Confidence threshold for rewalk
        mask_na_only: Only generate NA sequences (keep protein fixed)
        use_argmax: Use deterministic argmax decoding instead of sampling
        verbose: Print progress
    
    Returns:
        List of (sequence_string, sequence_tensor, trajectory) tuples
    """
    device = next(model.parameters()).device
    mask_token_idx = pdb_dataset.restype_to_int["MAS"]
    
    # Get NA mask
    if mask_na_only:
        na_mask = get_na_mask_from_polymer_masks(
            feature_dict["dna_mask"],
            feature_dict["rna_mask"]
        ).float()
    else:
        na_mask = None
    
    results = []
    
    for sample_idx in range(num_samples):
        if verbose:
            print(f"\nGenerating sample {sample_idx + 1}/{num_samples}...")
        
        # Run iterative denoising
        final_seq, trajectory = iterative_denoise(
            model=model,
            feature_dict=feature_dict,
            mask_token_idx=mask_token_idx,
            num_steps=num_steps,
            schedule_type=schedule_type,
            selection_strategy=selection_strategy,
            temperature=temperature,
            rewalk_enabled=rewalk_enabled,
            rewalk_ratio=rewalk_ratio,
            rewalk_threshold=rewalk_threshold,
            na_mask=na_mask,
            use_argmax=use_argmax,
            verbose=verbose,
        )
        
        # Convert to string (only NA positions)
        if mask_na_only:
            seq_str = sequence_to_string(
                final_seq[0], pdb_dataset, 
                mask=(feature_dict["dna_mask"][0] + feature_dict["rna_mask"][0])
            )
        else:
            seq_str = sequence_to_string(final_seq[0], pdb_dataset, feature_dict["mask"][0])
        
        results.append((seq_str, final_seq, trajectory))
        
        if verbose:
            print(f"  Generated sequence: {seq_str[:50]}..." if len(seq_str) > 50 else f"  Generated sequence: {seq_str}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate sequences using NA-MPNN diffusion model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON (optional, will use defaults)"
    )
    parser.add_argument(
        "--pdb_path", type=str, required=True,
        help="Path to input PDB/CIF structure"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./diffusion_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--schedule", type=str, default="cosine",
        choices=["linear", "cosine", "sqrt"],
        help="Denoising schedule type"
    )
    parser.add_argument(
        "--strategy", type=str, default="confidence",
        choices=["confidence", "entropy", "random"],
        help="Position selection strategy"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--rewalk", action="store_true",
        help="Enable rewalk mechanism for refinement"
    )
    parser.add_argument(
        "--rewalk_ratio", type=float, default=0.1,
        help="Proportion of positions to re-mask in rewalk"
    )
    parser.add_argument(
        "--rewalk_threshold", type=float, default=0.3,
        help="Confidence threshold for rewalk"
    )
    parser.add_argument(
        "--mask_all", action="store_true",
        help="Mask all positions (not just NA)"
    )
    parser.add_argument(
        "--argmax", action="store_true",
        help="Use deterministic argmax decoding for every sample"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Sample tokens from the temperature-scaled distribution"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model, pdb_dataset, params, device = load_model_and_dataset(
        args.checkpoint, args.config
    )
    
    print(f"Loading structure from {args.pdb_path}...")
    feature_dict, out_dict = load_structure(args.pdb_path, pdb_dataset, device)
    
    print(f"\nStructure info:")
    print(f"  Total residues: {out_dict['macromolecule_L']}")
    print(f"  Protein: {out_dict['protein_L']}")
    print(f"  DNA: {out_dict['dna_L']}")
    print(f"  RNA: {out_dict['rna_L']}")
    
    # Get original NA sequence for comparison
    original_seq = sequence_to_string(
        feature_dict["S"][0], pdb_dataset,
        mask=(feature_dict["dna_mask"][0] + feature_dict["rna_mask"][0])
    )
    print(f"\nOriginal NA sequence: {original_seq}")
    
    print(f"\nGenerating {args.num_samples} sequences with {args.num_steps} denoising steps...")
    print(f"  Schedule: {args.schedule}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Rewalk: {args.rewalk}")
    use_argmax = args.argmax or (not args.sample and args.num_samples == 1)
    print(f"  Decoding: {'argmax' if use_argmax else 'sampling'}")
    
    results = sample_sequences(
        model=model,
        feature_dict=feature_dict,
        pdb_dataset=pdb_dataset,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        schedule_type=args.schedule,
        selection_strategy=args.strategy,
        temperature=args.temperature,
        rewalk_enabled=args.rewalk,
        rewalk_ratio=args.rewalk_ratio,
        rewalk_threshold=args.rewalk_threshold,
        mask_na_only=not args.mask_all,
        use_argmax=use_argmax,
        verbose=True,
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, "generated_sequences.fasta")
    with open(output_file, 'w') as f:
        f.write(f">original\n{original_seq}\n")
        for i, (seq_str, seq_tensor, trajectory) in enumerate(results):
            f.write(f">sample_{i+1}\n{seq_str}\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Save detailed results
    results_json = {
        "input_structure": args.pdb_path,
        "original_sequence": original_seq,
        "num_steps": args.num_steps,
        "schedule": args.schedule,
        "strategy": args.strategy,
        "temperature": args.temperature,
        "rewalk_enabled": args.rewalk,
        "use_argmax": use_argmax,
        "samples": [
            {"index": i, "sequence": seq_str}
            for i, (seq_str, _, _) in enumerate(results)
        ]
    }
    
    json_file = os.path.join(args.output_dir, "results.json")
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Detailed results saved to {json_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATED SEQUENCES:")
    print("="*60)
    for i, (seq_str, _, _) in enumerate(results):
        # Calculate sequence identity with original
        matches = sum(a == b for a, b in zip(seq_str, original_seq))
        identity = matches / len(original_seq) * 100 if original_seq else 0
        print(f"Sample {i+1}: {seq_str[:60]}{'...' if len(seq_str) > 60 else ''} (Identity: {identity:.1f}%)")


if __name__ == "__main__":
    main()
