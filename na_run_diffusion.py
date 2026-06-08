"""
Training script for NAIAD.

This implements:
- Dynamic masking during training (forward diffusion process)
- DPLM-style loss weighting based on mask ratio
- Configurable noise schedules
- Support for refinement-focused training

Usage:
    python na_run_diffusion.py diffusion_config.json

Inspired by:
- ProRefiner: Iterative refinement for protein design
- DPLM: Discrete Diffusion Protein Language Model
- MaskGIT: Masked generative image transformer
"""

from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.cvar.obErrorLog.StopLogging()

import numpy as np
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import time
import json
import os
import random
import math

import cifutils
import pdbutils
from na_data_utils import PDBDataset, make_batch_iter, create_collate_fn
from na_model_utils import (
    featurize, loss_nll, compute_canonical_base_pair_accuracy, 
    get_std_opt, ProteinMPNNDiffusion
)
from na_metric_manager import generate_metric_manager
from na_diffusion_utils import (
    apply_absorbing_mask, sample_mask_ratio, compute_diffusion_loss,
    get_na_mask_from_polymer_masks, get_designable_mask
)
from na_model_utils import ProteinMPNN  # For baseline comparison

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run `pip install wandb` for experiment tracking.")


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

class CosineAnnealingWarmupScheduler:
    """
    Cosine Annealing with Linear Warmup.
    
    This is the recommended LR schedule for discrete diffusion models (DPLM, MaskGIT).
    
    LR schedule:
    - Warmup phase (0 to warmup_steps): Linear increase from 0 to lr_base
    - Cosine phase (warmup_steps to total_steps): Cosine decay from lr_base to lr_min
    """
    def __init__(self, optimizer, lr_base, lr_min, warmup_steps, total_steps, last_step=-1):
        self.optimizer = optimizer
        self.lr_base = lr_base
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = last_step
        self._rate = 0
        
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        """Update parameters and learning rate."""
        self._step += 1
        rate = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        return rate
    
    def get_lr(self, step=None):
        """Compute learning rate for current step."""
        if step is None:
            step = self._step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.lr_base * (step / max(1, self.warmup_steps))
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            return self.lr_min + 0.5 * (self.lr_base - self.lr_min) * (1 + math.cos(math.pi * progress))


def get_optimizer_and_scheduler(model, params, current_step=0):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        params: Configuration dictionary
        current_step: Starting step (for resuming training)
    
    Returns:
        optimizer_wrapper: Object with step() method that updates LR
    """
    lr_scheduler_type = params.get("LR_SCHEDULER", "noam")
    
    if lr_scheduler_type == "cosine_warmup":
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get("LR_BASE", 1e-4),
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=base_optimizer,
            lr_base=params.get("LR_BASE", 1e-4),
            lr_min=params.get("LR_MIN", 1e-6),
            warmup_steps=params.get("LR_WARMUP_STEPS", 2000),
            total_steps=params.get("TOTAL_STEPS", 150000),
            last_step=current_step
        )
        return scheduler
    elif lr_scheduler_type == "constant_warmup":
        # Warmup then constant
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get("LR_BASE", 1e-4),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=base_optimizer,
            lr_base=params.get("LR_BASE", 1e-4),
            lr_min=params.get("LR_BASE", 1e-4),  # No decay
            warmup_steps=params.get("LR_WARMUP_STEPS", 2000),
            total_steps=params.get("TOTAL_STEPS", 150000),
            last_step=current_step
        )
        return scheduler
    else:  # "noam" (original)
        return get_std_opt(model.parameters(), params["HIDDEN_DIM"], current_step)


def compute_positionwise_cross_entropy(logits, target_seq):
    """Return per-position cross entropy from unnormalized logits."""
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, V),
        target_seq.reshape(-1),
        reduction='none'
    ).view(B, L)


def compute_irm_penalty(
    logits,
    target_seq,
    diffusion_mask,
    valid_mask,
    polymer_masks,
    min_tokens_per_env=1,
):
    """Compute IRM penalty across protein / DNA / RNA environments."""
    logits = logits.float()
    target_seq = target_seq.long()

    supervision_mask = diffusion_mask.float()
    if valid_mask is not None:
        supervision_mask = supervision_mask * valid_mask.float()

    penalty_terms = []
    active_envs = 0
    env_metrics = {}

    for env_name in ("protein", "dna", "rna"):
        env_mask = supervision_mask * polymer_masks[env_name].float()
        token_count = env_mask.sum()
        env_metrics[f"irm_{env_name}_tokens"] = float(token_count.detach().item())

        if token_count.detach().item() < float(min_tokens_per_env):
            continue

        scale = torch.tensor(1.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        env_loss = (
            compute_positionwise_cross_entropy(logits * scale, target_seq) * env_mask
        ).sum() / token_count.clamp(min=1.0)
        env_grad = torch.autograd.grad(env_loss, scale, create_graph=True)[0]
        penalty_terms.append(env_grad.pow(2))
        active_envs += 1

        env_metrics[f"irm_{env_name}_loss"] = float(env_loss.detach().item())
        env_metrics[f"irm_{env_name}_grad"] = float(env_grad.detach().item())

    if penalty_terms:
        irm_penalty = torch.stack(penalty_terms).sum()
    else:
        irm_penalty = logits.new_zeros(())

    env_metrics["irm_penalty"] = float(irm_penalty.detach().item())
    env_metrics["irm_active_envs"] = float(active_envs)
    return irm_penalty, env_metrics


def main():
    JSON = sys.argv[1]
    params = json.load(open(JSON))
    
    # Set defaults for diffusion-specific parameters
    diffusion_defaults = {
        "MASK_RATIO_SCHEDULE": "uniform",  # uniform, log_uniform, cosine, refinement_focus
        "MASK_RATIO_MIN": 0.0,
        "MASK_RATIO_MAX": 1.0,
        "LOSS_WEIGHT_STRATEGY": "balanced",  # uniform, balanced, refinement, snr
        "LOSS_WEIGHT_EPSILON": 0.1,
        "MASK_NA_ONLY": True,  # Only mask nucleic acid positions (legacy, use MASK_MODE)
        "MASK_MODE": "na_only",  # all | protein_only | na_only | interface
        "USE_SEQUENCE_CONTEXT": True,  # Use sequence info in decoder
        "DIFFUSION_LABEL_SMOOTHING": 0.1,
        "COMPARE_WITH_BASELINE": False,  # Compare with AR baseline each epoch
        "BASELINE_CHECKPOINT": None,  # Path to AR baseline checkpoint
        # LR Scheduler
        "LR_SCHEDULER": "noam",  # noam, cosine_warmup, constant_warmup
        "LR_BASE": 1e-4,
        "LR_MIN": 1e-6,
        "LR_WARMUP_STEPS": 2000,
        # Callback task
        "FULL_MASK_RATIO": 0.1,  # Probability of training on 100% mask
        # IRM regularization
        "USE_IRM_PENALTY": False,
        "IRM_PENALTY_WEIGHT": 0.0,
        "IRM_MIN_TOKENS_PER_ENV": 1,
        "RESET_TRAINING_STATE": False,
        "DATASET_ROOT": None,
        # Wandb
        "USE_WANDB": False,
        "WANDB_PROJECT": "naiad",
        "WANDB_ENTITY": None,
        "WANDB_RUN_NAME": None,
    }
    for key, value in diffusion_defaults.items():
        if key not in params:
            params[key] = value
    
    # Initialize wandb
    wandb_run = None
    if params.get("USE_WANDB") and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=params.get("WANDB_PROJECT", "naiad"),
            entity=params.get("WANDB_ENTITY"),
            name=params.get("WANDB_RUN_NAME"),
            config=params,
            resume="allow"
        )
        print(f"✓ Wandb initialized: {wandb_run.url}")

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    if params["BASE_FOLDER"][-1] != '/':
        params["BASE_FOLDER"] += '/'
    if not os.path.exists(params["BASE_FOLDER"]):
        os.makedirs(params["BASE_FOLDER"])

    logfile = params["BASE_FOLDER"] + 'log.txt'
    if not params["PREV_CHECKPOINT"]:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\tMaskRatio\tLossWeight\n')

    # Atom list configuration
    if params["ATOMS_TO_LOAD"] == "backbone":
        atom_list_to_save = [
            'N', 'CA', 'C', 'O',  # protein atoms
            'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
            "C2'", "O2'", "C1'"  # nucleic acid atoms
        ]
    elif params["ATOMS_TO_LOAD"] == "all":
        atom_list_to_save = [
            'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 
            'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 
            'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 
            'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
            'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
            "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 
            'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1'
        ]

    cif_parser = cifutils.CIFParser(
        skip_res=params["EXCLUDE_RES"], 
        randomize_nmr_model=params["RANDOMIZE_NMR_MODEL"]
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
        drop_protein_probability=params["DROP_PROTEIN_PROBABILITY"],
        na_only_as_uniform_ppm=params["NA_ONLY_AS_UNIFORM_PPM"],
        protein_interface_residue_mutation_probability=params["PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY"],
        mutate_base_pair_together=params["MUTATE_BASE_PAIR_TOGETHER"],
        mutate_entire_side_chain_interface_probability=params["MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY"],
        na_non_interface_as_uniform_ppm=params["NA_NON_INTERFACE_AS_UNIFORM_PPM"]
    )

    # Create diffusion model
    model = ProteinMPNNDiffusion(
        node_features=params["HIDDEN_DIM"],
        edge_features=params["HIDDEN_DIM"],
        hidden_dim=params["HIDDEN_DIM"],
        num_encoder_layers=params["NUM_ENCODER_LAYERS"],
        num_decoder_layers=params["NUM_DECODER_LAYERS"],
        k_neighbors=params["NUM_NEIGHBORS"],
        dropout=params["DROPOUT"],
        atom_dict=pdb_dataset.atom_dict,
        restype_to_int=pdb_dataset.restype_to_int,
        polytype_to_int=pdb_dataset.polytype_to_int,
        protein_augment_eps=params["PROTEIN_BACKBONE_NOISE"],
        dna_augment_eps=params["DNA_BACKBONE_NOISE"],
        rna_augment_eps=params["RNA_BACKBONE_NOISE"],
        na_ref_atom=params["NA_REF_ATOM"],
        include_pred_na_N=params["INCLUDE_PRED_NA_N"],
        use_sequence_context=params["USE_SEQUENCE_CONTEXT"],
        device=device,
        vocab=params["VOCAB_SIZE"],
        num_letters=params["NUM_LETTERS"]
    )
    model.to(device)

    # Load AR baseline for comparison (frozen, eval mode)
    baseline_model = None
    if params["COMPARE_WITH_BASELINE"] and params["BASELINE_CHECKPOINT"]:
        print(f"Loading AR baseline for comparison from: {params['BASELINE_CHECKPOINT']}")
        baseline_model = ProteinMPNN(
            node_features=params["HIDDEN_DIM"],
            edge_features=params["HIDDEN_DIM"],
            hidden_dim=params["HIDDEN_DIM"],
            num_encoder_layers=params["NUM_ENCODER_LAYERS"],
            num_decoder_layers=params["NUM_DECODER_LAYERS"],
            k_neighbors=params["NUM_NEIGHBORS"],
            dropout=0.0,  # No dropout for evaluation
            atom_dict=pdb_dataset.atom_dict,
            restype_to_int=pdb_dataset.restype_to_int,
            polytype_to_int=pdb_dataset.polytype_to_int,
            protein_augment_eps=0.0,  # No noise for evaluation
            dna_augment_eps=0.0,
            rna_augment_eps=0.0,
            decode_protein_first=0,
            na_ref_atom=params["NA_REF_ATOM"],
            include_pred_na_N=params["INCLUDE_PRED_NA_N"],
            device=device,
            vocab=params["VOCAB_SIZE"],
            num_letters=params["NUM_LETTERS"]
        )
        baseline_model.to(device)
        try:
            baseline_checkpoint = torch.load(
                params["BASELINE_CHECKPOINT"],
                map_location=device,
                weights_only=False
            )
            baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
            baseline_model.eval()
            for param in baseline_model.parameters():
                param.requires_grad = False
            print("✓ AR baseline loaded successfully (frozen for comparison)")
        except Exception as e:
            print(f"⚠ Could not load baseline checkpoint: {e}")
            baseline_model = None

    # Load checkpoint if provided
    checkpoint = None
    reset_training = False
    if params["PREV_CHECKPOINT"]:
        try:
            # Use map_location for CPU compatibility
            checkpoint = torch.load(
                params["PREV_CHECKPOINT"], 
                map_location=device,
                weights_only=False
            )
            # Check if we should reset step/epoch (e.g., when starting from baseline
            # or warm-starting a new public run from a released checkpoint).
            reset_training = (
                params.get("RESET_TRAINING_STATE", False)
                or "original" in params["PREV_CHECKPOINT"]
                or "baseline" in params["PREV_CHECKPOINT"]
            )
            
            if reset_training:
                print(f"✓ Loading weights from {params['PREV_CHECKPOINT']}, but RESETTING step and epoch to 0")
                total_step = 0
                epoch = 0
                save_step = 0
            else:
                total_step = checkpoint.get('step', 0)
                save_step = checkpoint.get('save_step', 0)
                epoch = checkpoint.get('epoch', 0)
                print(f"✓ Loaded checkpoint from step {total_step}, epoch {epoch}")
                
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("  Starting fresh training...")
            total_step = 0
            epoch = 0
            save_step = 0
            params["PREV_CHECKPOINT"] = []
    else:
        total_step = 0
        epoch = 0
        save_step = 0

    # Initialize optimizer with scheduler
    optimizer = get_optimizer_and_scheduler(model, params, total_step)
    
    # Load optimizer state if resuming
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint and not reset_training:
        try:
            if hasattr(optimizer, 'optimizer'):
                optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"⚠ Could not load optimizer state: {e}. Starting with fresh optimizer.")
    elif checkpoint is not None and reset_training:
        print("✓ RESET_TRAINING_STATE is enabled; using a fresh optimizer state")
    
    # Print LR schedule info
    lr_scheduler_type = params.get("LR_SCHEDULER", "noam")
    print(f"\nLearning Rate Schedule: {lr_scheduler_type}")
    if lr_scheduler_type == "cosine_warmup":
        print(f"  - Peak LR: {params.get('LR_BASE', 1e-4)}")
        print(f"  - Min LR: {params.get('LR_MIN', 1e-6)}")
        print(f"  - Warmup steps: {params.get('LR_WARMUP_STEPS', 2000)}")
        print(f"  - Total steps: {params.get('TOTAL_STEPS', 150000)}")

    df_valid = pd.read_csv(params["DF_PATH_VALID"])
    df_train = pd.read_csv(params["DF_PATH_TRAIN"])

    dataset_root = params.get("DATASET_ROOT") or os.environ.get("NAIAD_DATASET_ROOT")
    if dataset_root:
        dataset_root = os.path.abspath(dataset_root)

        def _resolve_structure_paths(df):
            if "structure_path" not in df.columns:
                return df
            df = df.copy()
            mask = df["structure_path"].notna() & ~df["structure_path"].map(os.path.isabs)
            df.loc[mask, "structure_path"] = df.loc[mask, "structure_path"].map(
                lambda p: os.path.normpath(os.path.join(dataset_root, str(p)))
            )
            return df

        df_valid = _resolve_structure_paths(df_valid)
        df_train = _resolve_structure_paths(df_train)

    df_valid["date"] = pd.to_datetime(df_valid["date"], format="%Y-%m-%d")
    df_train["date"] = pd.to_datetime(df_train["date"], format="%Y-%m-%d")
    date_cutoff = pd.to_datetime(params["DATE_CUTOFF"], format="%Y-%m-%d")

    metric_manager = generate_metric_manager(
        pdb_dataset.restype_to_int, 
        metrics_to_compute=params["METRICS_TO_COMPUTE"]
    )

    tokens_with_no_loss = torch.tensor([
        pdb_dataset.restype_to_int["UNK"], 
        pdb_dataset.restype_to_int["DX"], 
        pdb_dataset.restype_to_int["RX"], 
        pdb_dataset.restype_to_int["MAS"], 
        pdb_dataset.restype_to_int["PAD"]
    ], device=device)

    # Mask token index for diffusion
    mask_token_idx = pdb_dataset.restype_to_int["MAS"]

    # NA restype mask for label smoothing
    na_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
    na_restype_mask[pdb_dataset.dna_restype_ints] = 1
    na_restype_mask[pdb_dataset.rna_restype_ints] = 1

    # Masks used for loss function
    protein_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
    protein_restype_mask[pdb_dataset.protein_restype_ints] = 1

    dna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
    dna_restype_mask[pdb_dataset.dna_restype_ints] = 1
        
    rna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
    rna_restype_mask[pdb_dataset.rna_restype_ints] = 1

    polymer_restype_masks = {
        "protein": protein_restype_mask,
        "dna": dna_restype_mask,
        "rna": rna_restype_mask
    }

    polymer_restype_nums = {
        "protein": len(pdb_dataset.protein_restype_ints),
        "dna": len(pdb_dataset.dna_restype_ints),
        "rna": len(pdb_dataset.rna_restype_ints)
    }

    print(f"Starting diffusion training with:")
    print(f"  - Mask ratio schedule: {params['MASK_RATIO_SCHEDULE']}")
    print(f"  - Mask ratio range: [{params['MASK_RATIO_MIN']}, {params['MASK_RATIO_MAX']}]")
    print(f"  - Loss weight strategy: {params['LOSS_WEIGHT_STRATEGY']}")
    mask_mode = params.get("MASK_MODE", "na_only" if params.get("MASK_NA_ONLY", True) else "all")
    print(f"  - Mask mode: {mask_mode} (designable positions)")

    # Training loop
    best_valid_acc = -1.0
    for e in range(100000):
        metric_manager.zero_metrics()
        
        # Track diffusion-specific metrics
        epoch_mask_ratios = []
        epoch_loss_weights = []
        epoch_irm_penalties = []
        epoch_irm_active_envs = []

        batch_iter_valid = make_batch_iter(
            df=df_valid, 
            batch_tokens=params["BATCH_TOKENS"], 
            length_cutoff=params["MIN_PROTEIN_LENGTH_CUTOFF"],
            date_cutoff=date_cutoff, 
            crop_large_structures=params["CROP_LARGE_STRUCTURES"],
            max_number_of_pdbs=params["MAX_NUMBER_OF_PDBS_VALID"]
        )
        
        batch_iter_train = make_batch_iter(
            df=df_train, 
            batch_tokens=params["BATCH_TOKENS"], 
            length_cutoff=params["MIN_PROTEIN_LENGTH_CUTOFF"],
            date_cutoff=date_cutoff, 
            crop_large_structures=params["CROP_LARGE_STRUCTURES"],
            max_number_of_pdbs=params["MAX_NUMBER_OF_PDBS_TRAIN"]
        )

        valid_sampler = torch.utils.data.sampler.BatchSampler(
            batch_iter_valid, batch_size=1, drop_last=False
        )
        train_sampler = torch.utils.data.sampler.BatchSampler(
            batch_iter_train, batch_size=1, drop_last=False
        )

        # Create optimized collate function that does tensor creation in workers
        collate_fn = create_collate_fn(
            pdb_dataset.polytype_to_int,
            pdb_dataset.restype_to_int,
            pdb_dataset.atom_dict
        )
        
        loader_kwargs = {
            "num_workers": params["NUM_WORKERS"],
            "pin_memory": True,
            "collate_fn": collate_fn,
        }
        if params["NUM_WORKERS"] > 0:
            loader_kwargs.update({
                "persistent_workers": True,  # Keep workers alive between epochs
                "prefetch_factor": 4,        # Prefetch 4 batches per worker
            })

        valid_loader = torch.utils.data.DataLoader(
            pdb_dataset, sampler=valid_sampler, **loader_kwargs
        )
        train_loader = torch.utils.data.DataLoader(
            pdb_dataset, sampler=train_sampler, **loader_kwargs
        )

        # ==================== TRAINING ====================
        model.train()
        e = epoch + e
        t0 = time.time()
        
        for ix, feature_dict in enumerate(train_loader):
            # Skip invalid batches (collate_fn returns None for empty/invalid batches)
            if feature_dict is None:
                continue
            
            optimizer.zero_grad()
            
            # Move tensors to GPU (they come from collate_fn on CPU with pin_memory)
            for key in feature_dict:
                if isinstance(feature_dict[key], torch.Tensor):
                    feature_dict[key] = feature_dict[key].to(device, non_blocking=True)
            
            S = feature_dict["S"]
            mask = feature_dict["mask"]
            B, L = S.shape
            if L == 0:  # Skip degenerate empty batches
                continue
            
            # Compute valid mask (exclude unknown tokens)
            S_mask = 1 - (torch.any(
                S[:,:,None] == tokens_with_no_loss[None,None,:], dim=-1
            )).long()
            mask_for_loss = mask * S_mask
            feature_dict["mask_for_loss"] = mask_for_loss

            polymer_masks = {
                "protein": feature_dict["protein_mask"], 
                "dna": feature_dict["dna_mask"], 
                "rna": feature_dict["rna_mask"]
            }
            position_loss_weights = (
                params.get("PROTEIN_LOSS_WEIGHT", 1.0) * polymer_masks["protein"].float()
                + params.get("DNA_LOSS_WEIGHT", 1.0) * polymer_masks["dna"].float()
                + params.get("RNA_LOSS_WEIGHT", 1.0) * polymer_masks["rna"].float()
            )

            # ========== DIFFUSION: Sample mask ratio and apply masking ==========
            
            # Sample mask ratio for this batch
            # CALLBACK TASK: Occasionally train on 100% masked sequences to prevent
            # forgetting the inverse folding task (cold start).
            # This ensures the model retains the ability to do de novo design.
            full_mask_prob = params.get("FULL_MASK_RATIO", 0.1)
            if random.random() < full_mask_prob:
                mask_ratio = torch.ones(B, device=device)
            else:
                mask_ratio = sample_mask_ratio(
                    batch_size=B,
                    schedule=params["MASK_RATIO_SCHEDULE"],
                    min_ratio=params["MASK_RATIO_MIN"],
                    max_ratio=params["MASK_RATIO_MAX"],
                    device=device
                )
            
            # Determine which positions to mask based on MASK_MODE
            mask_mode = params.get("MASK_MODE", "na_only" if params.get("MASK_NA_ONLY", True) else "all")
            designable_mask = get_designable_mask(
                feature_dict, mask_mode
            )
            
            # Apply absorbing state masking
            masked_S, diffusion_mask = apply_absorbing_mask(
                seq_labels=S,
                mask_ratio=mask_ratio,
                mask_token_idx=mask_token_idx,
                valid_mask=mask_for_loss,
                na_only_mask=designable_mask  # Now supports protein + NA
            )
            
            # Update feature dict with masked sequence
            feature_dict["S"] = masked_S
            feature_dict["S_original"] = S  # Keep original for loss computation
            feature_dict["diffusion_mask"] = diffusion_mask
            
            # Track metrics
            epoch_mask_ratios.append(mask_ratio.mean().item())

            if params["MIXED_PRECISION"]:
                with torch.cuda.amp.autocast():
                    log_probs, probs, h_V = model(feature_dict, return_embeddings=True)
                    logits = model.W_out(h_V).float()
                    
                    # Compute diffusion loss with DPLM-style weighting
                    diffusion_loss, diff_metrics = compute_diffusion_loss(
                        logits=logits,
                        target_seq=S,  # Original sequence
                        diffusion_mask=diffusion_mask,
                        mask_ratio=mask_ratio,
                        valid_mask=mask_for_loss,
                        position_weights=position_loss_weights,
                        loss_weight_strategy=params["LOSS_WEIGHT_STRATEGY"],
                        label_smoothing=params["DIFFUSION_LABEL_SMOOTHING"],
                        na_restype_mask=na_restype_mask,
                    )

                    irm_penalty = logits.new_zeros(())
                    irm_metrics = {"irm_penalty": 0.0, "irm_active_envs": 0.0}
                    if params["USE_IRM_PENALTY"] and params["IRM_PENALTY_WEIGHT"] > 0:
                        irm_penalty, irm_metrics = compute_irm_penalty(
                            logits=logits,
                            target_seq=S,
                            diffusion_mask=diffusion_mask,
                            valid_mask=mask_for_loss,
                            polymer_masks=polymer_masks,
                            min_tokens_per_env=params["IRM_MIN_TOKENS_PER_ENV"],
                        )
                    total_loss = diffusion_loss + params["IRM_PENALTY_WEIGHT"] * irm_penalty
                    
                epoch_loss_weights.append(diff_metrics["loss_weight"])
                epoch_irm_penalties.append(irm_metrics["irm_penalty"])
                epoch_irm_active_envs.append(irm_metrics["irm_active_envs"])

                scaler.scale(total_loss).backward()

                if params["GRADIENT_NORM"] > 0.0:
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["GRADIENT_NORM"]
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs, probs, h_V = model(feature_dict, return_embeddings=True)
                logits = model.W_out(h_V).float()
                
                diffusion_loss, diff_metrics = compute_diffusion_loss(
                    logits=logits,
                    target_seq=S,
                    diffusion_mask=diffusion_mask,
                    mask_ratio=mask_ratio,
                    valid_mask=mask_for_loss,
                    position_weights=position_loss_weights,
                    loss_weight_strategy=params["LOSS_WEIGHT_STRATEGY"],
                    label_smoothing=params["DIFFUSION_LABEL_SMOOTHING"],
                    na_restype_mask=na_restype_mask,
                )

                irm_penalty = logits.new_zeros(())
                irm_metrics = {"irm_penalty": 0.0, "irm_active_envs": 0.0}
                if params["USE_IRM_PENALTY"] and params["IRM_PENALTY_WEIGHT"] > 0:
                    irm_penalty, irm_metrics = compute_irm_penalty(
                        logits=logits,
                        target_seq=S,
                        diffusion_mask=diffusion_mask,
                        valid_mask=mask_for_loss,
                        polymer_masks=polymer_masks,
                        min_tokens_per_env=params["IRM_MIN_TOKENS_PER_ENV"],
                    )
                total_loss = diffusion_loss + params["IRM_PENALTY_WEIGHT"] * irm_penalty
                
                epoch_loss_weights.append(diff_metrics["loss_weight"])
                epoch_irm_penalties.append(irm_metrics["irm_penalty"])
                epoch_irm_active_envs.append(irm_metrics["irm_active_envs"])
                
                total_loss.backward()
                
                if params["GRADIENT_NORM"] > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["GRADIENT_NORM"]
                    )
                
                optimizer.step()

            # Compute metrics on original sequence for logging
            # Restore original S for metric computation
            feature_dict["S"] = S
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
            canonical_base_pair_accuracy = compute_canonical_base_pair_accuracy(
                log_probs, 
                feature_dict["canonical_base_pair_mask"], 
                feature_dict["canonical_base_pair_index"], 
                pdb_dataset
            )
            
            S_pred = torch.argmax(log_probs, -1)

            if params["METRICS_TO_COMPUTE"] == "all":
                interface_masks = {
                    "interface": feature_dict["interface_mask"],
                    "nonInterface": 1 - feature_dict["interface_mask"]
                }
            else:
                interface_masks = {}

            metric_manager.accumulate(
                loss, true_false, canonical_base_pair_accuracy, 
                feature_dict["canonical_base_pair_mask"], 
                S, S_pred, "train", mask_for_loss, 
                polymer_masks, interface_masks
            )

            total_step += 1

        # ==================== VALIDATION ====================
        model.eval()
        t1 = time.time()
        
        # Track baseline comparison metrics
        baseline_total_correct = 0
        baseline_total_count = 0
        diffusion_total_correct = 0
        diffusion_total_count = 0
        
        with torch.no_grad():
            for ix, feature_dict in enumerate(valid_loader):
                # Skip invalid batches
                if feature_dict is None:
                    continue
                
                # Move tensors to GPU
                for key in feature_dict:
                    if isinstance(feature_dict[key], torch.Tensor):
                        feature_dict[key] = feature_dict[key].to(device, non_blocking=True)
                    
                S = feature_dict["S"]
                mask = feature_dict["mask"]
                S_mask = 1 - (torch.any(
                    S[:,:,None] == tokens_with_no_loss[None,None,:], dim=-1
                )).long()
                mask_for_loss = mask * S_mask
                feature_dict["mask_for_loss"] = mask_for_loss

                polymer_masks = {
                    "protein": feature_dict["protein_mask"], 
                    "dna": feature_dict["dna_mask"], 
                    "rna": feature_dict["rna_mask"]
                }

                # For validation, test at multiple mask ratios
                # Here we use 50% mask ratio as a standard benchmark
                B, L = S.shape
                val_mask_ratio = torch.full((B,), 0.5, device=device)
                
                # Use same mask mode as training
                mask_mode = params.get("MASK_MODE", "na_only" if params.get("MASK_NA_ONLY", True) else "all")
                designable_mask = get_designable_mask(feature_dict, mask_mode)
                
                masked_S, diffusion_mask = apply_absorbing_mask(
                    seq_labels=S,
                    mask_ratio=val_mask_ratio,
                    mask_token_idx=mask_token_idx,
                    valid_mask=mask_for_loss,
                    na_only_mask=designable_mask
                )
                
                feature_dict["S"] = masked_S
                
                log_probs, probs = model(feature_dict)
                
                # Restore original S for metrics
                feature_dict["S"] = S
                
                validation_score_mask = mask_for_loss * diffusion_mask.long()
                if validation_score_mask.sum() == 0:
                    continue
                loss, loss_av, true_false = loss_nll(S, log_probs, validation_score_mask)
                canonical_base_pair_accuracy = compute_canonical_base_pair_accuracy(
                    log_probs, 
                    feature_dict["canonical_base_pair_mask"], 
                    feature_dict["canonical_base_pair_index"],
                    pdb_dataset
                )
                S_pred = torch.argmax(log_probs, -1)

                if params["METRICS_TO_COMPUTE"] == "all":
                    interface_masks = {
                        "interface": feature_dict["interface_mask"],
                        "nonInterface": 1 - feature_dict["interface_mask"]
                    }
                else:
                    interface_masks = {}

                metric_manager.accumulate(
                    loss, true_false, canonical_base_pair_accuracy, 
                    feature_dict["canonical_base_pair_mask"], 
                    S, S_pred, "valid", validation_score_mask, 
                    polymer_masks, interface_masks
                )
                
                # ========== BASELINE COMPARISON ==========
                if baseline_model is not None:
                    # Get comparison mask based on mask_mode
                    # diffusion_mask indicates which positions were masked (1 = masked, 0 = not masked)
                    mask_mode = params.get("MASK_MODE", "na_only" if params.get("MASK_NA_ONLY", True) else "all")
                    
                    if mask_mode == "all":
                        # Compare on all masked positions
                        comparison_mask = diffusion_mask.float()
                    elif mask_mode == "protein_only":
                        comparison_mask = feature_dict["protein_mask"].float() * diffusion_mask.float()
                    else:  # na_only or interface
                        comparison_mask = (feature_dict["dna_mask"] + feature_dict["rna_mask"]) * diffusion_mask.float()
                    
                    # FAIR COMPARISON: Both models receive the SAME masked input
                    # Save original S and use masked_S for baseline too
                    feature_dict["S"] = masked_S
                    baseline_log_probs, _ = baseline_model(feature_dict)
                    baseline_S_pred = torch.argmax(baseline_log_probs, -1)
                    feature_dict["S"] = S  # Restore for next iteration
                    
                    # Calculate accuracy on MASKED positions
                    diffusion_correct = ((S_pred == S) * comparison_mask).sum().item()
                    baseline_correct = ((baseline_S_pred == S) * comparison_mask).sum().item()
                    total_masked = comparison_mask.sum().item()
                    
                    diffusion_total_correct += diffusion_correct
                    baseline_total_correct += baseline_correct
                    diffusion_total_count += total_masked
                    baseline_total_count += total_masked

        t2 = time.time()
        metric_manager.compute_metrics()

        train_dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=3)
        valid_dt = np.format_float_positional(np.float32(t2-t1), unique=False, precision=3)
        
        avg_mask_ratio = np.mean(epoch_mask_ratios) if epoch_mask_ratios else 0
        avg_loss_weight = np.mean(epoch_loss_weights) if epoch_loss_weights else 0
        avg_irm_penalty = np.mean(epoch_irm_penalties) if epoch_irm_penalties else 0
        avg_irm_active_envs = np.mean(epoch_irm_active_envs) if epoch_irm_active_envs else 0

        output_string = metric_manager.create_print_string(
            e, total_step, train_dt, valid_dt
        )
        output_string += f" | MaskRatio: {avg_mask_ratio:.3f} | LossWeight: {avg_loss_weight:.3f}"
        if params["USE_IRM_PENALTY"] and params["IRM_PENALTY_WEIGHT"] > 0:
            output_string += (
                f" | IRM: {avg_irm_penalty:.4f}"
                f" | IRMEnvs: {avg_irm_active_envs:.2f}"
            )
        
        # Add baseline comparison results
        if baseline_model is not None and baseline_total_count > 0:
            baseline_acc = baseline_total_correct / baseline_total_count
            diffusion_acc = diffusion_total_correct / diffusion_total_count
            improvement = diffusion_acc - baseline_acc
            output_string += f" | NA-Acc(Diff): {diffusion_acc:.4f} | NA-Acc(Base): {baseline_acc:.4f} | Δ: {improvement:+.4f}"

        with open(logfile, 'a') as f:
            f.write(output_string + "\n")
        print(output_string)
        
        # Get current learning rate
        current_lr = optimizer._rate if hasattr(optimizer, '_rate') else optimizer.param_groups[0]['lr']
        
        # Helper function to get metric from MetricManager
        def get_metric(mask_name, metric_name):
            """Get metric value from MetricManager's metrics array."""
            try:
                row = metric_manager.mask_to_row.get(mask_name)
                col = metric_manager.metric_to_col.get(metric_name)
                if row is not None and col is not None:
                    val = metric_manager.metrics[row, col]
                    return float(val) if not np.isnan(val) else 0.0
                return 0.0
            except Exception:
                return 0.0
        
        # Log to wandb
        if wandb_run is not None:
            wandb_log = {
                "epoch": e,
                "step": total_step,
                "train/loss": get_metric("train", "loss"),
                "train/accuracy": get_metric("train", "accuracy"),
                "train/dna_loss": get_metric("train_dna", "loss"),
                "train/dna_accuracy": get_metric("train_dna", "accuracy"),
                "train/rna_loss": get_metric("train_rna", "loss"),
                "train/rna_accuracy": get_metric("train_rna", "accuracy"),
                "valid/loss": get_metric("valid", "loss"),
                "valid/accuracy": get_metric("valid", "accuracy"),
                "valid/dna_loss": get_metric("valid_dna", "loss"),
                "valid/dna_accuracy": get_metric("valid_dna", "accuracy"),
                "valid/rna_loss": get_metric("valid_rna", "loss"),
                "valid/rna_accuracy": get_metric("valid_rna", "accuracy"),
                "diffusion/mask_ratio": avg_mask_ratio,
                "diffusion/loss_weight": avg_loss_weight,
                "diffusion/irm_penalty": avg_irm_penalty,
                "diffusion/irm_active_envs": avg_irm_active_envs,
                "learning_rate": current_lr,
            }
            if baseline_model is not None and baseline_total_count > 0:
                wandb_log["comparison/diffusion_na_acc"] = diffusion_acc
                wandb_log["comparison/baseline_na_acc"] = baseline_acc
                wandb_log["comparison/delta"] = improvement
            wandb.log(wandb_log, step=total_step)
        
        # Get optimizer state dict (handle different scheduler types)
        if hasattr(optimizer, 'optimizer'):
            opt_state_dict = optimizer.optimizer.state_dict()
        else:
            opt_state_dict = optimizer.state_dict()
        
        # Save checkpoint
        checkpoint = {
            'epoch': e+1,
            'step': total_step,
            'save_step': save_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt_state_dict,
            'diffusion_params': {
                'mask_ratio_schedule': params['MASK_RATIO_SCHEDULE'],
                'loss_weight_strategy': params['LOSS_WEIGHT_STRATEGY'],
                'use_irm_penalty': params['USE_IRM_PENALTY'],
                'irm_penalty_weight': params['IRM_PENALTY_WEIGHT'],
                'irm_min_tokens_per_env': params['IRM_MIN_TOKENS_PER_ENV'],
            }
        }
        torch.save(checkpoint, params["BASE_FOLDER"]+'last.pt')
        
        # Save best model based on a balanced validation accuracy
        # We average Protein and Nucleic Acid (DNA+RNA) accuracy to ensure 
        # the model performs well on both, avoiding bias towards protein 
        # which usually has more residues.
        val_acc_prot = get_metric("valid_protein", "accuracy")
        val_acc_dna = get_metric("valid_dna", "accuracy")
        val_acc_rna = get_metric("valid_rna", "accuracy")
        
        # Calculate NA accuracy (average of DNA and RNA if both present)
        na_acc_list = [a for a in [val_acc_dna, val_acc_rna] if a > 0]
        val_acc_na = sum(na_acc_list) / len(na_acc_list) if na_acc_list else 0.0
        
        # Combined balanced accuracy
        if val_acc_prot > 0 and val_acc_na > 0:
            current_valid_acc = (val_acc_prot + val_acc_na) / 2
        else:
            # Fallback to whatever is available (or total accuracy)
            current_valid_acc = get_metric("valid", "accuracy")
            
        if current_valid_acc > best_valid_acc:
            best_valid_acc = current_valid_acc
            torch.save(checkpoint, params["BASE_FOLDER"]+'best.pt')
            print(f"  ✓ New best balanced validation accuracy: {best_valid_acc:.4f} (saved best.pt)")
        
        if total_step > save_step + params["SAVE_EVERY_N_STEPS"]:
            save_step += params["SAVE_EVERY_N_STEPS"]
            torch.save({
                'epoch': e+1,
                'step': total_step,
                'save_step': save_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt_state_dict,
                'diffusion_params': {
                    'mask_ratio_schedule': params['MASK_RATIO_SCHEDULE'],
                    'loss_weight_strategy': params['LOSS_WEIGHT_STRATEGY'],
                    'use_irm_penalty': params['USE_IRM_PENALTY'],
                    'irm_penalty_weight': params['IRM_PENALTY_WEIGHT'],
                    'irm_min_tokens_per_env': params['IRM_MIN_TOKENS_PER_ENV'],
                }
            }, params["BASE_FOLDER"]+f's_{total_step}.pt')
            
        if total_step > params["TOTAL_STEPS"]:
            break
    
    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
