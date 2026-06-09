"""
Absorbing State Diffusion utilities for NAIAD.

Implements:
- Forward diffusion process (masking)
- DPLM-style loss weighting based on mask ratio
- Noise schedules (linear, cosine, log-uniform)
- Iterative denoising inference

Inspired by:
- ProRefiner: Iterative refinement for protein design
- DPLM: Discrete Diffusion Protein Language Model (ByteDance)
- MaskGIT / Muse: Masked generative transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Literal


# ============================================================================
# Forward Diffusion Process (Masking)
# ============================================================================

def apply_absorbing_mask(
    seq_labels: torch.Tensor,
    mask_ratio: Union[float, torch.Tensor],
    mask_token_idx: int,
    valid_mask: Optional[torch.Tensor] = None,
    na_only_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply absorbing state masking to sequences for diffusion training.
    
    In absorbing state diffusion, tokens transition to [MASK] and stay there.
    This function samples which positions to mask based on mask_ratio.
    
    Args:
        seq_labels: [Batch, Length] Original sequence token indices
        mask_ratio: float or [Batch] tensor, proportion of tokens to mask
        mask_token_idx: Index of the [MASK] token in vocabulary
        valid_mask: [Batch, Length] Binary mask for valid positions (1=valid)
        na_only_mask: [Batch, Length] Binary mask for NA positions to mask
                      (if provided, only masks nucleic acid positions)
    
    Returns:
        masked_seq: [Batch, Length] Sequence with some positions masked
        diffusion_mask: [Batch, Length] Boolean mask (True = masked for loss)
    """
    device = seq_labels.device
    B, L = seq_labels.shape
    
    # Ensure mask_ratio is a tensor
    if isinstance(mask_ratio, float):
        mask_ratio = torch.full((B,), mask_ratio, device=device)
    elif mask_ratio.dim() == 0:
        mask_ratio = mask_ratio.unsqueeze(0).expand(B)
    
    # Generate random values for each position
    rand_tensor = torch.rand(B, L, device=device)
    
    # Create mask based on mask_ratio (per-sample)
    # mask_ratio[:, None] broadcasts to [B, L]
    diffusion_mask = rand_tensor < mask_ratio[:, None]
    
    # Apply valid_mask if provided (only mask valid positions)
    if valid_mask is not None:
        diffusion_mask = diffusion_mask & (valid_mask > 0)
    
    # Apply na_only_mask if provided (only mask NA positions)
    if na_only_mask is not None:
        diffusion_mask = diffusion_mask & (na_only_mask > 0)
    
    # Create masked sequence
    masked_seq = seq_labels.clone()
    masked_seq[diffusion_mask] = mask_token_idx
    
    return masked_seq, diffusion_mask


def sample_mask_ratio(
    batch_size: int,
    schedule: Literal["uniform", "log_uniform", "cosine", "refinement_focus"] = "uniform",
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample mask ratios for a batch based on different schedules.
    
    Args:
        batch_size: Number of samples in batch
        schedule: Distribution to sample from:
            - "uniform": U[min_ratio, max_ratio]
            - "log_uniform": Biased toward lower ratios (more refinement training)
            - "cosine": Cosine schedule (smoother transitions)
            - "refinement_focus": Heavily biased toward low mask ratios
        min_ratio: Minimum mask ratio
        max_ratio: Maximum mask ratio
        device: Device for tensor
    
    Returns:
        mask_ratios: [batch_size] tensor of mask ratios
    """
    if device is None:
        device = torch.device("cpu")
    
    if schedule == "uniform":
        mask_ratios = torch.rand(batch_size, device=device) * (max_ratio - min_ratio) + min_ratio
        
    elif schedule == "log_uniform":
        # Log-uniform distribution biases toward lower values
        # Sample u ~ U[0,1], then r = min_ratio * (max_ratio/min_ratio)^u
        eps = 1e-6
        min_r = max(min_ratio, eps)
        u = torch.rand(batch_size, device=device)
        mask_ratios = min_r * ((max_ratio / min_r) ** u)
        
    elif schedule == "cosine":
        # Cosine schedule: more samples near 0 and 1
        u = torch.rand(batch_size, device=device)
        # Map through cosine to get more samples near extremes
        mask_ratios = 0.5 * (1 - torch.cos(np.pi * u)) * (max_ratio - min_ratio) + min_ratio
        
    elif schedule == "refinement_focus":
        # Beta distribution biased toward low mask ratios
        # More training on refinement (low mask) phase
        alpha, beta = 1.0, 3.0  # Biased toward 0
        u = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        mask_ratios = u * (max_ratio - min_ratio) + min_ratio
        
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return mask_ratios


# ============================================================================
# DPLM-Style Loss Weighting
# ============================================================================

def compute_loss_weight(
    mask_ratio: Union[float, torch.Tensor],
    strategy: Literal["uniform", "balanced", "refinement", "snr"] = "balanced",
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Compute loss weight based on mask ratio (DPLM-style gradient scaling).
    
    Different mask ratios require different gradient magnitudes:
    - High mask ratio: Learn global patterns (many masked tokens)
    - Low mask ratio: Learn local refinement (few masked tokens)
    
    Args:
        mask_ratio: float or tensor of mask ratios
        strategy: Weighting strategy:
            - "uniform": All mask ratios weighted equally (λ=1)
            - "balanced": Balance gradient flow across ratios (λ=1/(r+ε))
            - "refinement": Strongly emphasize low ratios (λ=1/r²)
            - "snr": Signal-to-noise ratio inspired (λ=(1-r)/r)
        epsilon: Small constant for numerical stability
    
    Returns:
        loss_weight: Scalar or tensor of loss weights
    """
    if isinstance(mask_ratio, float):
        mask_ratio = torch.tensor(mask_ratio)
    
    if strategy == "uniform":
        return torch.ones_like(mask_ratio)
    
    elif strategy == "balanced":
        # DPLM-style: balance gradients across noise levels
        # Low mask ratio -> high weight (focus on refinement)
        return 1.0 / (mask_ratio + epsilon)
    
    elif strategy == "refinement":
        # Strongly emphasize refinement phase
        return 1.0 / ((mask_ratio + epsilon) ** 2)
    
    elif strategy == "snr":
        # Signal-to-noise ratio inspired
        # Higher weight when signal is strong (low noise/mask)
        return (1.0 - mask_ratio + epsilon) / (mask_ratio + epsilon)
    
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")


def compute_diffusion_loss(
    logits: torch.Tensor,
    target_seq: torch.Tensor,
    diffusion_mask: torch.Tensor,
    mask_ratio: Union[float, torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
    position_weights: Optional[torch.Tensor] = None,
    loss_weight_strategy: str = "balanced",
    loss_weight_epsilon: float = 0.1,
    label_smoothing: float = 0.0,
    na_restype_mask: Optional[torch.Tensor] = None,
    polymer_masks: Optional[dict] = None,
    polymer_restype_masks: Optional[dict] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute diffusion loss with DPLM-style weighting.
    
    Only computes loss on masked positions, with weight based on mask ratio.
    
    Args:
        logits: [B, L, V] Model output logits
        target_seq: [B, L] Ground truth sequence
        diffusion_mask: [B, L] Boolean mask (True = masked position)
        mask_ratio: Mask ratio used for this batch
        valid_mask: [B, L] Valid sequence positions
        position_weights: [B, L] Optional per-position loss multipliers
        loss_weight_strategy: Strategy for loss weighting
        loss_weight_epsilon: Small constant for loss weighting stability
        label_smoothing: Label smoothing factor
        na_restype_mask: [V] mask for valid NA tokens (for label smoothing)
        polymer_masks: Optional position masks for protein, DNA, and RNA tokens
        polymer_restype_masks: Optional class masks for protein, DNA, and RNA
    
    Returns:
        loss: Scalar loss value
        metrics: Dictionary with additional metrics
    """
    B, L, V = logits.shape
    device = logits.device
    
    # Compute per-position cross entropy loss
    if (
        label_smoothing > 0 and
        polymer_masks is not None and
        polymer_restype_masks is not None
    ):
        target_onehot = F.one_hot(target_seq, V).float()
        smoothed_target = target_onehot * (1 - label_smoothing)

        smoothing_distribution = torch.zeros_like(logits)
        for polymer_name in ("protein", "dna", "rna"):
            position_mask = polymer_masks[polymer_name].to(
                device=device, dtype=logits.dtype
            )
            restype_mask = polymer_restype_masks[polymer_name].to(
                device=device, dtype=logits.dtype
            )
            num_classes = restype_mask.sum().clamp(min=1.0)
            smoothing_distribution += (
                position_mask[:, :, None] *
                restype_mask[None, None, :] /
                num_classes
            )

        target_probs = smoothed_target + label_smoothing * smoothing_distribution
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(target_probs * log_probs).sum(dim=-1)  # [B, L]
    elif label_smoothing > 0 and na_restype_mask is not None:
        # Backward-compatible NA-only label smoothing.
        target_onehot = F.one_hot(target_seq, V).float()
        num_na_classes = na_restype_mask.sum()
        smooth_value = label_smoothing / num_na_classes
        target_onehot = target_onehot * (1 - label_smoothing)
        target_onehot += smooth_value * na_restype_mask[None, None, :]
        
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(target_onehot * log_probs).sum(dim=-1)  # [B, L]
    else:
        ce_loss = F.cross_entropy(
            logits.view(-1, V), 
            target_seq.view(-1), 
            reduction='none'
        ).view(B, L)
    
    # Apply masks
    loss_mask = diffusion_mask.float()
    if valid_mask is not None:
        loss_mask = loss_mask * valid_mask.float()
    if position_weights is not None:
        loss_mask = loss_mask * position_weights.float()
    
    masked_loss = ce_loss * loss_mask
    
    # Compute per-sample mean loss (only over masked positions)
    num_masked = loss_mask.sum(dim=1).clamp(min=1e-6)
    per_sample_loss = masked_loss.sum(dim=1) / num_masked
    
    # Apply DPLM-style loss weighting
    loss_weight = compute_loss_weight(
        mask_ratio,
        strategy=loss_weight_strategy,
        epsilon=loss_weight_epsilon,
    )
    if loss_weight.dim() == 0:
        loss_weight = loss_weight.unsqueeze(0).expand(B)
    
    weighted_loss = per_sample_loss * loss_weight
    final_loss = weighted_loss.mean()
    
    # Compute metrics
    with torch.no_grad():
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == target_seq).float() * loss_mask
        accuracy = correct.sum() / num_masked.sum()
        
        metrics = {
            "loss": final_loss.item(),
            "accuracy": accuracy.item(),
            "num_masked": num_masked.mean().item(),
            "loss_weight": loss_weight.mean().item(),
        }
    
    return final_loss, metrics


# ============================================================================
# Iterative Denoising Inference
# ============================================================================

def get_denoising_schedule(
    num_steps: int,
    schedule_type: Literal["linear", "cosine", "sqrt"] = "cosine",
) -> np.ndarray:
    """
    Get the masking ratio schedule for iterative denoising.
    
    Args:
        num_steps: Number of denoising steps
        schedule_type: Type of schedule
    
    Returns:
        schedule: Array of mask ratios from 1.0 to 0.0 (high to low)
                  schedule[0] ≈ 1.0 (start with all masked)
                  schedule[-1] ≈ 0.0 (end with none masked)
    """
    # t goes from 0 to 1 over the denoising steps
    t = np.linspace(0, 1, num_steps + 1)
    
    if schedule_type == "linear":
        # Linear decay from 1 to 0
        return 1.0 - t
    elif schedule_type == "cosine":
        # Cosine decay from 1 to 0 (smooth transitions at endpoints)
        return 0.5 * (1.0 + np.cos(t * np.pi))
    elif schedule_type == "sqrt":
        # Sqrt decay (faster initial unmasking, slower refinement)
        return 1.0 - np.sqrt(t)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def select_positions_to_unmask(
    current_seq: torch.Tensor,
    logits: torch.Tensor,
    mask_token_idx: int,
    num_to_unmask: int,
    strategy: Literal["confidence", "random", "entropy"] = "confidence",
    temperature: float = 1.0,
    use_argmax: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select which masked positions to unmask based on model confidence.
    
    Args:
        current_seq: [B, L] Current sequence with masks
        logits: [B, L, V] Model output logits
        mask_token_idx: Index of mask token
        num_to_unmask: Number of positions to unmask
        strategy: Selection strategy
        temperature: Temperature for sampling
        use_argmax: If True, use argmax instead of sampling (default True for benchmark)
    
    Returns:
        new_seq: [B, L] Updated sequence with unmasked positions
        unmasked_positions: [B, L] Boolean tensor of newly unmasked positions
    """
    B, L, V = logits.shape
    device = logits.device
    
    # Find currently masked positions
    is_masked = (current_seq == mask_token_idx)  # [B, L]
    
    # Compute probabilities
    probs = F.softmax(logits / temperature, dim=-1)
    
    if strategy == "confidence":
        # Use maximum probability as confidence
        confidence = probs.max(dim=-1)[0]  # [B, L]
    elif strategy == "entropy":
        # Use negative entropy as confidence (low entropy = high confidence)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        confidence = -entropy
    elif strategy == "random":
        # Random confidence (for comparison)
        confidence = torch.rand(B, L, device=device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Set confidence of non-masked positions to -inf to ensure we only select from currently masked ones
    confidence = torch.where(is_masked, confidence, torch.tensor(-float('inf'), device=device))
    
    new_seq = current_seq.clone()
    unmasked_positions = torch.zeros(B, L, dtype=torch.bool, device=device)
    
    for b in range(B):
        # Get indices of masked positions
        masked_indices = torch.where(is_masked[b])[0]
        if len(masked_indices) == 0:
            continue
        
        # How many to unmask for this sample
        n_unmask = min(num_to_unmask, len(masked_indices))
        if n_unmask == 0:
            continue
        
        # Select top-k by confidence among MASKED positions
        conf_masked = confidence[b, masked_indices]
        _, top_indices = conf_masked.topk(n_unmask)
        positions_to_unmask = masked_indices[top_indices]
        
        # Fill these positions
        for pos in positions_to_unmask:
            if use_argmax:
                # Use argmax for deterministic benchmark performance
                token = logits[b, pos].argmax()
            else:
                # Sample from probability distribution for diversity
                token = torch.multinomial(probs[b, pos], 1).squeeze()
            
            new_seq[b, pos] = token
            unmasked_positions[b, pos] = True
    
    return new_seq, unmasked_positions


def rewalk_positions(
    current_seq: torch.Tensor,
    mask_token_idx: int,
    probs: torch.Tensor,
    rewalk_ratio: float = 0.1,
    min_confidence_threshold: float = 0.3,
    designable_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Re-mask low-confidence positions (ProRefiner-style rewalk).
    
    This allows the model to reconsider uncertain predictions,
    similar to Langevin dynamics noise injection.
    
    Args:
        current_seq: [B, L] Current sequence
        mask_token_idx: Index of mask token
        probs: [B, L, V] Model output probabilities
        rewalk_ratio: Proportion of non-masked low-confidence positions to re-mask
        min_confidence_threshold: Only re-mask if confidence below this
        designable_mask: [B, L] Optional mask limiting positions that can be
            re-masked. Fixed context positions are preserved when provided.
    
    Returns:
        rewalked_seq: [B, L] Sequence with some positions re-masked
    """
    B, L, V = probs.shape
    device = current_seq.device
    
    # Find non-masked positions
    is_not_masked = (current_seq != mask_token_idx)
    
    # Get confidence for each position
    confidence = probs.max(dim=-1)[0]  # [B, L]
    
    # Find low-confidence non-masked positions
    low_conf_mask = (confidence < min_confidence_threshold) & is_not_masked
    if designable_mask is not None:
        low_conf_mask = low_conf_mask & (designable_mask > 0)
    
    # Randomly select subset to re-mask
    rand_mask = torch.rand(B, L, device=device) < rewalk_ratio
    rewalk_mask = low_conf_mask & rand_mask
    
    # Apply re-masking
    rewalked_seq = current_seq.clone()
    rewalked_seq[rewalk_mask] = mask_token_idx
    
    return rewalked_seq


def iterative_denoise_threshold(
    model: nn.Module,
    feature_dict: dict,
    mask_token_idx: int,
    max_steps: int = 50,
    recover_threshold: float = 0.9,
    remask_threshold: float = 0.3,
    na_mask: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, list]:
    """
    Threshold-based iterative denoising (Confidence-based).
    
    Logic:
    1. Predict all positions.
    2. Recover positions where confidence > recover_threshold.
    3. Remask positions where confidence < remask_threshold (Rewalk).
    4. Repeat for max_steps.
    5. At the final step, fill all remaining masks with argmax.
    """
    model.eval()
    device = next(model.parameters()).device
    B, L = feature_dict["S"].shape
    
    # Initialize with all masks (for NA positions)
    current_seq = feature_dict["S"].clone()
    if na_mask is not None:
        current_seq[na_mask > 0] = mask_token_idx
    else:
        current_seq = torch.full((B, L), mask_token_idx, device=device, dtype=torch.long)
    
    trajectory = [current_seq.clone()]
    
    for step in range(max_steps):
        feature_dict_step = feature_dict.copy()
        feature_dict_step["S"] = current_seq
        
        # Get model predictions
        log_probs, probs = model(feature_dict_step)
        confidence, pred_tokens = probs.max(dim=-1)
        
        # 1. Recover high-confidence positions
        recover_mask = (confidence > recover_threshold) & (current_seq == mask_token_idx)
        if na_mask is not None:
            recover_mask = recover_mask & (na_mask > 0)
        
        current_seq[recover_mask] = pred_tokens[recover_mask]
        
        # 2. Remask low-confidence positions (except at the very last step)
        if step < max_steps - 1:
            remask_mask = (confidence < remask_threshold) & (current_seq != mask_token_idx)
            if na_mask is not None:
                remask_mask = remask_mask & (na_mask > 0)
            current_seq[remask_mask] = mask_token_idx
            
        trajectory.append(current_seq.clone())
        
        # If no more masks, we can stop early
        if na_mask is not None:
            if not (current_seq[na_mask > 0] == mask_token_idx).any():
                break
        elif not (current_seq == mask_token_idx).any():
            break
            
    # Final step: fill remaining masks
    if (current_seq == mask_token_idx).any():
        feature_dict_final = feature_dict.copy()
        feature_dict_final["S"] = current_seq
        log_probs, _ = model(feature_dict_final)
        final_preds = log_probs.argmax(dim=-1)
        
        remaining_masks = (current_seq == mask_token_idx)
        if na_mask is not None:
            remaining_masks = remaining_masks & (na_mask > 0)
            
        current_seq[remaining_masks] = final_preds[remaining_masks]
        trajectory.append(current_seq.clone())
        
    return current_seq, trajectory


@torch.no_grad()
def iterative_denoise(
    model: nn.Module,
    feature_dict: dict,
    mask_token_idx: int,
    num_steps: int = 50,
    schedule_type: str = "cosine",
    selection_strategy: str = "confidence",
    temperature: float = 1.0,
    use_argmax: bool = True,
    rewalk_enabled: bool = True,   # Default: enabled for best performance
    rewalk_ratio: float = 0.1,
    rewalk_threshold: float = 0.3,
    designable_mask: Optional[torch.Tensor] = None,
    na_mask: Optional[torch.Tensor] = None,  # Legacy alias for designable_mask
    verbose: bool = False,
) -> Tuple[torch.Tensor, list]:
    """
    Iterative denoising for sequence generation (MaskGIT/Muse style).
    
    This function supports designing BOTH protein and nucleic acid sequences
    when designable_mask includes protein positions (M-MPNN-D mode).
    
    DEFAULT CONFIG (Production):
    - num_steps: 50 (optimal), 30 (efficient alternative)
    - schedule_type: "cosine" (controls unmasking rate)
    - selection_strategy: "confidence" (unmask highest confidence first)
    - use_argmax: True (deterministic decoding)
    - rewalk_enabled: True (re-mask low confidence positions)
    
    Starting from all-mask sequence, iteratively predict and unmask positions
    until full sequence is generated.
    
    Args:
        model: M-MPNN-D model (supports protein + NA)
        feature_dict: Dictionary with structure features
        mask_token_idx: Index of mask token
        num_steps: Number of denoising steps
        schedule_type: Masking schedule type
        selection_strategy: How to select positions to unmask
        temperature: Sampling temperature
        use_argmax: Use argmax for token selection
        rewalk_enabled: Whether to use rewalk mechanism
        rewalk_ratio: Proportion of low-conf positions to re-mask
        rewalk_threshold: Confidence threshold for rewalk
        designable_mask: [B, L] Mask for designable positions (protein + NA)
        na_mask: [B, L] Legacy alias for designable_mask
        verbose: Print progress
    
    Returns:
        final_seq: [B, L] Generated sequence
        trajectory: List of intermediate sequences
    """
    # Handle legacy na_mask parameter
    if designable_mask is None and na_mask is not None:
        designable_mask = na_mask
    model.eval()
    device = next(model.parameters()).device
    
    B, L = feature_dict["S"].shape
    
    # Initialize with all masks (for designable positions)
    current_seq = feature_dict["S"].clone()
    if designable_mask is not None:
        current_seq[designable_mask > 0] = mask_token_idx
    else:
        current_seq = torch.full((B, L), mask_token_idx, device=device, dtype=torch.long)
    
    # Get denoising schedule
    schedule = get_denoising_schedule(num_steps, schedule_type)
    
    trajectory = [current_seq.clone()]
    
    for step in range(num_steps):
        # Current and target mask ratios
        r_current = schedule[step]
        r_next = schedule[step + 1]
        
        # Update feature dict with current sequence
        feature_dict_step = feature_dict.copy()
        feature_dict_step["S"] = current_seq
        
        # Get model predictions
        log_probs, probs = model(feature_dict_step)
        logits = log_probs  # Model returns log_probs
        
        # Calculate how many positions to unmask
        if designable_mask is not None:
            total_maskable = (designable_mask > 0).sum(dim=1).float()
        else:
            total_maskable = torch.full((B,), L, device=device).float()
        
        num_masked_current = (current_seq == mask_token_idx).sum(dim=1).float()
        num_masked_target = (total_maskable * r_next).long()
        num_to_unmask = (num_masked_current - num_masked_target).clamp(min=0).long()
        
        # Unmask positions
        # For simplicity, use batch-wide number (could be per-sample)
        avg_unmask = int(num_to_unmask.float().mean().item())
        if avg_unmask > 0:
            current_seq, _ = select_positions_to_unmask(
                current_seq, logits, mask_token_idx, avg_unmask,
                strategy=selection_strategy, temperature=temperature,
                use_argmax=use_argmax
            )
        
        # Optional: rewalk mechanism
        if rewalk_enabled and step < num_steps - 1:
            current_seq = rewalk_positions(
                current_seq, mask_token_idx, probs,
                rewalk_ratio=rewalk_ratio,
                min_confidence_threshold=rewalk_threshold,
                designable_mask=designable_mask
            )
        
        trajectory.append(current_seq.clone())
        
        if verbose and step % 10 == 0:
            n_masked = (current_seq == mask_token_idx).sum().item()
            print(f"Step {step}/{num_steps}: {n_masked} masked positions remaining")
    
    # Final prediction for any remaining masks
    feature_dict_final = feature_dict.copy()
    feature_dict_final["S"] = current_seq
    log_probs, probs = model(feature_dict_final)
    
    # Fill remaining masks with argmax predictions
    remaining_masks = (current_seq == mask_token_idx)
    if designable_mask is not None:
        remaining_masks = remaining_masks & (designable_mask > 0)
    if remaining_masks.any():
        final_preds = log_probs.argmax(dim=-1)
        current_seq[remaining_masks] = final_preds[remaining_masks]
    
    trajectory.append(current_seq.clone())
    
    return current_seq, trajectory


# ============================================================================
# Utility Functions
# ============================================================================

def get_na_mask_from_polymer_masks(
    dna_mask: torch.Tensor,
    rna_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Combine DNA and RNA masks into a single nucleic acid mask.
    
    Args:
        dna_mask: [B, L] DNA position mask
        rna_mask: [B, L] RNA position mask
    
    Returns:
        na_mask: [B, L] Combined NA mask
    """
    return (dna_mask > 0) | (rna_mask > 0)


def get_designable_mask(
    feature_dict: dict,
    mask_mode: str = "na_only",
) -> Optional[torch.Tensor]:
    """
    Get designable positions mask based on mask_mode.
    
    This is the key function for M-MPNN-D: it controls which positions
    can be masked and designed during diffusion training.
    
    Args:
        feature_dict: Feature dictionary containing polymer masks
        mask_mode: One of:
            - "all": Mask all valid positions (protein + NA)
            - "protein_only": Only mask protein positions
            - "na_only": Only mask nucleic acid positions
            - "interface": Only mask interface positions
    
    Returns:
        designable_mask: [B, L] Boolean mask for designable positions,
                        or None if all positions should be designable
    """
    dna_mask = feature_dict.get("dna_mask")
    rna_mask = feature_dict.get("rna_mask")
    protein_mask = feature_dict.get("protein_mask")
    interface_mask = feature_dict.get("interface_mask")
    
    if mask_mode == "all":
        # Mask all valid positions (protein + NA)
        # Return None to let apply_absorbing_mask use valid_mask
        return None
    
    elif mask_mode == "protein_only":
        # Only mask protein positions
        if protein_mask is not None:
            return (protein_mask > 0).float()
        else:
            return None
    
    elif mask_mode == "na_only":
        # Only mask nucleic acid positions (original behavior)
        if dna_mask is not None and rna_mask is not None:
            return ((dna_mask > 0) | (rna_mask > 0)).float()
        else:
            return None
    
    elif mask_mode == "interface":
        # Only mask interface positions
        if interface_mask is not None:
            return (interface_mask > 0).float()
        else:
            return None
    
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}. "
                        f"Expected one of: all, protein_only, na_only, interface")


def create_bidirectional_attention_mask(
    mask: torch.Tensor,
    E_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Create bidirectional attention mask (all valid positions can attend to each other).
    
    This replaces the causal mask used in autoregressive decoding.
    
    Args:
        mask: [B, L] Valid position mask
        E_idx: [B, L, K] Edge indices
    
    Returns:
        mask_attend: [B, L, K] Attention mask
    """
    from na_model_utils import gather_nodes
    
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    
    return mask_attend
