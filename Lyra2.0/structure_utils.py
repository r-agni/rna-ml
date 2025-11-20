"""
Utilities for RNA secondary structure prediction and constraint enforcement.

This module provides functions for:
1. Converting between different structure representations
2. Enforcing structural constraints (balanced brackets, no conflicts)
3. Post-processing predictions to ensure valid structures
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """
    Convert dot-bracket notation to list of base pairs.

    Args:
        structure: Dot-bracket string (e.g., "(((...)))")

    Returns:
        List of (i, j) tuples representing paired positions (0-indexed)
    """
    stack = []
    pairs = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    return pairs


def pairs_to_dotbracket(pairs: List[Tuple[int, int]], length: int) -> str:
    """
    Convert list of base pairs to dot-bracket notation.

    Args:
        pairs: List of (i, j) tuples where i < j
        length: Total sequence length

    Returns:
        Dot-bracket string
    """
    structure = ['.'] * length

    for i, j in pairs:
        structure[i] = '('
        structure[j] = ')'

    return ''.join(structure)


def is_valid_structure(pairs: List[Tuple[int, int]], length: int) -> bool:
    """
    Check if a set of base pairs forms a valid secondary structure.

    Valid structure requirements:
    1. No position pairs with multiple partners
    2. No crossing base pairs (pseudoknots)
    3. All indices within bounds

    Args:
        pairs: List of (i, j) tuples
        length: Sequence length

    Returns:
        True if structure is valid
    """
    if not pairs:
        return True

    # Check bounds
    for i, j in pairs:
        if i < 0 or j < 0 or i >= length or j >= length:
            return False
        if i >= j:
            return False

    # Check for multiple partners
    paired_positions = set()
    for i, j in pairs:
        if i in paired_positions or j in paired_positions:
            return False
        paired_positions.add(i)
        paired_positions.add(j)

    # Check for crossing pairs (pseudoknots)
    for idx1, (i1, j1) in enumerate(pairs):
        for idx2 in range(idx1 + 1, len(pairs)):
            i2, j2 = pairs[idx2]
            # Check if pairs cross: i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1
            if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                return False

    return True


def enforce_pairing_constraints(pairing_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Enforce symmetry constraint on pairing predictions.

    If position i predicts pairing with j, encourage j to predict pairing with i.

    Args:
        pairing_logits: (B, L, L+1) logits before softmax
        mask: (B, L) valid position mask

    Returns:
        Constrained logits with enforced symmetry
    """
    B, L, num_classes = pairing_logits.shape

    # Extract pairwise scores (exclude unpaired class)
    pair_scores = pairing_logits[:, :, :L]  # (B, L, L)
    unpaired_scores = pairing_logits[:, :, L:L+1]  # (B, L, 1)

    # Symmetrize the pair scores: avg(score[i,j], score[j,i])
    pair_scores_sym = (pair_scores + pair_scores.transpose(1, 2)) / 2

    # Mask invalid positions
    mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
    pair_scores_sym = pair_scores_sym.masked_fill(~mask_2d, float('-inf'))

    # Combine back with unpaired scores
    constrained_logits = torch.cat([pair_scores_sym, unpaired_scores], dim=-1)

    return constrained_logits


def greedy_decode_pairs(pairing_logits: torch.Tensor, mask: torch.Tensor,
                        enforce_symmetry: bool = True) -> List[List[Tuple[int, int]]]:
    """
    Greedy decoding of base pairs with constraint enforcement.

    Args:
        pairing_logits: (B, L, L+1) prediction logits
        mask: (B, L) valid position mask
        enforce_symmetry: Whether to enforce pairing symmetry

    Returns:
        List of lists of (i, j) pairs for each sequence in batch
    """
    B, L, num_classes = pairing_logits.shape

    if enforce_symmetry:
        pairing_logits = enforce_pairing_constraints(pairing_logits, mask)

    # Get predictions
    predictions = pairing_logits.argmax(dim=-1)  # (B, L)

    batch_pairs = []

    for b in range(B):
        pairs = []
        used = set()

        # Get valid length
        length = mask[b].sum().item()

        for i in range(length):
            pred_j = predictions[b, i].item()

            # Skip if unpaired or already used
            if pred_j >= length or i in used:
                continue

            # Check symmetry: does j also predict i?
            if enforce_symmetry:
                pred_i_from_j = predictions[b, pred_j].item()
                if pred_i_from_j != i:
                    continue  # Not symmetric, skip

            # Add pair (always store with i < j)
            if i < pred_j:
                pairs.append((i, pred_j))
            else:
                pairs.append((pred_j, i))

            used.add(i)
            used.add(pred_j)

        batch_pairs.append(pairs)

    return batch_pairs


def ensemble_predictions(dotbracket_logits: torch.Tensor,
                        binary_logits: torch.Tensor,
                        pairing_logits: torch.Tensor,
                        mask: torch.Tensor,
                        weights: Optional[dict] = None) -> List[List[Tuple[int, int]]]:
    """
    Ensemble multiple prediction heads for final structure prediction.

    Strategy:
    1. Use binary head to identify likely paired positions
    2. Use dot-bracket head to get structure topology
    3. Use pairing head to resolve exact partners
    4. Combine predictions with weighted voting

    Args:
        dotbracket_logits: (B, L, 3) dot-bracket predictions
        binary_logits: (B, L, 2) binary paired/unpaired predictions
        pairing_logits: (B, L, L+1) pairing partner predictions
        mask: (B, L) valid position mask
        weights: Optional weights for each head

    Returns:
        List of lists of (i, j) pairs for each sequence
    """
    if weights is None:
        weights = {'dotbracket': 0.5, 'binary': 0.2, 'pairing': 0.3}

    B, L = mask.shape

    # Method 1: Convert dot-bracket predictions to pairs
    dotbracket_preds = dotbracket_logits.argmax(dim=-1)  # (B, L)
    dotbracket_pairs = []

    for b in range(B):
        length = mask[b].sum().item()
        # Convert indices to characters
        chars = ['.' if dotbracket_preds[b, i] == 0
                 else '(' if dotbracket_preds[b, i] == 1
                 else ')'
                 for i in range(length)]
        structure = ''.join(chars)
        pairs = dotbracket_to_pairs(structure)
        dotbracket_pairs.append(pairs)

    # Method 2: Use pairing head with constraints
    pairing_pairs = greedy_decode_pairs(pairing_logits, mask, enforce_symmetry=True)

    # Method 3: Filter pairing predictions by binary confidence
    binary_probs = F.softmax(binary_logits, dim=-1)  # (B, L, 2)
    paired_confidence = binary_probs[:, :, 1]  # (B, L) - prob of being paired

    # Ensemble: Start with dot-bracket, validate with pairing head
    final_pairs = []

    for b in range(B):
        # Start with dot-bracket pairs (primary)
        pairs = set(dotbracket_pairs[b])

        # Add high-confidence pairs from pairing head
        for i, j in pairing_pairs[b]:
            # Check if both positions have high paired confidence
            conf_i = paired_confidence[b, i].item()
            conf_j = paired_confidence[b, j].item()

            if conf_i > 0.5 and conf_j > 0.5:
                pairs.add((i, j))

        # Validate and remove conflicting pairs
        pairs = list(pairs)
        length = mask[b].sum().item()

        # Remove conflicts greedily (keep higher confidence pairs)
        used = set()
        valid_pairs = []

        # Sort by combined confidence
        pairs_with_conf = []
        for i, j in pairs:
            conf = (paired_confidence[b, i] + paired_confidence[b, j]).item()
            pairs_with_conf.append(((i, j), conf))

        pairs_with_conf.sort(key=lambda x: x[1], reverse=True)

        for (i, j), conf in pairs_with_conf:
            if i not in used and j not in used:
                valid_pairs.append((i, j))
                used.add(i)
                used.add(j)

        final_pairs.append(sorted(valid_pairs))

    return final_pairs


def compute_structure_metrics(pred_pairs: List[Tuple[int, int]],
                              true_pairs: List[Tuple[int, int]]) -> dict:
    """
    Compute metrics for structure prediction.

    Args:
        pred_pairs: Predicted base pairs
        true_pairs: Ground truth base pairs

    Returns:
        Dictionary with precision, recall, F1
    """
    pred_set = set(pred_pairs)
    true_set = set(true_pairs)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
