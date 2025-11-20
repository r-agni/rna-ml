"""
Comprehensive evaluation script for RNA secondary structure prediction.

Evaluates all three prediction heads (dot-bracket, binary, pairing) and
computes detailed metrics with constraint-based post-processing.
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm

from model import LyraContact
from dataset import get_dataloaders, RNA_VOCAB_SIZE, parse_base_pairs
from structure_utils import (
    ensemble_predictions,
    greedy_decode_pairs,
    dotbracket_to_pairs,
    compute_structure_metrics
)


@torch.no_grad()
def evaluate_comprehensive(model, data_loader, device):
    """
    Comprehensive evaluation with all prediction heads.

    Returns:
        Dictionary with metrics for each head and ensemble
    """
    model.eval()

    all_results = {
        'dotbracket': [],
        'pairing': [],
        'pairing_constrained': [],
        'ensemble': []
    }

    pbar = tqdm(data_loader, desc="Evaluating")

    for batch in pbar:
        # Move to device
        sequences = batch['sequence'].to(device)
        pairing_targets = batch['pairing_targets'].to(device)
        dotbracket_targets = batch['dotbracket_targets'].to(device)
        binary_targets = batch['binary_targets'].to(device)
        mask = batch['mask'].to(device)
        lengths = batch['length']

        # Forward pass
        outputs = model(sequences)

        # Get predictions from each head
        dotbracket_logits = outputs['dotbracket']  # (B, L, 3)
        binary_logits = outputs['binary']  # (B, L, 2)
        pairing_logits = outputs['pairing']  # (B, L, L+1)

        B = sequences.size(0)

        # Method 1: Dot-bracket predictions
        dotbracket_preds = dotbracket_logits.argmax(dim=-1)  # (B, L)
        for b in range(B):
            length = lengths[b].item()

            # Convert predicted dot-bracket to pairs
            chars = []
            for i in range(length):
                idx = dotbracket_preds[b, i].item()
                chars.append('.' if idx == 0 else '(' if idx == 1 else ')')
            pred_structure = ''.join(chars)
            pred_pairs_db = dotbracket_to_pairs(pred_structure)

            # Convert target dot-bracket to pairs
            true_chars = []
            for i in range(length):
                idx = dotbracket_targets[b, i].item()
                if idx == -100:  # padding
                    break
                true_chars.append('.' if idx == 0 else '(' if idx == 1 else ')')
            true_structure = ''.join(true_chars)
            true_pairs_db = dotbracket_to_pairs(true_structure)

            metrics_db = compute_structure_metrics(pred_pairs_db, true_pairs_db)
            all_results['dotbracket'].append(metrics_db)

        # Method 2: Pairing head (unconstrained)
        pairing_preds = pairing_logits.argmax(dim=-1)  # (B, L)
        for b in range(B):
            length = lengths[b].item()

            # Extract pairs from pairing predictions
            pred_pairs_p = []
            used = set()
            for i in range(length):
                j = pairing_preds[b, i].item()
                if j < length and i < j and i not in used and j not in used:
                    pred_pairs_p.append((i, j))
                    used.add(i)
                    used.add(j)

            # Ground truth pairs from pairing targets
            true_pairs_p = []
            used_true = set()
            for i in range(length):
                j = pairing_targets[b, i].item()
                if j < length and i < j and i not in used_true and j not in used_true:
                    true_pairs_p.append((i, j))
                    used_true.add(i)
                    used_true.add(j)

            metrics_p = compute_structure_metrics(pred_pairs_p, true_pairs_p)
            all_results['pairing'].append(metrics_p)

        # Method 3: Pairing head with constraints
        constrained_pairs = greedy_decode_pairs(pairing_logits, mask, enforce_symmetry=True)
        for b in range(B):
            length = lengths[b].item()

            # Ground truth (same as above)
            true_pairs_c = []
            used_true = set()
            for i in range(length):
                j = pairing_targets[b, i].item()
                if j < length and i < j and i not in used_true and j not in used_true:
                    true_pairs_c.append((i, j))
                    used_true.add(i)
                    used_true.add(j)

            metrics_c = compute_structure_metrics(constrained_pairs[b], true_pairs_c)
            all_results['pairing_constrained'].append(metrics_c)

        # Method 4: Ensemble of all heads
        ensemble_pairs = ensemble_predictions(
            dotbracket_logits, binary_logits, pairing_logits, mask
        )
        for b in range(B):
            length = lengths[b].item()

            # Ground truth (same as above)
            true_pairs_e = []
            used_true = set()
            for i in range(length):
                j = pairing_targets[b, i].item()
                if j < length and i < j and i not in used_true and j not in used_true:
                    true_pairs_e.append((i, j))
                    used_true.add(i)
                    used_true.add(j)

            metrics_e = compute_structure_metrics(ensemble_pairs[b], true_pairs_e)
            all_results['ensemble'].append(metrics_e)

    # Aggregate results
    aggregated = {}
    for method, results in all_results.items():
        if results:
            aggregated[method] = {
                'precision': np.mean([r['precision'] for r in results]),
                'recall': np.mean([r['recall'] for r in results]),
                'f1': np.mean([r['f1'] for r in results]),
                'std_f1': np.std([r['f1'] for r in results])
            }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA structure prediction")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='dataset.csv',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')

    # Model args (should match training)
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--num_s4', type=int, default=4)
    parser.add_argument('--pgc_expansion', type=float, default=2.0)
    parser.add_argument('--num_pgc', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.split} data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=0
    )

    # Select split
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader

    # Create model
    print("\nCreating model...")
    model = LyraContact(
        model_dimension=args.model_dim,
        pgc_configs=[(args.pgc_expansion, args.num_pgc)],
        num_s4=args.num_s4,
        d_input=RNA_VOCAB_SIZE,
        d_state=args.d_state,
        dropout=args.dropout
    ).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    print("=" * 80)

    results = evaluate_comprehensive(model, data_loader, device)

    # Print results
    print("\nResults:")
    print("=" * 80)

    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f} ± {metrics['std_f1']:.4f}")

    print("\n" + "=" * 80)
    print("\nBest method by F1 score:")
    best_method = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"  {best_method[0].upper()}: {best_method[1]['f1']:.4f}")


if __name__ == "__main__":
    main()
