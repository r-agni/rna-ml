import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np


def evaluate_model(model, test_loader, device, output_dir=None):
    """
    Evaluate model on test dataset.

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        output_dir: Optional directory to save detailed results

    Returns:
        dict: Dictionary containing aggregate metrics
    """
    model.eval()

    all_preds = []
    all_trues = []
    exact_matches = 0
    total_samples = 0

    # For per-sample results
    sample_results = []

    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sequences = batch['sequence'].to(device)
            contact_matrices = batch['contact_matrix'].to(device)
            seq_lens = batch['seq_len'].to(device)

            # Forward pass
            pred_contacts = model(sequences, seq_lens)

            # Apply sigmoid to get probabilities
            pred_probs = torch.sigmoid(pred_contacts)

            # Binarize predictions (threshold 0.5)
            pred_binary = (pred_probs > 0.5).float()

            batch_size = pred_binary.shape[0]

            # Process each sample in batch
            for i in range(batch_size):
                seq_len = seq_lens[i].item()

                # Extract valid region (upper triangle to avoid double counting)
                pred_valid = pred_binary[i, :seq_len, :seq_len].triu(diagonal=1)
                true_valid = contact_matrices[i, :seq_len, :seq_len].triu(diagonal=1)

                # Flatten
                pred_flat = pred_valid.flatten().cpu().numpy()
                true_flat = true_valid.flatten().cpu().numpy()

                all_preds.extend(pred_flat)
                all_trues.extend(true_flat)

                # Check exact match
                is_exact_match = np.array_equal(pred_flat, true_flat)
                if is_exact_match:
                    exact_matches += 1

                # Calculate per-sample metrics
                sample_f1 = f1_score(true_flat, pred_flat, zero_division=0)
                sample_mcc = matthews_corrcoef(true_flat, pred_flat)
                sample_precision = precision_score(true_flat, pred_flat, zero_division=0)
                sample_recall = recall_score(true_flat, pred_flat, zero_division=0)

                sample_results.append({
                    'sample_idx': total_samples,
                    'seq_len': seq_len,
                    'f1': sample_f1,
                    'mcc': sample_mcc,
                    'precision': sample_precision,
                    'recall': sample_recall,
                    'exact_match': int(is_exact_match),
                    'num_true_contacts': int(true_flat.sum()),
                    'num_pred_contacts': int(pred_flat.sum())
                })

                total_samples += 1

    # Calculate aggregate metrics
    if total_samples == 0:
        raise ValueError("Test loader is empty - no samples to evaluate")

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    metrics = {
        'test_f1': f1_score(all_trues, all_preds, zero_division=0),
        'test_mcc': matthews_corrcoef(all_trues, all_preds),
        'test_precision': precision_score(all_trues, all_preds, zero_division=0),
        'test_recall': recall_score(all_trues, all_preds, zero_division=0),
        'test_exact_match': exact_matches / total_samples,
        'test_samples': total_samples
    }

    # Save detailed results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(sample_results)
        results_path = os.path.join(output_dir, 'test_results_per_sample.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved per-sample results to {results_path}")

        # Save summary metrics
        summary_path = os.path.join(output_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Test Set Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"F1 Score: {metrics['test_f1']:.4f}\n")
            f.write(f"MCC: {metrics['test_mcc']:.4f}\n")
            f.write(f"Precision: {metrics['test_precision']:.4f}\n")
            f.write(f"Recall: {metrics['test_recall']:.4f}\n")
            f.write(f"Exact Match: {metrics['test_exact_match']:.4f}\n")
        print(f"Saved summary to {summary_path}")

    return metrics
