import os
import yaml
import torch
import numpy as np
import pandas as pd
from .model import create_lyra_model
from .data_loader import RNADataset


class RNAPredictor:
    """Predictor class for RNA secondary structure contact maps"""

    def __init__(self, checkpoint_path, config_path=None, device=None):
        """
        Initialize predictor with trained model.

        Args:
            checkpoint_path: Path to saved model checkpoint
            config_path: Optional path to config file (uses checkpoint config if None)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint['config']

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Create and load model
        self.model = create_lyra_model(self.config['model']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")

    def predict_sequence(self, sequence, threshold=0.5):
        """
        Predict contact map for a single RNA sequence.

        Args:
            sequence: RNA sequence string (e.g., "ACGUACGU...")
            threshold: Threshold for binary contact prediction

        Returns:
            dict with:
                - 'contact_map': Binary contact matrix (L, L)
                - 'contact_probs': Probability contact matrix (L, L)
                - 'base_pairs': List of base pair tuples [(i, j), ...]
        """
        # One-hot encode sequence
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        encoding = np.zeros((len(sequence), 4), dtype=np.float32)
        for i, nuc in enumerate(sequence):
            if nuc in nuc_to_idx:
                encoding[i, nuc_to_idx[nuc]] = 1.0

        # Convert to tensor and add batch dimension
        seq_tensor = torch.from_numpy(encoding).unsqueeze(0).to(self.device)
        seq_len = torch.tensor([len(sequence)]).to(self.device)

        # Predict
        with torch.no_grad():
            contact_logits = self.model(seq_tensor, seq_len)
            contact_probs = torch.sigmoid(contact_logits).squeeze(0).cpu().numpy()

        # Binarize
        contact_map = (contact_probs > threshold).astype(int)

        # Extract base pairs (upper triangle only to avoid duplicates)
        base_pairs = []
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                if contact_map[i, j] == 1:
                    base_pairs.append((i, j))

        return {
            'contact_map': contact_map,
            'contact_probs': contact_probs,
            'base_pairs': base_pairs
        }

    def predict_batch(self, sequences, threshold=0.5, batch_size=32):
        """
        Predict contact maps for a batch of sequences.

        Args:
            sequences: List of RNA sequence strings
            threshold: Threshold for binary contact prediction
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Find max length in batch
            max_len = max(len(seq) for seq in batch_seqs)

            # Encode and pad
            batch_encodings = []
            seq_lens = []
            nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

            for seq in batch_seqs:
                encoding = np.zeros((max_len, 4), dtype=np.float32)
                for idx, nuc in enumerate(seq):
                    if nuc in nuc_to_idx:
                        encoding[idx, nuc_to_idx[nuc]] = 1.0
                batch_encodings.append(encoding)
                seq_lens.append(len(seq))

            # Convert to tensors
            batch_tensor = torch.from_numpy(np.array(batch_encodings)).to(self.device)
            seq_lens_tensor = torch.tensor(seq_lens).to(self.device)

            # Predict
            with torch.no_grad():
                contact_logits = self.model(batch_tensor, seq_lens_tensor)
                contact_probs = torch.sigmoid(contact_logits).cpu().numpy()

            # Process each sequence in batch
            for j, seq in enumerate(batch_seqs):
                seq_len = seq_lens[j]
                probs = contact_probs[j, :seq_len, :seq_len]
                contact_map = (probs > threshold).astype(int)

                # Extract base pairs
                base_pairs = []
                for row in range(seq_len):
                    for col in range(row + 1, seq_len):
                        if contact_map[row, col] == 1:
                            base_pairs.append((row, col))

                predictions.append({
                    'contact_map': contact_map,
                    'contact_probs': probs,
                    'base_pairs': base_pairs
                })

        return predictions

    def predict_from_csv(self, csv_path, output_path=None, threshold=0.5):
        """
        Predict on sequences from CSV file.

        Args:
            csv_path: Path to CSV with 'id' and 'sequence' columns
            output_path: Path to save predictions (CSV format)
            threshold: Threshold for binary contact prediction

        Returns:
            DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Predict
        print(f"Predicting on {len(df)} sequences...")
        predictions = self.predict_batch(df['sequence'].tolist(), threshold=threshold)

        # Add predictions to dataframe
        df['predicted_base_pairs'] = [pred['base_pairs'] for pred in predictions]

        # Save if output path provided
        if output_path is not None:
            df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

        return df


def contact_map_to_dot_bracket(contact_map):
    """
    Convert contact map to dot-bracket notation.

    Args:
        contact_map: Binary contact matrix (L, L)

    Returns:
        Dot-bracket string
    """
    seq_len = contact_map.shape[0]
    structure = ['.'] * seq_len

    # Extract base pairs
    base_pairs = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if contact_map[i, j] == 1:
                base_pairs.append((i, j))

    # Sort by first position
    base_pairs.sort()

    # Assign brackets (simple version, doesn't handle pseudoknots)
    for i, j in base_pairs:
        structure[i] = '('
        structure[j] = ')'

    return ''.join(structure)


def main():
    """Example usage of predictor"""
    import argparse

    parser = argparse.ArgumentParser(description='Predict RNA secondary structure contact maps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--csv', type=str, default=None, help='Path to input CSV file')
    parser.add_argument('--sequence', type=str, default=None, help='Single RNA sequence to predict')
    parser.add_argument('--output', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Create predictor
    predictor = RNAPredictor(args.checkpoint, args.config, args.device)

    if args.sequence is not None:
        # Predict single sequence
        print(f"\nPredicting structure for sequence: {args.sequence}")
        result = predictor.predict_sequence(args.sequence, args.threshold)

        print(f"\nPredicted {len(result['base_pairs'])} base pairs:")
        for i, j in result['base_pairs']:
            print(f"  ({i}, {j})")

        # Convert to dot-bracket
        dot_bracket = contact_map_to_dot_bracket(result['contact_map'])
        print(f"\nDot-bracket notation:")
        print(f"  {args.sequence}")
        print(f"  {dot_bracket}")

    elif args.csv is not None:
        # Predict from CSV
        predictions = predictor.predict_from_csv(args.csv, args.output, args.threshold)
        print(f"\nProcessed {len(predictions)} sequences")

    else:
        print("Please provide either --sequence or --csv argument")


if __name__ == '__main__':
    main()
