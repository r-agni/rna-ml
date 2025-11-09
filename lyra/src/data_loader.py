import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RNADataset(Dataset):
    """
    Dataset for RNA secondary structure prediction.

    Loads RNA sequences and converts base pairs to contact matrices.
    """

    def __init__(self, csv_path, indices=None, augment=False, augment_prob=0.5):
        """
        Args:
            csv_path: Path to the CSV file
            indices: Optional list of indices to use (for train/val/test splits)
            augment: If True, apply data augmentation (default False)
            augment_prob: Probability of applying augmentation (default 0.5)
        """
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.augment_prob = augment_prob

        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        # Filter out invalid rows (where sequence is not a string or is NaN)
        initial_len = len(self.df)
        self.df = self.df[self.df['sequence'].apply(lambda x: isinstance(x, str) and pd.notna(x))]
        self.df = self.df.reset_index(drop=True)

        filtered_count = initial_len - len(self.df)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} invalid sequences from dataset")

        # Nucleotide to index mapping for one-hot encoding
        self.nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

        # Reverse complement mapping for RNA
        self.complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}

    def __len__(self):
        return len(self.df)

    def reverse_complement(self, sequence):
        """
        Get reverse complement of RNA sequence.

        Args:
            sequence: String of nucleotides (A, C, G, U)

        Returns:
            Reverse complement sequence
        """
        return ''.join(self.complement.get(nuc, nuc) for nuc in reversed(sequence))

    def apply_augmentation(self, sequence, contact_matrix):
        """
        Apply data augmentation to sequence and contact matrix.

        Args:
            sequence: RNA sequence string
            contact_matrix: Contact matrix tensor

        Returns:
            Augmented sequence string and contact matrix
        """
        if np.random.random() < self.augment_prob:
            # Reverse complement augmentation
            sequence = self.reverse_complement(sequence)
            # Reverse the contact matrix along both dimensions
            contact_matrix = torch.flip(contact_matrix, dims=[0, 1])

        return sequence, contact_matrix

    def encode_sequence(self, sequence):
        """
        One-hot encode RNA sequence.

        Args:
            sequence: String of nucleotides (A, C, G, U)

        Returns:
            Tensor of shape (len, 4) with one-hot encoding
        """
        encoding = np.zeros((len(sequence), 4), dtype=np.float32)
        for i, nuc in enumerate(sequence):
            if nuc in self.nuc_to_idx:
                encoding[i, self.nuc_to_idx[nuc]] = 1.0
        return torch.from_numpy(encoding)

    def create_contact_matrix(self, base_pairs, seq_len):
        """
        Create contact matrix from base pairs.

        Args:
            base_pairs: JSON string of [[i, j], ...] pairs
            seq_len: Length of the sequence

        Returns:
            Tensor of shape (seq_len, seq_len) with 1s at base pair positions
        """
        seq_len = int(seq_len)  # Ensure seq_len is an integer
        matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

        try:
            pairs = json.loads(base_pairs)
            for pair in pairs:
                i, j = pair
                if 0 <= i < seq_len and 0 <= j < seq_len:
                    matrix[i, j] = 1.0
                    matrix[j, i] = 1.0  # Symmetric
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Handle malformed base_pairs
            pass

        return torch.from_numpy(matrix)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict with keys:
                - 'sequence': One-hot encoded sequence (L, 4)
                - 'contact_matrix': Contact matrix (L, L)
                - 'seq_len': Sequence length
                - 'id': Sequence ID
        """
        row = self.df.iloc[idx]

        sequence = row['sequence']
        seq_len = row['len']
        base_pairs = row['base_pairs']
        seq_id = row['id']

        # Create contact matrix
        contact_matrix = self.create_contact_matrix(base_pairs, seq_len)

        # Apply augmentation if enabled
        if self.augment:
            sequence, contact_matrix = self.apply_augmentation(sequence, contact_matrix)

        # Encode sequence
        encoded_seq = self.encode_sequence(sequence)

        return {
            'sequence': encoded_seq,
            'contact_matrix': contact_matrix,
            'seq_len': seq_len,
            'id': seq_id
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of samples from RNADataset

    Returns:
        dict with batched and padded tensors
    """
    # Find max length in batch (ensure it's an integer)
    max_len = int(max(sample['seq_len'] for sample in batch))

    batch_size = len(batch)

    # Initialize padded tensors
    sequences = torch.zeros(batch_size, max_len, 4)
    contact_matrices = torch.zeros(batch_size, max_len, max_len)
    seq_lens = torch.tensor([int(sample['seq_len']) for sample in batch])
    ids = [sample['id'] for sample in batch]

    # Fill in the data
    for i, sample in enumerate(batch):
        seq_len = int(sample['seq_len'])
        sequences[i, :seq_len, :] = sample['sequence']
        contact_matrices[i, :seq_len, :seq_len] = sample['contact_matrix']

    return {
        'sequence': sequences,
        'contact_matrix': contact_matrices,
        'seq_len': seq_lens,
        'id': ids,
        'max_len': max_len
    }


def create_data_splits(csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42, deduplicate=True):
    """
    Create train/val/test splits from the dataset, ensuring no sequence leakage.

    Args:
        csv_path: Path to CSV file
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for testing (default 0.1)
        random_seed: Random seed for reproducibility
        deduplicate: If True, split by unique sequences to prevent leakage (default True)

    Returns:
        tuple of (train_indices, val_indices, test_indices)
    """
    df = pd.read_csv(csv_path)

    if deduplicate:
        # Group by sequence to prevent leakage across splits
        print("Creating splits by unique sequences to prevent data leakage...")

        # Create a mapping of unique sequences to their indices
        sequence_groups = df.groupby('sequence').apply(lambda x: x.index.tolist()).to_dict()
        unique_sequences = list(sequence_groups.keys())

        print(f"  Total samples: {len(df)}")
        print(f"  Unique sequences: {len(unique_sequences)}")
        print(f"  Duplicate rate: {100 * (1 - len(unique_sequences) / len(df)):.2f}%")

        # Split unique sequences
        train_seqs, temp_seqs = train_test_split(
            unique_sequences,
            train_size=train_ratio,
            random_state=random_seed
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_seqs, test_seqs = train_test_split(
            temp_seqs,
            train_size=val_size,
            random_state=random_seed
        )

        # Map sequences back to indices
        train_indices = []
        for seq in train_seqs:
            train_indices.extend(sequence_groups[seq])

        val_indices = []
        for seq in val_seqs:
            val_indices.extend(sequence_groups[seq])

        test_indices = []
        for seq in test_seqs:
            test_indices.extend(sequence_groups[seq])

        # Convert to numpy arrays and sort for consistency
        train_indices = np.array(sorted(train_indices))
        val_indices = np.array(sorted(val_indices))
        test_indices = np.array(sorted(test_indices))

    else:
        # Original behavior: split by indices (may cause leakage)
        print("WARNING: Splitting by indices without deduplication (may cause data leakage)")
        n_samples = len(df)
        indices = np.arange(n_samples)

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_seed
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=random_seed
        )

    return train_indices, val_indices, test_indices


def get_data_loaders(csv_path, batch_size=32, num_workers=0, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42, deduplicate=True, augment=False, augment_prob=0.5, prefetch_factor=2, persistent_workers=False):
    """
    Create DataLoaders for train/val/test splits.

    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        deduplicate: If True, split by unique sequences to prevent leakage (default True)
        augment: If True, apply data augmentation to training set (default False)
        augment_prob: Probability of applying augmentation (default 0.5)
        prefetch_factor: Number of batches to prefetch per worker (default 2)
        persistent_workers: Keep workers alive between epochs (default False)

    Returns:
        tuple of (train_loader, val_loader, test_loader)
    """
    # Create splits
    train_indices, val_indices, test_indices = create_data_splits(
        csv_path, train_ratio, val_ratio, test_ratio, random_seed, deduplicate
    )

    # Create datasets (only augment training set)
    train_dataset = RNADataset(csv_path, train_indices, augment=augment, augment_prob=augment_prob)
    val_dataset = RNADataset(csv_path, val_indices, augment=False)
    test_dataset = RNADataset(csv_path, test_indices, augment=False)

    # DataLoader kwargs (only use prefetch_factor and persistent_workers if num_workers > 0)
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True
    }

    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    print(f"Dataset splits created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
