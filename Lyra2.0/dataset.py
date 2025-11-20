import json
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


# RNA nucleotide vocabulary with IUPAC ambiguity codes
# Multi-hot encoding: ambiguity codes represented as probability distributions
# A, U, G, C = standard bases
# R = purine (A or G), Y = pyrimidine (C or U)
# S = strong (G or C), W = weak (A or U), K = keto (G or U), M = amino (A or C)
# B = not A, D = not C, H = not G, V = not U
# N = any nucleotide
# Format: [A, U, G, C] probabilities
RNA_VOCAB = {
    'A': [1.0, 0.0, 0.0, 0.0],
    'U': [0.0, 1.0, 0.0, 0.0],
    'G': [0.0, 0.0, 1.0, 0.0],
    'C': [0.0, 0.0, 0.0, 1.0],
    'R': [0.5, 0.0, 0.5, 0.0],      # A or G (purine)
    'Y': [0.0, 0.5, 0.0, 0.5],      # U or C (pyrimidine)
    'S': [0.0, 0.0, 0.5, 0.5],      # G or C (strong)
    'W': [0.5, 0.5, 0.0, 0.0],      # A or U (weak)
    'K': [0.0, 0.5, 0.5, 0.0],      # G or U (keto)
    'M': [0.5, 0.0, 0.0, 0.5],      # A or C (amino)
    'B': [0.0, 1/3, 1/3, 1/3], # not A (U, G, C)
    'D': [1/3, 0.0, 1/3, 1/3], # not C (A, G, U)
    'H': [1/3, 1/3, 0.0, 1/3], # not G (A, U, C)
    'V': [1/3, 0.0, 1/3, 1/3], # not U (A, G, C)
    'N': [0.25, 0.25, 0.25, 0.25]   # any base
}
RNA_VOCAB_SIZE = 4  # Only standard bases A, U, G, C


class DatasetError(Exception):
    """Custom exception for dataset-related errors."""
    pass


def tokenize_sequence(sequence: str) -> torch.Tensor:
    """
    Convert RNA sequence string to multi-hot encoded tensor.

    Standard bases (A, U, G, C) are one-hot encoded.
    IUPAC ambiguity codes are represented as probability distributions.

    Args:
        sequence: RNA sequence string (e.g., "AUCG...")

    Returns:
        tokens: FloatTensor of shape (L, 4) where dim 4 = [A, U, G, C] probabilities
    """
    if not sequence:
        raise DatasetError("Empty sequence provided")

    if not isinstance(sequence, str):
        raise DatasetError(f"Sequence must be a string, got {type(sequence)}")

    # Check for invalid nucleotides
    invalid_chars = set(sequence.upper()) - set(RNA_VOCAB.keys())
    if invalid_chars:
        raise DatasetError(f"Invalid nucleotides in sequence: {invalid_chars}. Valid: {', '.join(sorted(RNA_VOCAB.keys()))}")

    # Convert each nucleotide to its multi-hot representation
    tokens = [RNA_VOCAB[nt] for nt in sequence.upper()]
    return torch.tensor(tokens, dtype=torch.float)


def one_hot_encode(sequence: str) -> torch.Tensor:
    """
    Convert RNA sequence string to multi-hot encoding.

    Standard bases (A, U, G, C) are one-hot encoded.
    IUPAC ambiguity codes are represented as probability distributions.

    Args:
        sequence: RNA sequence string (e.g., "AUCG...")

    Returns:
        encoded: FloatTensor of shape (L, 4) where dim 4 = [A, U, G, C] probabilities
    """
    # tokenize_sequence now returns the multi-hot representation directly
    return tokenize_sequence(sequence)


def parse_base_pairs(base_pairs_str: str) -> List[Tuple[int, int]]:
    """
    Parse base pairs string from CSV into list of tuples.

    Args:
        base_pairs_str: JSON string like "[[1, 116], [2, 115], ...]"

    Returns:
        List of (i, j) tuples representing paired positions
    """
    if not base_pairs_str or pd.isna(base_pairs_str):
        raise DatasetError("Empty or NaN base_pairs string")

    try:
        pairs = json.loads(base_pairs_str)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Failed to parse base_pairs JSON: {e}. Input: {base_pairs_str[:100]}")

    if not isinstance(pairs, list):
        raise DatasetError(f"base_pairs must be a list, got {type(pairs)}")

    result = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DatasetError(f"Invalid pair at index {idx}: {pair}. Expected [i, j]")
        try:
            i, j = int(pair[0]), int(pair[1])
            if i <= 0 or j <= 0:
                raise DatasetError(f"Invalid 1-based index in pair at index {idx}: [{i}, {j}]")
            # Convert from 1-based to 0-based indexing
            result.append((i - 1, j - 1))
        except (ValueError, TypeError) as e:
            raise DatasetError(f"Failed to convert pair at index {idx} to integers: {pair}. Error: {e}")

    return result


def create_contact_matrix(base_pairs: List[Tuple[int, int]], seq_length: int) -> torch.Tensor:
    """
    Create a contact matrix from base pairs.

    Args:
        base_pairs: List of (i, j) tuples
        seq_length: Length of the sequence

    Returns:
        contact: FloatTensor of shape (L, L) with 1s at paired positions
    """
    if seq_length <= 0:
        raise DatasetError(f"Invalid sequence length: {seq_length}. Must be positive.")

    contact = torch.zeros(seq_length, seq_length, dtype=torch.float)

    for i, j in base_pairs:
        if i >= seq_length or j >= seq_length:
            raise DatasetError(f"Base pair index out of bounds: [{i}, {j}] for sequence length {seq_length}")
        contact[i, j] = 1.0
        contact[j, i] = 1.0  # Symmetric

    return contact


def create_pairing_targets(base_pairs: List[Tuple[int, int]], seq_length: int) -> torch.Tensor:
    """
    Create per-position pairing targets.

    For each position, stores the index of its pairing partner, or seq_length if unpaired.

    Args:
        base_pairs: List of (i, j) tuples (0-indexed)
        seq_length: Length of the sequence

    Returns:
        targets: LongTensor of shape (L,) where targets[i] = j if i pairs with j,
                 or targets[i] = seq_length if i is unpaired
    """
    if seq_length <= 0:
        raise DatasetError(f"Invalid sequence length: {seq_length}. Must be positive.")

    # Initialize all positions as unpaired (class = seq_length)
    targets = torch.full((seq_length,), seq_length, dtype=torch.long)

    for i, j in base_pairs:
        if i >= seq_length or j >= seq_length:
            raise DatasetError(f"Base pair index out of bounds: [{i}, {j}] for sequence length {seq_length}")
        targets[i] = j
        targets[j] = i  # Symmetric

    return targets


def create_dotbracket_targets(structure: str) -> torch.Tensor:
    """
    Create dot-bracket targets from structure string.

    Args:
        structure: Dot-bracket notation string (e.g., "(((...)))")

    Returns:
        targets: LongTensor of shape (L,) where 0='.', 1='(', 2=')'
    """
    if not structure:
        raise DatasetError("Empty structure string provided")

    mapping = {'.': 0, '(': 1, ')': 2}

    # Validate structure characters
    invalid_chars = set(structure) - set(mapping.keys())
    if invalid_chars:
        raise DatasetError(f"Invalid characters in structure: {invalid_chars}. Expected: '.', '(', ')'")

    targets = torch.tensor([mapping[c] for c in structure], dtype=torch.long)
    return targets


def create_binary_pairing_targets(base_pairs: List[Tuple[int, int]], seq_length: int) -> torch.Tensor:
    """
    Create binary paired/unpaired targets.

    Args:
        base_pairs: List of (i, j) tuples (0-indexed)
        seq_length: Length of the sequence

    Returns:
        targets: LongTensor of shape (L,) where 0=unpaired, 1=paired
    """
    if seq_length <= 0:
        raise DatasetError(f"Invalid sequence length: {seq_length}. Must be positive.")

    targets = torch.zeros(seq_length, dtype=torch.long)

    for i, j in base_pairs:
        if i >= seq_length or j >= seq_length:
            raise DatasetError(f"Base pair index out of bounds: [{i}, {j}] for sequence length {seq_length}")
        targets[i] = 1
        targets[j] = 1

    return targets


def create_distance_targets(base_pairs: List[Tuple[int, int]], seq_length: int, max_distance: int = 150) -> torch.Tensor:
    """
    Create relative distance targets for pairing.

    For paired positions, stores the relative distance to partner.
    For unpaired positions, stores max_distance.

    Args:
        base_pairs: List of (i, j) tuples (0-indexed)
        seq_length: Length of the sequence
        max_distance: Maximum distance to consider (distances >= max_distance get clamped)

    Returns:
        targets: LongTensor of shape (L,) where targets[i] = |i - j| if i pairs with j,
                 or targets[i] = max_distance if unpaired
    """
    if seq_length <= 0:
        raise DatasetError(f"Invalid sequence length: {seq_length}. Must be positive.")

    # Initialize all as unpaired (class = max_distance)
    targets = torch.full((seq_length,), max_distance, dtype=torch.long)

    for i, j in base_pairs:
        if i >= seq_length or j >= seq_length:
            raise DatasetError(f"Base pair index out of bounds: [{i}, {j}] for sequence length {seq_length}")

        distance = abs(i - j)
        # Clamp to max_distance
        distance = min(distance, max_distance - 1)

        targets[i] = distance
        targets[j] = distance

    return targets


def create_all_targets(
    base_pairs: List[Tuple[int, int]],
    seq_length: int,
    structure: str,
    max_distance: int = 150
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create all target types in a single pass over base_pairs for efficiency.

    This avoids iterating over base_pairs multiple times, improving dataset loading speed.

    Args:
        base_pairs: List of (i, j) tuples (0-indexed)
        seq_length: Length of the sequence
        structure: Dot-bracket notation string
        max_distance: Maximum distance for distance targets

    Returns:
        Tuple of (pairing_targets, dotbracket_targets, binary_targets, distance_targets)
    """
    if seq_length <= 0:
        raise DatasetError(f"Invalid sequence length: {seq_length}. Must be positive.")

    # Initialize all targets
    pairing_targets = torch.full((seq_length,), seq_length, dtype=torch.long)
    binary_targets = torch.zeros(seq_length, dtype=torch.long)
    distance_targets = torch.full((seq_length,), max_distance, dtype=torch.long)

    # Single pass over base pairs to populate pairing, binary, and distance targets
    for i, j in base_pairs:
        if i >= seq_length or j >= seq_length:
            raise DatasetError(f"Base pair index out of bounds: [{i}, {j}] for sequence length {seq_length}")

        # Pairing targets
        pairing_targets[i] = j
        pairing_targets[j] = i

        # Binary targets
        binary_targets[i] = 1
        binary_targets[j] = 1

        # Distance targets
        distance = abs(i - j)
        distance = min(distance, max_distance - 1)
        distance_targets[i] = distance
        distance_targets[j] = distance

    # Dot-bracket targets (independent of base_pairs iteration)
    dotbracket_targets = create_dotbracket_targets(structure)

    return pairing_targets, dotbracket_targets, binary_targets, distance_targets


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA secondary structure prediction.

    Loads sequences and converts base pairs to contact matrices.
    """

    def __init__(
        self,
        csv_path: str,
        max_length: Optional[int] = None,
        transform=None
    ):
        """
        Args:
            csv_path: Path to the CSV file
            max_length: Maximum sequence length to include (None for all)
            transform: Optional transform to apply to sequences
        """
        self.transform = transform

        # Validate file path
        if not os.path.exists(csv_path):
            raise DatasetError(f"Dataset file not found: {csv_path}")

        # Load data
        print(f"Loading dataset from {csv_path}...")
        try:
            self.data = pd.read_csv(csv_path)
        except Exception as e:
            raise DatasetError(f"Failed to load CSV file: {csv_path}. Error: {e}")

        # Validate required columns
        required_columns = ['id', 'sequence', 'base_pairs', 'len']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise DatasetError(f"Missing required columns: {missing_columns}. Found: {list(self.data.columns)}")

        # Check for empty dataset
        if len(self.data) == 0:
            raise DatasetError(f"Dataset is empty: {csv_path}")

        # CRITICAL: Check if 'structure' column exists
        # If missing, dot-bracket targets will be all zeros, causing degenerate learning!
        if 'structure' not in self.data.columns:
            print("\n" + "=" * 80)
            print("CRITICAL WARNING: 'structure' column is MISSING from dataset!")
            print("=" * 80)
            print("This will cause the model to learn a trivial solution (all zeros).")
            print("Dot-bracket targets will be dummy values, causing loss to drop to 0.")
            print("\nSOLUTION:")
            print(f"  Run: python add_structure_column.py {csv_path}")
            print("=" * 80 + "\n")
            raise DatasetError(
                f"Missing 'structure' column in dataset! "
                f"Run: python add_structure_column.py {csv_path}"
            )

        # Check if structure column has valid entries
        non_null_structures = self.data['structure'].notna().sum()
        if non_null_structures == 0:
            print("\n" + "=" * 80)
            print("CRITICAL WARNING: 'structure' column is all NaN/empty!")
            print("=" * 80)
            print("This will cause the model to learn a trivial solution (all zeros).")
            print("\nSOLUTION:")
            print(f"  Run: python add_structure_column.py {csv_path}")
            print("=" * 80 + "\n")
            raise DatasetError(
                f"'structure' column is all NaN/empty! "
                f"Run: python add_structure_column.py {csv_path}"
            )

        if non_null_structures < len(self.data):
            print(f"WARNING: {len(self.data) - non_null_structures} rows have empty structure column")

        # Filter by length if specified
        if max_length is not None:
            original_len = len(self.data)
            self.data = self.data[self.data['len'] <= max_length].reset_index(drop=True)
            print(f"Filtered {original_len} -> {len(self.data)} sequences (max_length={max_length})")

            if len(self.data) == 0:
                raise DatasetError(f"No sequences remaining after filtering with max_length={max_length}")

        print(f"Loaded {len(self.data)} sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
        except IndexError:
            raise DatasetError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")

        # Get sequence and convert to one-hot
        sequence = row['sequence']
        if pd.isna(sequence):
            raise DatasetError(f"NaN sequence at index {idx}, id={row.get('id', 'unknown')}")

        try:
            x = one_hot_encode(sequence)
        except DatasetError as e:
            raise DatasetError(f"Error encoding sequence at index {idx}, id={row.get('id', 'unknown')}: {e}")

        # Parse base pairs
        try:
            base_pairs = parse_base_pairs(row['base_pairs'])
        except DatasetError as e:
            raise DatasetError(f"Error parsing base_pairs at index {idx}, id={row.get('id', 'unknown')}: {e}")

        seq_length = len(sequence)

        # Create all target types for multi-task learning in a single pass (optimized)
        try:
            # Get structure from row if available, otherwise use dummy
            if 'structure' in row and not pd.isna(row['structure']):
                structure = row['structure']
            else:
                # Fallback: create dummy structure (will be masked in collate_fn)
                structure = '.' * seq_length

            # Create all targets in single pass over base_pairs
            pairing_targets, dotbracket_targets, binary_targets, distance_targets = create_all_targets(
                base_pairs, seq_length, structure, max_distance=150
            )
        except DatasetError as e:
            raise DatasetError(f"Error creating targets at index {idx}, id={row.get('id', 'unknown')}: {e}")

        # Apply transform if specified
        if self.transform:
            try:
                x = self.transform(x)
            except Exception as e:
                raise DatasetError(f"Error applying transform at index {idx}: {e}")

        return {
            'sequence': x,                         # (L, 4) - multi-hot encoded [A, U, G, C]
            'pairing_targets': pairing_targets,    # (L,) - pairing partner index
            'dotbracket_targets': dotbracket_targets,  # (L,) - dot-bracket classes
            'binary_targets': binary_targets,      # (L,) - paired/unpaired binary
            'distance_targets': distance_targets,  # (L,) - relative distance to partner
            'length': seq_length,
            'id': row['id']
        }


def collate_fn(batch):
    """
    Collate function for variable-length sequences.

    Pads sequences and all target types to the maximum length in the batch.
    """
    if not batch:
        raise DatasetError("Empty batch received in collate_fn")

    # Find max length in batch
    lengths = [item['length'] for item in batch]
    if not lengths:
        raise DatasetError("No lengths found in batch items")

    max_len = max(lengths)
    if max_len <= 0:
        raise DatasetError(f"Invalid max length in batch: {max_len}")

    batch_size = len(batch)

    # Initialize padded tensors (use empty for sequences, full for targets/masks)
    # torch.empty doesn't initialize memory, so we'll explicitly zero padding later
    sequences = torch.zeros(batch_size, max_len, RNA_VOCAB_SIZE)  # Keep zeros for multi-hot padding

    # For all targets, use -100 as the "ignore" index for padded positions
    # -100 is ignored by CrossEntropyLoss
    pairing_targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
    dotbracket_targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
    binary_targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
    distance_targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)  # Keep zeros for False padding

    ids = []

    for i, item in enumerate(batch):
        L = item['length']

        # Copy sequence
        sequences[i, :L, :] = item['sequence']

        # Copy pairing targets and remap unpaired class from original seq_len to max_len
        # Original targets use L as unpaired index, we need to use max_len
        item_pairing_targets = item['pairing_targets']
        pairing_targets[i, :L] = torch.where(
            item_pairing_targets == L,  # was unpaired (original seq_len)
            max_len,                     # new unpaired index (batch max_len)
            item_pairing_targets         # keep pair indices unchanged
        )

        # Copy dot-bracket targets (no remapping needed)
        dotbracket_targets[i, :L] = item['dotbracket_targets']

        # Copy binary targets (no remapping needed)
        binary_targets[i, :L] = item['binary_targets']

        # Copy distance targets (no remapping needed)
        distance_targets[i, :L] = item['distance_targets']

        # Create mask (True for valid positions)
        masks[i, :L] = True

        ids.append(item['id'])

    return {
        'sequence': sequences,                    # (B, L_max, 4) - multi-hot [A, U, G, C]
        'pairing_targets': pairing_targets,       # (B, L_max) - partner index or max_len for unpaired, -100 for padding
        'dotbracket_targets': dotbracket_targets, # (B, L_max) - dot-bracket classes, -100 for padding
        'binary_targets': binary_targets,         # (B, L_max) - 0/1 binary, -100 for padding
        'distance_targets': distance_targets,     # (B, L_max) - distance or 150 for unpaired, -100 for padding
        'mask': masks,                            # (B, L_max)
        'length': torch.tensor(lengths),          # (B,)
        'id': ids
    }


def get_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create train/val/test dataloaders from a CSV file.

    Args:
        csv_path: Path to the CSV file
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction for training
        val_split: Fraction for validation (test = 1 - train - val)
        num_workers: Number of dataloader workers
        seed: Random seed for splitting

    Returns:
        train_loader, val_loader, test_loader
    """
    # Validate split fractions
    if train_split <= 0 or train_split >= 1:
        raise DatasetError(f"train_split must be in (0, 1), got {train_split}")
    if val_split < 0 or val_split >= 1:
        raise DatasetError(f"val_split must be in [0, 1), got {val_split}")
    if train_split + val_split >= 1:
        raise DatasetError(f"train_split + val_split must be < 1, got {train_split + val_split}")

    # Create full dataset
    full_dataset = RNADataset(csv_path, max_length=max_length)

    # Split dataset
    n_total = len(full_dataset)
    if n_total == 0:
        raise DatasetError("Dataset is empty, cannot create splits")

    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    if n_train == 0:
        raise DatasetError(f"Training set is empty with {n_total} samples and train_split={train_split}")
    if n_test == 0:
        raise DatasetError(f"Test set is empty. Adjust train_split or val_split.")

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    print(f"Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import sys

    csv_path = "dataset.csv"

    # Test loading a small sample
    dataset = RNADataset(csv_path, max_length=200)

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  ID: {sample['id']}")
        print(f"  Sequence shape: {sample['sequence'].shape}")
        print(f"  Contact shape: {sample['contact'].shape}")
        print(f"  Length: {sample['length']}")
        print(f"  Num base pairs: {sample['contact'].sum().item() / 2}")

        # Test dataloader
        train_loader, val_loader, test_loader = get_dataloaders(
            csv_path,
            batch_size=4,
            max_length=200
        )

        batch = next(iter(train_loader))
        print(f"\nBatch:")
        print(f"  Sequence shape: {batch['sequence'].shape}")
        print(f"  Contact shape: {batch['contact'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Lengths: {batch['length']}")
