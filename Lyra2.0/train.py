import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

from model import LyraContact, get_optimizer_groups
from dataset import get_dataloaders, RNA_VOCAB_SIZE


def compute_metrics(logits, targets, mask, lengths):
    """
    Compute evaluation metrics for per-position pairing prediction.

    Args:
        logits: (B, L, L+1) predicted logits
        targets: (B, L) ground truth pairing partners (or L_max for unpaired)
        mask: (B, L) sequence mask
        lengths: (B,) sequence lengths

    Returns:
        Dictionary of metrics
    """
    B, L_max, num_classes = logits.shape
    unpaired_idx = L_max  # unpaired class is at index L_max (last class)

    # Get predictions (argmax over classes)
    predictions = logits.argmax(dim=-1)  # (B, L_max)

    # Overall accuracy (including unpaired)
    correct = (predictions == targets) & mask
    total_positions = mask.sum()
    accuracy = correct.sum().float() / total_positions

    # Metrics for paired positions only (excluding unpaired class)
    # Unpaired class is now consistently at index L_max for all sequences
    paired_mask = mask & (targets != unpaired_idx)

    if paired_mask.sum() > 0:
        # Among paired positions, how many did we get right?
        paired_correct = (predictions == targets) & paired_mask
        paired_accuracy = paired_correct.sum().float() / paired_mask.sum()

        # Precision: of positions we predicted as paired, how many are correct?
        # Recall: of actually paired positions, how many did we find?
        predicted_paired = mask & (predictions != unpaired_idx)

        tp = ((predictions == targets) & paired_mask).sum().float()
        fp = (predicted_paired & ~paired_mask & mask).sum().float()
        fn = (~(predictions == targets) & paired_mask).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        paired_accuracy = torch.tensor(0.0)
        precision = torch.tensor(0.0)
        recall = torch.tensor(0.0)
        f1 = torch.tensor(0.0)

    # Sequence-level accuracy: entire sequence must be 100% correct
    seq_correct = 0
    for b in range(B):
        seq_len = lengths[b].item()
        seq_pred = predictions[b, :seq_len]
        seq_target = targets[b, :seq_len]
        if (seq_pred == seq_target).all():
            seq_correct += 1
    seq_accuracy = seq_correct / B

    return {
        'pos_accuracy': accuracy.item(),      # Per-position accuracy
        'seq_accuracy': seq_accuracy,         # Sequence-level accuracy (100% correct)
        'paired_accuracy': paired_accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
    }


def compute_multitask_loss(outputs, batch, weights=None, debug=False):
    """
    Multi-task loss combining dot-bracket, binary, and pairing losses.

    Args:
        outputs: Dictionary with keys 'dotbracket', 'binary', 'pairing'
                 - dotbracket: (B, L, 3)
                 - binary: (B, L, 2)
                 - pairing: (B, L, L+1)
        batch: Dictionary with target tensors
        weights: Dictionary of task weights (default: {'dotbracket': 0.5, 'binary': 0.2, 'pairing': 0.3})
        debug: If True, print debugging information about targets

    Returns:
        Dictionary with 'total' loss and individual task losses
    """
    if weights is None:
        weights = {
            'dotbracket': 0.5,  # Primary task
            'binary': 0.2,      # Auxiliary task
            'pairing': 0.3      # Auxiliary task
        }

    losses = {}

    # Debug: Check for degenerate targets
    if debug:
        dotbracket_targets = batch['dotbracket_targets']
        valid_targets = dotbracket_targets[dotbracket_targets != -100]
        if len(valid_targets) > 0:
            unique_vals = torch.unique(valid_targets)
            print(f"\n[DEBUG] Dotbracket target analysis:")
            print(f"  Valid targets: {len(valid_targets)}")
            print(f"  Unique values: {unique_vals.tolist()}")
            for val in unique_vals:
                count = (valid_targets == val).sum().item()
                pct = 100 * count / len(valid_targets)
                label = {0: '.', 1: '(', 2: ')'}[val.item()] if val.item() in {0, 1, 2} else '?'
                print(f"    Class {val.item()} ({label}): {count} ({pct:.1f}%)")

            if len(unique_vals) == 1 and unique_vals[0] == 0:
                print(f"  WARNING: All targets are class 0 (dot)!")
                print(f"  This indicates the 'structure' column is missing from CSV!")
                print(f"  Run: python add_structure_column.py dataset.csv")

    # Loss 1: Dot-bracket prediction (3 classes)
    dotbracket_logits = outputs['dotbracket']  # (B, L, 3)
    dotbracket_targets = batch['dotbracket_targets']  # (B, L)

    dotbracket_logits_flat = dotbracket_logits.view(-1, 3)
    dotbracket_targets_flat = dotbracket_targets.view(-1)

    losses['dotbracket'] = F.cross_entropy(
        dotbracket_logits_flat,
        dotbracket_targets_flat,
        ignore_index=-100
    )

    # Loss 2: Binary pairing prediction (2 classes)
    binary_logits = outputs['binary']  # (B, L, 2)
    binary_targets = batch['binary_targets']  # (B, L)

    binary_logits_flat = binary_logits.view(-1, 2)
    binary_targets_flat = binary_targets.view(-1)

    losses['binary'] = F.cross_entropy(
        binary_logits_flat,
        binary_targets_flat,
        ignore_index=-100
    )

    # Loss 3: Pairing partner prediction (L+1 classes)
    pairing_logits = outputs['pairing']  # (B, L, L+1)
    pairing_targets = batch['pairing_targets']  # (B, L)

    B, L, num_classes = pairing_logits.shape
    pairing_logits_flat = pairing_logits.view(-1, num_classes)
    pairing_targets_flat = pairing_targets.view(-1)

    losses['pairing'] = F.cross_entropy(
        pairing_logits_flat,
        pairing_targets_flat,
        ignore_index=-100
    )

    # Weighted combination
    losses['total'] = (
        weights['dotbracket'] * losses['dotbracket'] +
        weights['binary'] * losses['binary'] +
        weights['pairing'] * losses['pairing']
    )

    return losses


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, log_wandb=True):
    """Train for one epoch with multi-task learning."""
    model.train()

    total_loss = 0
    total_dotbracket_loss = 0
    total_binary_loss = 0
    total_pairing_loss = 0
    total_samples = 0
    all_metrics = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for step, batch in enumerate(pbar):
        # Move batch to device (non_blocking for async transfer with pinned memory)
        batch_device = {
            'sequence': batch['sequence'].to(device, non_blocking=True),
            'pairing_targets': batch['pairing_targets'].to(device, non_blocking=True),
            'dotbracket_targets': batch['dotbracket_targets'].to(device, non_blocking=True),
            'binary_targets': batch['binary_targets'].to(device, non_blocking=True),
            'distance_targets': batch['distance_targets'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
            'length': batch['length'].to(device, non_blocking=True)
        }

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_device['sequence'])

        # Compute multi-task loss (debug first batch of first epoch)
        debug_mode = (epoch == 1 and step == 0)
        losses = compute_multitask_loss(outputs, batch_device, debug=debug_mode)
        loss = losses['total']

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        batch_size = len(batch['sequence'])
        total_loss += loss.item() * batch_size
        total_dotbracket_loss += losses['dotbracket'].item() * batch_size
        total_binary_loss += losses['binary'].item() * batch_size
        total_pairing_loss += losses['pairing'].item() * batch_size
        total_samples += batch_size

        # Compute batch metrics (using pairing head for main metrics)
        with torch.no_grad():
            metrics = compute_metrics(
                outputs['pairing'],
                batch_device['pairing_targets'],
                batch_device['mask'],
                batch_device['length']
            )
            all_metrics.append(metrics)

        # Log to wandb every step
        if log_wandb and wandb.run is not None:
            wandb.log({
                'train/step_loss': loss.item(),
                'train/step_dotbracket_loss': losses['dotbracket'].item(),
                'train/step_binary_loss': losses['binary'].item(),
                'train/step_pairing_loss': losses['pairing'].item(),
                'train/step_f1': metrics['f1'],
                'train/step_pos_accuracy': metrics['pos_accuracy'],
                'train/step_seq_accuracy': metrics['seq_accuracy'],
                'train/step_precision': metrics['precision'],
                'train/step_recall': metrics['recall'],
                'train/grad_norm': grad_norm.item(),
                'epoch': epoch
            })

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'db': f"{losses['dotbracket'].item():.3f}",
            'bin': f"{losses['binary'].item():.3f}",
            'pair': f"{losses['pairing'].item():.3f}",
            'f1': f"{metrics['f1']:.4f}"
        })

    # Step scheduler
    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
        scheduler.step()

    # Average metrics
    avg_loss = total_loss / total_samples
    avg_dotbracket_loss = total_dotbracket_loss / total_samples
    avg_binary_loss = total_binary_loss / total_samples
    avg_pairing_loss = total_pairing_loss / total_samples

    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = avg_loss
    avg_metrics['dotbracket_loss'] = avg_dotbracket_loss
    avg_metrics['binary_loss'] = avg_binary_loss
    avg_metrics['pairing_loss'] = avg_pairing_loss

    # Log epoch averages to wandb
    if log_wandb and wandb.run is not None:
        wandb.log({
            'train/epoch_loss': avg_loss,
            'train/epoch_dotbracket_loss': avg_dotbracket_loss,
            'train/epoch_binary_loss': avg_binary_loss,
            'train/epoch_pairing_loss': avg_pairing_loss,
            'train/epoch_f1': avg_metrics['f1'],
            'train/epoch_pos_accuracy': avg_metrics['pos_accuracy'],
            'train/epoch_seq_accuracy': avg_metrics['seq_accuracy'],
            'train/epoch_paired_accuracy': avg_metrics['paired_accuracy'],
            'train/epoch_precision': avg_metrics['precision'],
            'train/epoch_recall': avg_metrics['recall'],
            'learning_rate': current_lr,
            'epoch': epoch
        })

    return avg_metrics


@torch.no_grad()
def evaluate(model, data_loader, device, desc="Eval", epoch=None, log_wandb=True, split_name='val'):
    """Evaluate the model with multi-task outputs."""
    model.eval()

    total_loss = 0
    total_dotbracket_loss = 0
    total_binary_loss = 0
    total_pairing_loss = 0
    total_samples = 0
    all_metrics = []

    pbar = tqdm(data_loader, desc=desc)

    for batch in pbar:
        # Move batch to device (non_blocking for async transfer with pinned memory)
        batch_device = {
            'sequence': batch['sequence'].to(device, non_blocking=True),
            'pairing_targets': batch['pairing_targets'].to(device, non_blocking=True),
            'dotbracket_targets': batch['dotbracket_targets'].to(device, non_blocking=True),
            'binary_targets': batch['binary_targets'].to(device, non_blocking=True),
            'distance_targets': batch['distance_targets'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
            'length': batch['length'].to(device, non_blocking=True)
        }

        # Forward pass
        outputs = model(batch_device['sequence'])

        # Compute multi-task loss
        losses = compute_multitask_loss(outputs, batch_device)
        loss = losses['total']

        # Track metrics
        batch_size = len(batch['sequence'])
        total_loss += loss.item() * batch_size
        total_dotbracket_loss += losses['dotbracket'].item() * batch_size
        total_binary_loss += losses['binary'].item() * batch_size
        total_pairing_loss += losses['pairing'].item() * batch_size
        total_samples += batch_size

        # Compute metrics (using pairing head for main metrics)
        metrics = compute_metrics(
            outputs['pairing'],
            batch_device['pairing_targets'],
            batch_device['mask'],
            batch_device['length']
        )
        all_metrics.append(metrics)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'seq_acc': f"{metrics['seq_accuracy']:.4f}",
            'f1': f"{metrics['f1']:.4f}"
        })

    # Average metrics
    avg_loss = total_loss / total_samples
    avg_dotbracket_loss = total_dotbracket_loss / total_samples
    avg_binary_loss = total_binary_loss / total_samples
    avg_pairing_loss = total_pairing_loss / total_samples

    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = avg_loss
    avg_metrics['dotbracket_loss'] = avg_dotbracket_loss
    avg_metrics['binary_loss'] = avg_binary_loss
    avg_metrics['pairing_loss'] = avg_pairing_loss

    # Log to wandb
    if log_wandb and wandb.run is not None and epoch is not None:
        wandb.log({
            f'{split_name}/loss': avg_loss,
            f'{split_name}/dotbracket_loss': avg_dotbracket_loss,
            f'{split_name}/binary_loss': avg_binary_loss,
            f'{split_name}/pairing_loss': avg_pairing_loss,
            f'{split_name}/f1': avg_metrics['f1'],
            f'{split_name}/pos_accuracy': avg_metrics['pos_accuracy'],
            f'{split_name}/seq_accuracy': avg_metrics['seq_accuracy'],
            f'{split_name}/paired_accuracy': avg_metrics['paired_accuracy'],
            f'{split_name}/precision': avg_metrics['precision'],
            f'{split_name}/recall': avg_metrics['recall'],
            'epoch': epoch
        })

    return avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint with error handling."""
    try:
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        torch.save(checkpoint, path)

    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device)

        # Validate checkpoint contents
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'metrics']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        # Load state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['metrics']

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")


def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description="Train Lyra for RNA contact prediction")

    # Data arguments
    parser.add_argument('--data_path', type=str, default='dataset.csv',
                        help='Path to the dataset CSV')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')

    # Model arguments
    parser.add_argument('--model_dim', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--d_state', type=int, default=64,
                        help='S4D state dimension')
    parser.add_argument('--num_s4', type=int, default=4,
                        help='Number of S4D layers')
    parser.add_argument('--pgc_expansion', type=float, default=2.0,
                        help='PGC expansion factor')
    parser.add_argument('--num_pgc', type=int, default=1,
                        help='Number of PGC layers')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='lyra-rna-contact-prediction',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='rna-ml',
                        help='Wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate arguments
    if args.batch_size <= 0:
        print(f"ERROR: batch_size must be positive, got {args.batch_size}")
        return 1

    if args.epochs <= 0:
        print(f"ERROR: epochs must be positive, got {args.epochs}")
        return 1

    if args.lr <= 0:
        print(f"ERROR: learning rate must be positive, got {args.lr}")
        return 1

    if not os.path.exists(args.data_path):
        print(f"ERROR: Dataset file not found: {args.data_path}")
        print(f"Please check the file path and try again.")
        return 1

    # Create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory {args.output_dir}: {e}")
        return 1

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            # Create run name if not specified
            if args.wandb_run_name is None:
                run_name = f"lyra_dim{args.model_dim}_s4-{args.num_s4}_bs{args.batch_size}_lr{args.lr}"
            else:
                run_name = args.wandb_run_name

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    # Data config
                    'data_path': args.data_path,
                    'max_length': args.max_length,
                    'batch_size': args.batch_size,
                    # Model config
                    'model_dim': args.model_dim,
                    'd_state': args.d_state,
                    'num_s4': args.num_s4,
                    'pgc_expansion': args.pgc_expansion,
                    'num_pgc': args.num_pgc,
                    'dropout': args.dropout,
                    # Training config
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'seed': args.seed,
                    # Architecture
                    'architecture': 'multi-task',
                    'loss_weights': {
                        'dotbracket': 0.5,
                        'binary': 0.2,
                        'pairing': 0.3
                    }
                }
            )
            print(f"Initialized wandb: {args.wandb_entity}/{args.wandb_project}/{run_name}")
        except Exception as e:
            print(f"ERROR: Failed to initialize wandb: {e}")
            print(f"Try running 'wandb login' or use --no_wandb flag")
            return 1
    else:
        print("Wandb logging disabled")

    # Load data
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            seed=args.seed
        )

        # Validate dataloaders
        if len(train_loader) == 0:
            print(f"ERROR: Training set is empty!")
            return 1
        if len(val_loader) == 0:
            print(f"ERROR: Validation set is empty!")
            return 1
        if len(test_loader) == 0:
            print(f"ERROR: Test set is empty!")
            return 1

        print(f"Loaded {len(train_loader)} training batches")
        print(f"Loaded {len(val_loader)} validation batches")
        print(f"Loaded {len(test_loader)} test batches")

    except FileNotFoundError as e:
        print(f"ERROR: Dataset file not found: {e}")
        return 1
    except pd.errors.EmptyDataError:
        print(f"ERROR: Dataset file is empty: {args.data_path}")
        return 1
    except KeyError as e:
        print(f"ERROR: Missing required column in dataset: {e}")
        print(f"Required columns: id, sequence, structure, base_pairs, len")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        print(f"Try running: python diagnose_data.py {args.data_path}")
        return 1

    # Create model
    print("\nCreating model...")
    try:
        model = LyraContact(
            model_dimension=args.model_dim,
            pgc_configs=[(args.pgc_expansion, args.num_pgc)],
            num_s4=args.num_s4,
            d_input=RNA_VOCAB_SIZE,  # RNA: A, U, G, C, N
            d_state=args.d_state,
            dropout=args.dropout
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"ERROR: Out of GPU memory. Try reducing --model_dim or --batch_size")
            print(f"Current settings: model_dim={args.model_dim}, batch_size={args.batch_size}")
        else:
            print(f"ERROR: Failed to create model: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        return 1

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Log model to wandb
    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)
        wandb.config.update({'num_parameters': num_params})

    # Create optimizer with custom parameter groups
    param_groups = get_optimizer_groups(model, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdamW(param_groups)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_f1 = 0

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint file not found: {args.resume}")
            return 1

        try:
            print(f"\nResuming from {args.resume}")
            start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume, device)
            best_f1 = metrics.get('val_f1', 0)
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch-1}, best F1: {best_f1:.4f}")
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            return 1

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()

            # Train
            try:
                train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, log_wandb=use_wandb)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nERROR: Out of GPU memory during training at epoch {epoch}")
                    print(f"Try reducing --batch_size (current: {args.batch_size})")
                    return 1
                else:
                    raise

            # Check for NaN/Inf in training metrics
            if not np.isfinite(train_metrics['loss']):
                print(f"\nERROR: Training loss became NaN/Inf at epoch {epoch}")
                print(f"Last valid loss: {train_metrics.get('loss', 'N/A')}")
                print(f"Try reducing --lr (current: {args.lr})")
                return 1

            # Validate
            try:
                val_metrics = evaluate(model, val_loader, device, f"Epoch {epoch} [Val]", epoch=epoch, log_wandb=use_wandb, split_name='val')
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nERROR: Out of GPU memory during validation at epoch {epoch}")
                    print(f"Try reducing --batch_size (current: {args.batch_size})")
                    return 1
                else:
                    raise

            elapsed = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary ({elapsed:.1f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Seq Acc: {train_metrics['seq_accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Seq Acc: {val_metrics['seq_accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                try:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        {'val_f1': best_f1, **val_metrics},
                        os.path.join(args.output_dir, 'best_model.pt')
                    )
                    print(f"  -> New best model! F1: {best_f1:.4f}")
                except Exception as e:
                    print(f"WARNING: Failed to save best model checkpoint: {e}")

                # Log best metrics to wandb
                if use_wandb:
                    try:
                        wandb.run.summary['best_val_f1'] = best_f1
                        wandb.run.summary['best_val_epoch'] = epoch
                        wandb.run.summary['best_val_loss'] = val_metrics['loss']
                        wandb.run.summary['best_val_seq_accuracy'] = val_metrics['seq_accuracy']
                    except Exception as e:
                        print(f"WARNING: Failed to log to wandb: {e}")

            # Save latest checkpoint
            try:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'val_f1': val_metrics['f1'], **val_metrics},
                    os.path.join(args.output_dir, 'latest_model.pt')
                )
            except Exception as e:
                print(f"WARNING: Failed to save latest checkpoint: {e}")

            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Saving checkpoint at epoch {epoch}...")
        try:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_f1': val_metrics.get('f1', 0), **val_metrics} if 'val_metrics' in locals() else {},
                os.path.join(args.output_dir, 'interrupted_model.pt')
            )
            print(f"Checkpoint saved to {args.output_dir}/interrupted_model.pt")
        except Exception as e:
            print(f"ERROR: Failed to save interrupted checkpoint: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: Training failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Final evaluation on test set
    print("\nEvaluating on test set...")

    try:
        # Load best model
        best_model_path = os.path.join(args.output_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            print(f"WARNING: Best model checkpoint not found at {best_model_path}")
            print(f"Using current model state for test evaluation")
        else:
            load_checkpoint(model, optimizer, scheduler, best_model_path, device)
            print(f"Loaded best model from {best_model_path}")

        test_metrics = evaluate(model, test_loader, device, "Test", epoch=args.epochs, log_wandb=use_wandb, split_name='test')

        print("\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Seq Accuracy: {test_metrics['seq_accuracy']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")

        # Log test metrics to wandb summary
        if use_wandb:
            try:
                wandb.run.summary['test_loss'] = test_metrics['loss']
                wandb.run.summary['test_f1'] = test_metrics['f1']
                wandb.run.summary['test_seq_accuracy'] = test_metrics['seq_accuracy']
                wandb.run.summary['test_pos_accuracy'] = test_metrics['pos_accuracy']
                wandb.run.summary['test_precision'] = test_metrics['precision']
                wandb.run.summary['test_recall'] = test_metrics['recall']

                # Finish wandb run
                wandb.finish()
                print("\nWandb run finished!")
            except Exception as e:
                print(f"WARNING: Failed to log test metrics to wandb: {e}")

    except Exception as e:
        print(f"ERROR: Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nTraining completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
