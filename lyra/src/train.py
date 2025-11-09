import os
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from .model import create_lyra_model
from .data_loader import get_data_loaders
from .wandb_logging import WandBLogger
from .focal_loss import FocalLoss, CombinedLoss, WeightedBCEWithLogitsLoss
from .evaluate import evaluate_model


class MetricsCalculator:
    """Calculate evaluation metrics for contact map prediction"""

    @staticmethod
    def calculate_metrics(pred_contacts, true_contacts, seq_lens, threshold=0.5):
        """
        Calculate F1, MCC, and exact match accuracy.

        Args:
            pred_contacts: Predicted contact maps (B, L, L)
            true_contacts: True contact maps (B, L, L)
            seq_lens: Actual sequence lengths (B,)
            threshold: Threshold for binary prediction

        Returns:
            dict with metrics: f1, mcc, exact_match
        """
        batch_size = pred_contacts.shape[0]

        # Binarize predictions
        pred_binary = (pred_contacts > threshold).float()

        all_preds = []
        all_trues = []
        exact_matches = 0

        for i in range(batch_size):
            seq_len = seq_lens[i].item()

            # Extract valid region (upper triangle to avoid double counting)
            pred_valid = pred_binary[i, :seq_len, :seq_len].triu(diagonal=1)
            true_valid = true_contacts[i, :seq_len, :seq_len].triu(diagonal=1)

            # Flatten
            pred_flat = pred_valid.flatten().cpu().numpy()
            true_flat = true_valid.flatten().cpu().numpy()

            all_preds.extend(pred_flat)
            all_trues.extend(true_flat)

            # Check exact match
            if np.array_equal(pred_flat, true_flat):
                exact_matches += 1

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        f1 = f1_score(all_trues, all_preds, zero_division=0)
        mcc = matthews_corrcoef(all_trues, all_preds)
        exact_match = exact_matches / batch_size

        return {
            'f1': f1,
            'mcc': mcc,
            'exact_match': exact_match
        }


class Trainer:
    """Trainer class for Lyra RNA contact prediction"""

    def __init__(self, config, wandb_logger=None):
        self.config = config
        self.wandb_logger = wandb_logger
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Create model
        self.model = create_lyra_model(config['model']).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created with {num_params} parameters")

        # Loss function - choose based on config
        loss_type = config['training'].get('loss_type', 'focal')

        if loss_type == 'focal':
            # Focal Loss - best for extreme class imbalance
            focal_alpha = config['training'].get('focal_alpha', 0.25)
            focal_gamma = config['training'].get('focal_gamma', 2.0)
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif loss_type == 'combined':
            # Combined Focal + Dice Loss
            focal_alpha = config['training'].get('focal_alpha', 0.25)
            focal_gamma = config['training'].get('focal_gamma', 2.0)
            dice_weight = config['training'].get('dice_weight', 0.3)
            self.criterion = CombinedLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma, dice_weight=dice_weight)
            print(f"Using Combined Focal+Dice Loss (alpha={focal_alpha}, gamma={focal_gamma}, dice_weight={dice_weight})")
        else:
            # BCE with pos_weight (original)
            self.pos_weight = torch.tensor([config['training']['pos_weight']]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')
            print(f"Using BCE Loss (pos_weight={config['training']['pos_weight']})")

        self.loss_type = loss_type

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Warmup configuration
        self.warmup_epochs = config['training'].get('warmup_epochs', 0)
        self.base_lr = config['training']['learning_rate']
        self.current_epoch = 0

        # Performance optimizations
        self.use_amp = config['training'].get('use_amp', False)
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) for faster training")
        if self.gradient_accumulation_steps > 1:
            print(f"Using gradient accumulation: {self.gradient_accumulation_steps} steps")

        # Learning rate scheduler
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config['mode'],
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            self.scheduler = None

        # Metrics
        self.metrics_calc = MetricsCalculator()

        # Tracking
        self.best_val_metric = -float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Log model architecture to wandb
        if self.wandb_logger and self.wandb_logger.is_enabled:
            self.wandb_logger.log_model_info(
                self.model, self.criterion, num_params, self.device
            )

    def update_learning_rate_warmup(self, epoch):
        """Update learning rate during warmup period"""
        if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            # Linear warmup
            warmup_factor = epoch / self.warmup_epochs
            lr = self.base_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            sequences = batch['sequence'].to(self.device)
            contact_matrices = batch['contact_matrix'].to(self.device)
            seq_lens = batch['seq_len'].to(self.device)

            # Create mask for valid positions (exclude padding) - OUTSIDE autocast
            max_len = contact_matrices.shape[1]
            mask = torch.zeros_like(contact_matrices)
            for i, length in enumerate(seq_lens):
                mask[i, :length, :length] = 1.0

            # Forward pass with AMP if enabled
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_contacts = self.model(sequences, seq_lens)

                # Calculate masked loss (only on valid positions)
                # Standardized approach: compute loss first, then mask
                loss_unreduced = self.criterion(pred_contacts, contact_matrices)
                # Apply mask and average (consistent for all loss types)
                loss = (loss_unreduced * mask).sum() / (mask.sum() + 1e-8)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only update weights every N batches (gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = 0
                if self.config['training']['grad_clip'] > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
            else:
                grad_norm = 0

            total_loss += loss.item() * self.gradient_accumulation_steps
            batch_count += 1

            # Update progress bar with detailed info
            current_loss = loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg': f"{total_loss/batch_count:.4f}"
            })

            # Log to wandb per batch
            if self.wandb_logger and self.wandb_logger.is_enabled:
                self.wandb_logger.log_batch_metrics(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_batches=len(train_loader),
                    loss=loss.item() * self.gradient_accumulation_steps,
                    grad_norm=grad_norm,
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )

            # Log interval
            if self.config['logging']['verbose'] and batch_idx % self.config['logging']['log_interval'] == 0:
                avg_loss = total_loss / batch_count
                print(f"\nBatch {batch_idx}/{len(train_loader)} - Loss: {avg_loss:.4f}")

        avg_loss = total_loss / batch_count
        return avg_loss

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        batch_count = 0

        all_metrics = {'f1': [], 'mcc': [], 'exact_match': []}

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                sequences = batch['sequence'].to(self.device)
                contact_matrices = batch['contact_matrix'].to(self.device)
                seq_lens = batch['seq_len'].to(self.device)

                # Create mask for valid positions (exclude padding) - OUTSIDE autocast
                max_len = contact_matrices.shape[1]
                mask = torch.zeros_like(contact_matrices)
                for i, length in enumerate(seq_lens):
                    mask[i, :length, :length] = 1.0

                # Forward pass with AMP if enabled
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred_contacts = self.model(sequences, seq_lens)

                    # Calculate masked loss (only on valid positions)
                    # Calculate masked loss (standardized approach)
                    loss_unreduced = self.criterion(pred_contacts, contact_matrices)
                    # Apply mask and average (consistent for all loss types)
                    loss = (loss_unreduced * mask).sum() / (mask.sum() + 1e-8)

                total_loss += loss.item()
                batch_count += 1

                # Calculate metrics (apply sigmoid for probabilities)
                pred_probs = torch.sigmoid(pred_contacts)
                metrics = self.metrics_calc.calculate_metrics(
                    pred_probs, contact_matrices, seq_lens
                )

                for key in all_metrics:
                    all_metrics[key].append(metrics[key])

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'f1': f"{metrics['f1']:.4f}"
                })

        # Average metrics
        avg_loss = total_loss / batch_count
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        # Log standard deviations for metrics
        std_metrics = {key: np.std(values) for key, values in all_metrics.items()}

        return avg_loss, avg_metrics, std_metrics

    def save_checkpoint(self, epoch, val_metric, is_best=False, periodic=False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpoint']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric,
            'config': self.config
        }

        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            print(f"  Best model saved to {path}")

        # Save periodic checkpoint with epoch number
        if periodic:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            print(f"  Checkpoint saved to {path}")

        # Save latest checkpoint
        path = os.path.join(checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, path)

    def save_training_history(self):
        """Save training history to CSV file"""
        checkpoint_dir = self.config['checkpoint']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        history_file = os.path.join(checkpoint_dir, self.config['checkpoint']['history_file'])
        
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 
                'val_f1', 'val_mcc', 'val_exact_match', 'learning_rate'
            ])
            
            # Write data for each epoch
            for epoch in range(len(self.train_losses)):
                lr = self.optimizer.param_groups[0]['lr']
                writer.writerow([
                    epoch + 1,
                    f"{self.train_losses[epoch]:.6f}",
                    f"{self.val_losses[epoch]:.6f}",
                    f"{self.val_metrics[epoch]['f1']:.6f}",
                    f"{self.val_metrics[epoch]['mcc']:.6f}",
                    f"{self.val_metrics[epoch]['exact_match']:.6f}",
                    f"{lr:.8f}"
                ])
        
        print(f"\nTraining history saved to {history_file}")

    def train(self, train_loader, val_loader):
        """Full training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Monitor metric: {self.config['checkpoint']['monitor']}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 60}")

            # Apply warmup if in warmup period
            in_warmup = self.update_learning_rate_warmup(epoch)
            if in_warmup:
                print(f"  Warmup: LR = {self.optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics, val_std_metrics = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f} ± {val_std_metrics['f1']:.4f}")
            print(f"  Val MCC:    {val_metrics['mcc']:.4f} ± {val_std_metrics['mcc']:.4f}")
            print(f"  Val Exact:  {val_metrics['exact_match']:.4f} ± {val_std_metrics['exact_match']:.4f}")

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch metrics to wandb
            if self.wandb_logger and self.wandb_logger.is_enabled:
                self.wandb_logger.log_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    val_std_metrics=val_std_metrics,
                    learning_rate=current_lr,
                    epochs_without_improvement=self.epochs_without_improvement
                )

            # Check for improvement
            monitor_metric = self.config['checkpoint']['monitor']
            current_metric = val_metrics[monitor_metric.replace('val_', '')]

            is_best = current_metric > self.best_val_metric
            if is_best:
                prev_best = self.best_val_metric
                self.best_val_metric = current_metric
                self.epochs_without_improvement = 0
                print(f"  New best {monitor_metric}: {current_metric:.4f} (previous: {prev_best:.4f})")
                
                # Log best metric to wandb
                if self.wandb_logger and self.wandb_logger.is_enabled:
                    self.wandb_logger.log_best_metric(monitor_metric, current_metric, epoch)
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")

            # Save checkpoint
            save_interval = self.config['checkpoint']['save_interval']
            is_periodic = (epoch % save_interval == 0)
            
            if self.config['checkpoint']['save_best_only']:
                if is_best:
                    self.save_checkpoint(epoch, current_metric, is_best=True)
            else:
                # Save periodic checkpoints and best model
                self.save_checkpoint(epoch, current_metric, is_best=is_best, periodic=is_periodic)

            # Learning rate scheduler (only apply after warmup)
            if self.scheduler is not None and not in_warmup:
                prev_lr = current_lr
                self.scheduler.step(current_metric)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")

                # Log LR change to wandb
                if self.wandb_logger and self.wandb_logger.is_enabled and current_lr != prev_lr:
                    self.wandb_logger.log_lr_reduction(epoch, current_lr)
            elif not in_warmup:
                print(f"  Learning Rate: {current_lr:.6f}")

            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                if self.wandb_logger and self.wandb_logger.is_enabled:
                    self.wandb_logger.log_early_stopping(epoch, epoch * len(train_loader))
                break

        print(f"\nTraining completed!")
        print(f"Best {monitor_metric}: {self.best_val_metric:.4f}")
        
        # Log final training statistics
        if self.wandb_logger and self.wandb_logger.is_enabled:
            self.wandb_logger.log_training_completion(
                len(self.train_losses), 
                len(self.train_losses) * len(train_loader)
            )
        
        # Save training history to CSV
        self.save_training_history()


def main():
    """Main training function"""
    # Check CUDA availability
    print("=" * 60)
    print("CUDA/GPU Information:")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Training will use CPU.")
        print("To use GPU, ensure you have:")
        print("  1. A CUDA-capable GPU")
        print("  2. PyTorch installed with CUDA support")
        print("  3. Compatible CUDA drivers installed")
    print("=" * 60)
    print()
    
    # Load configuration
    config_path = os.path.join('lyra', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Initialize wandb logger
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', True)
    wandb_logger = WandBLogger(config, enabled=use_wandb)

    # Create data loaders
    print("\nPreparing data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path=config['data']['csv_path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['data']['random_seed'],
        deduplicate=config['dataloader'].get('deduplicate', True),
        augment=config['dataloader'].get('augment', False),
        augment_prob=config['dataloader'].get('augment_prob', 0.5),
        prefetch_factor=config['dataloader'].get('prefetch_factor', 2),
        persistent_workers=config['dataloader'].get('persistent_workers', False)
    )
    
    # Log dataset info to wandb
    if wandb_logger.is_enabled:
        wandb_logger.log_dataset_info(train_loader, val_loader, test_loader)

    # Create trainer
    trainer = Trainer(config, wandb_logger=wandb_logger)

    # Train model
    trainer.train(train_loader, val_loader)

    # Test on test set - using validate method for loss calculation
    print("\nEvaluating on test set...")
    test_loss, test_metrics, test_std_metrics = trainer.validate(test_loader, "Test")
    print(f"\nTest Set Results (from validate method):")
    print(f"  Test Loss:   {test_loss:.4f}")
    print(f"  Test F1:     {test_metrics['f1']:.4f} ± {test_std_metrics['f1']:.4f}")
    print(f"  Test MCC:    {test_metrics['mcc']:.4f} ± {test_std_metrics['mcc']:.4f}")
    print(f"  Test Exact:  {test_metrics['exact_match']:.4f} ± {test_std_metrics['exact_match']:.4f}")

    # Also run detailed evaluation with precision/recall and save per-sample results
    print("\nRunning detailed test evaluation...")
    detailed_metrics = evaluate_model(
        trainer.model,
        test_loader,
        trainer.device,
        output_dir=config['output']['save_dir']
    )

    print(f"\nDetailed Test Set Results:")
    print(f"  Test F1:         {detailed_metrics['test_f1']:.4f}")
    print(f"  Test MCC:        {detailed_metrics['test_mcc']:.4f}")
    print(f"  Test Precision:  {detailed_metrics['test_precision']:.4f}")
    print(f"  Test Recall:     {detailed_metrics['test_recall']:.4f}")
    print(f"  Test Exact:      {detailed_metrics['test_exact_match']:.4f}")
    print(f"  Total Samples:   {detailed_metrics['test_samples']}")

    # Log test metrics to wandb
    if wandb_logger.is_enabled:
        wandb_logger.log_test_metrics(test_loss, test_metrics, test_std_metrics)

        # Also log additional detailed metrics
        import wandb
        wandb.log({
            "test/precision": detailed_metrics['test_precision'],
            "test/recall": detailed_metrics['test_recall'],
            "test/detailed_f1": detailed_metrics['test_f1'],
            "test/detailed_mcc": detailed_metrics['test_mcc'],
            "test/detailed_exact_match": detailed_metrics['test_exact_match']
        })

        # Update wandb summary with detailed metrics
        wandb.run.summary["final_test_precision"] = detailed_metrics['test_precision']
        wandb.run.summary["final_test_recall"] = detailed_metrics['test_recall']
        wandb.run.summary["final_test_f1"] = detailed_metrics['test_f1']
        wandb.run.summary["final_test_mcc"] = detailed_metrics['test_mcc']

        wandb_logger.finish()


if __name__ == '__main__':
    main()
