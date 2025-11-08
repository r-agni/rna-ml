"""
Weights & Biases logging utilities for Lyra RNA contact prediction.

This module handles all wandb integration including initialization,
metric logging, and run management.
"""

import wandb
from typing import Dict, Any, Optional


class WandBLogger:
    """Manages Weights & Biases logging for training runs"""
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            config: Full configuration dictionary
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled
        self.config = config
        self.run = None
        
        if self.enabled:
            self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize wandb run with configuration"""
        wandb_config = self.config.get('wandb', {})
        
        self.run = wandb.init(
            entity=wandb_config.get('entity', None),
            project=wandb_config.get('project', 'lyra-rna-contact-prediction'),
            name=wandb_config.get('run_name', None),
            tags=wandb_config.get('tags', ['rna', 'contact-prediction']),
            notes=wandb_config.get('notes', 'RNA contact map prediction with Lyra model'),
            config={
                # Model config
                "model_dimension": self.config['model']['model_dimension'],
                "pgc_configs": self.config['model']['pgc_configs'],
                "num_s4": self.config['model']['num_s4'],
                "d_input": self.config['model']['d_input'],
                "dropout": self.config['model']['dropout'],
                "prenorm": self.config['model']['prenorm'],
                "final_dropout": self.config['model']['final_dropout'],
                
                # Training config
                "learning_rate": self.config['training']['learning_rate'],
                "weight_decay": self.config['training']['weight_decay'],
                "num_epochs": self.config['training']['num_epochs'],
                "batch_size": self.config['dataloader']['batch_size'],
                "pos_weight": self.config['training']['pos_weight'],
                "grad_clip": self.config['training']['grad_clip'],
                "early_stopping_patience": self.config['training']['early_stopping_patience'],
                
                # Data config
                "train_ratio": self.config['data']['train_ratio'],
                "val_ratio": self.config['data']['val_ratio'],
                "test_ratio": self.config['data']['test_ratio'],
                "random_seed": self.config['data']['random_seed'],
                
                # Scheduler config
                "scheduler_type": self.config['training']['scheduler']['type'],
                "scheduler_factor": self.config['training']['scheduler']['factor'],
                "scheduler_patience": self.config['training']['scheduler']['patience'],
                
                # Monitoring
                "monitor_metric": self.config['checkpoint']['monitor']
            }
        )
        
        print(f"\n✓ Weights & Biases initialized")
        print(f"  Project: {wandb.run.project}")
        print(f"  Run: {wandb.run.name}")
        print(f"  URL: {wandb.run.url}")
    
    def log_model_info(self, model, criterion, num_params: int, device: str):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            num_params: Total number of model parameters
            device: Device being used (cuda/cpu)
        """
        if not self.enabled:
            return
        
        wandb.config.update({
            "model_params": num_params,
            "device": str(device)
        })
        
        # Watch model gradients and parameters
        wandb.watch(model, criterion, log="all", log_freq=100)
    
    def log_dataset_info(self, train_loader, val_loader, test_loader):
        """
        Log dataset information.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        if not self.enabled:
            return
        
        wandb.config.update({
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "num_train_batches": len(train_loader),
            "num_val_batches": len(val_loader),
            "num_test_batches": len(test_loader),
            "total_batches_per_epoch": len(train_loader),
            "total_epochs_planned": self.config['training']['num_epochs']
        })
    
    def log_batch_metrics(self, epoch: int, batch_idx: int, total_batches: int,
                         loss: float, grad_norm: float, learning_rate: float):
        """
        Log metrics for a training batch.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            total_batches: Total number of batches per epoch
            loss: Batch loss value
            grad_norm: Gradient norm
            learning_rate: Current learning rate
        """
        if not self.enabled:
            return
        
        step = (epoch - 1) * total_batches + batch_idx
        wandb.log({
            "train/batch_loss": loss,
            "train/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "train/learning_rate": learning_rate,
            "train/step": step,
            "train/batch_idx": batch_idx,
            "train/current_epoch": epoch
        }, step=step)
    
    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float,
                         val_metrics: Dict[str, float], val_std_metrics: Dict[str, float],
                         learning_rate: float, epochs_without_improvement: int):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Average training loss
            val_loss: Average validation loss
            val_metrics: Dictionary of validation metrics (f1, mcc, exact_match)
            val_std_metrics: Dictionary of validation metric standard deviations
            learning_rate: Current learning rate
            epochs_without_improvement: Number of epochs without improvement
        """
        if not self.enabled:
            return
        
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
            "val/f1": val_metrics['f1'],
            "val/f1_std": val_std_metrics['f1'],
            "val/mcc": val_metrics['mcc'],
            "val/mcc_std": val_std_metrics['mcc'],
            "val/exact_match": val_metrics['exact_match'],
            "val/exact_match_std": val_std_metrics['exact_match'],
            "train/learning_rate_epoch": learning_rate,
            "train/epochs_without_improvement": epochs_without_improvement
        }, step=epoch)
    
    def log_best_metric(self, monitor_metric: str, value: float, epoch: int):
        """
        Log best metric achievement.
        
        Args:
            monitor_metric: Name of the monitored metric
            value: Best metric value
            epoch: Epoch where best metric was achieved
        """
        if not self.enabled:
            return
        
        wandb.run.summary[f"best_{monitor_metric}"] = value
        wandb.run.summary["best_epoch"] = epoch
    
    def log_lr_reduction(self, epoch: int, new_lr: float):
        """
        Log learning rate reduction.
        
        Args:
            epoch: Current epoch number
            new_lr: New learning rate value
        """
        if not self.enabled:
            return
        
        wandb.log({
            "train/lr_reduction": True,
            "train/lr_new": new_lr
        }, step=epoch)
    
    def log_early_stopping(self, epoch: int, total_batches: int):
        """
        Log early stopping information.
        
        Args:
            epoch: Epoch at which training stopped
            total_batches: Total number of batches trained
        """
        if not self.enabled:
            return
        
        wandb.run.summary["early_stopped"] = True
        wandb.run.summary["total_epochs"] = epoch
        wandb.run.summary["total_batches_trained"] = total_batches
    
    def log_training_completion(self, num_epochs: int, total_batches: int):
        """
        Log training completion statistics.
        
        Args:
            num_epochs: Total number of epochs completed
            total_batches: Total number of batches trained
        """
        if not self.enabled:
            return
        
        wandb.run.summary["total_epochs_completed"] = num_epochs
        wandb.run.summary["total_batches_trained"] = total_batches
    
    def log_test_metrics(self, test_loss: float, test_metrics: Dict[str, float],
                        test_std_metrics: Dict[str, float]):
        """
        Log test set evaluation metrics.
        
        Args:
            test_loss: Test set loss
            test_metrics: Dictionary of test metrics (f1, mcc, exact_match)
            test_std_metrics: Dictionary of test metric standard deviations
        """
        if not self.enabled:
            return
        
        wandb.log({
            "test/loss": test_loss,
            "test/f1": test_metrics['f1'],
            "test/f1_std": test_std_metrics['f1'],
            "test/mcc": test_metrics['mcc'],
            "test/mcc_std": test_std_metrics['mcc'],
            "test/exact_match": test_metrics['exact_match'],
            "test/exact_match_std": test_std_metrics['exact_match']
        })
        
        # Save final results to wandb summary
        wandb.run.summary["final_test_loss"] = test_loss
        wandb.run.summary["final_test_f1"] = test_metrics['f1']
        wandb.run.summary["final_test_mcc"] = test_metrics['mcc']
        wandb.run.summary["final_test_exact_match"] = test_metrics['exact_match']
    
    def finish(self):
        """Finish the wandb run and upload any remaining data"""
        if not self.enabled:
            return
        
        wandb.finish()
        print("\n✓ Weights & Biases run finished")
    
    @property
    def is_enabled(self) -> bool:
        """Check if wandb logging is enabled"""
        return self.enabled
