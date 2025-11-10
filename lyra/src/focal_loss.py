"""
Focal Loss implementation for handling extreme class imbalance in RNA contact prediction.

Focal Loss focuses training on hard examples and down-weights easy examples (like the massive
number of true negatives in sparse contact maps).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - gamma is the focusing parameter (gamma > 0 reduces loss for well-classified examples)
    - alpha balances positive/negative examples

    Args:
        alpha: Weighting factor for positive class (0-1). Use class frequency for balance.
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (B, L, L)
            targets: Ground truth binary labels (B, L, L)

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate p_t (probability of true class)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal loss formula
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Per-sample weighted BCE loss that adjusts for varying contact densities.

    This helps when some sequences have many contacts and others have few,
    preventing the model from just learning the average density.
    """

    def __init__(self, global_pos_weight=5.0):
        super().__init__()
        self.global_pos_weight = global_pos_weight

    def forward(self, inputs, targets, seq_lens):
        """
        Args:
            inputs: Logits from model (B, L, L)
            targets: Ground truth binary labels (B, L, L)
            seq_lens: Actual sequence lengths (B,)

        Returns:
            Weighted loss value
        """
        batch_size = inputs.shape[0]

        # Calculate per-sample contact density
        densities = []
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            num_contacts = targets[i, :seq_len, :seq_len].sum().item()
            total_positions = seq_len * seq_len
            density = num_contacts / total_positions if total_positions > 0 else 0
            densities.append(density)

        densities = torch.tensor(densities, device=inputs.device)

        # Adaptive pos_weight based on density
        # Lower density = higher weight for positives
        # Use inverse density, clamped to reasonable range
        adaptive_weights = torch.clamp(0.01 / (densities + 1e-6), 1.0, 50.0)

        # Calculate loss per sample
        total_loss = 0
        for i in range(batch_size):
            seq_len = seq_lens[i].item()

            # Extract valid region
            pred = inputs[i, :seq_len, :seq_len]
            target = targets[i, :seq_len, :seq_len]

            # Calculate weighted BCE
            pos_weight = torch.tensor([adaptive_weights[i].item()], device=inputs.device)
            loss = F.binary_cross_entropy_with_logits(
                pred, target,
                pos_weight=pos_weight,
                reduction='mean'
            )
            total_loss += loss

        return total_loss / batch_size


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-like tasks.

    Dice coefficient measures overlap between prediction and ground truth.
    Works well for imbalanced data without needing pos_weight tuning.

    DiceLoss = 1 - DiceCoefficient
    DiceCoefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (B, L, L)
            targets: Ground truth binary labels (B, L, L)

        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - dice coefficient
        return 1.0 - dice_coef


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss.

    This combines the benefits of both:
    - Focal Loss handles hard examples and class imbalance
    - Dice Loss optimizes for overlap/F1-like metric
    """

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_weight=0.3):
        super().__init__()
        # Use reduction='none' to match other loss functions for consistent masking
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        self.dice_loss = DiceLoss(smooth=1.0)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (B, L, L)
            targets: Ground truth binary labels (B, L, L)

        Returns:
            Combined loss value (element-wise)
        """
        # Focal loss returns element-wise loss (B, L, L) with reduction='none'
        focal = self.focal_loss(inputs, targets)
        # Dice loss returns scalar
        dice = self.dice_loss(inputs, targets)

        # Combine losses: broadcast dice loss to match focal loss shape
        # This ensures dice loss contributes meaningfully regardless of tensor size
        dice_broadcast = (self.dice_weight * dice) * torch.ones_like(focal)

        # Return element-wise loss
        return focal + dice_broadcast
