# Lyra - RNA Secondary Structure Prediction

Deep learning model for predicting RNA secondary structures using contact map prediction with the Lyra architecture (PGC + S4D layers).

## Overview

This project implements a neural network to predict RNA secondary structures by learning base pairing patterns. The model outputs a contact matrix where each position indicates the probability of two nucleotides forming a base pair.

### Architecture

- **Input**: One-hot encoded RNA sequences (A, C, G, U)
- **Model**:
  - Projected Gated Convolution (PGC) layers
  - Structured State Space (S4D) layers
  - Contact prediction head with symmetric output
- **Output**: Contact matrix (L x L) with base pair predictions

## Project Structure

```
Lyra/
├── dataset.csv                      # RNA dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── config/
│   └── config.yaml                  # Model and training configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── model.py                     # Lyra model implementation
│   ├── train.py                     # Training script
│   └── predict.py                   # Prediction/inference script
└── models/
    └── checkpoints/                 # Saved model checkpoints
        ├── best_model.pt            # Best model based on validation
        └── latest_model.pt          # Latest checkpoint
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify PyTorch CUDA installation (if using GPU):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Format

The CSV dataset should have the following columns:
- `id`: Unique identifier for each sequence
- `sequence`: RNA nucleotide sequence (A, C, G, U)
- `structure`: Secondary structure in dot-bracket notation
- `base_pairs`: JSON array of base pair coordinates [[i, j], ...]
- `len`: Sequence length

## Usage

### Training

Train the model on your dataset:

```bash
cd src
python train.py
```

This will:
- Load data from `dataset.csv`
- Split into train/val/test (80/10/10)
- Train the Lyra model with the configuration from `config/config.yaml`
- Save checkpoints to `models/checkpoints/`
- Display training progress and metrics (F1, MCC, Exact Match)

### Configuration

Edit `config/config.yaml` to adjust:
- Model architecture (dimensions, layers, dropout)
- Training hyperparameters (learning rate, epochs, batch size)
- Data paths and splits

Key parameters:
```yaml
model:
  model_dimension: 128        # Internal dimension
  num_s4: 4                   # Number of S4D layers
  pgc_configs: [[2.0, 2]]     # PGC expansion factor and layers

training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 16
  pos_weight: 10.0            # Weight for positive class (base pairs)
```

### Prediction

#### Single Sequence Prediction

```bash
cd src
python predict.py \
    --checkpoint ../models/checkpoints/best_model.pt \
    --sequence "ACGUACGUACGUACGU" \
    --threshold 0.5
```

#### Batch Prediction from CSV

```bash
cd src
python predict.py \
    --checkpoint ../models/checkpoints/best_model.pt \
    --csv ../test_sequences.csv \
    --output ../predictions.csv \
    --threshold 0.5
```

### Prediction API

You can also use the predictor in your Python code:

```python
from predict import RNAPredictor

# Load model
predictor = RNAPredictor('models/checkpoints/best_model.pt')

# Predict single sequence
result = predictor.predict_sequence("ACGUACGUACGU")

print(f"Base pairs: {result['base_pairs']}")
print(f"Contact map shape: {result['contact_map'].shape}")
print(f"Contact probabilities: {result['contact_probs']}")

# Predict batch
sequences = ["ACGUACGU", "GGCCAAUU", "AUGCAUGC"]
predictions = predictor.predict_batch(sequences)
```

## Evaluation Metrics

The model is evaluated using three metrics:

1. **F1 Score**: Harmonic mean of precision and recall for base pair prediction
2. **Matthews Correlation Coefficient (MCC)**: Balanced measure considering all confusion matrix categories
3. **Exact Match Accuracy**: Percentage of sequences with perfectly predicted structures

## Model Details

### PGC (Projected Gated Convolution)
- Applies depthwise convolution for local feature extraction
- Uses gating mechanism for selective information flow
- RMS normalization for stability

### S4D (Structured State Space)
- Efficient sequence modeling with state space layers
- Handles long-range dependencies
- Diagonal parameterization for computational efficiency

### Contact Prediction Head
- Transforms sequence embeddings to contact space
- Uses outer product for pairwise interactions
- Convolutional layers refine contact predictions
- Enforces symmetry in output matrix

## Training Tips

1. **GPU Memory**: If you run out of memory, reduce `batch_size` in config
2. **Convergence**: Monitor validation F1 score - training typically converges in 20-30 epochs
3. **Class Imbalance**: Adjust `pos_weight` if base pairs are very sparse in your data
4. **Learning Rate**: If training is unstable, reduce `learning_rate` to 0.0005 or lower

## Hardware Requirements

- **GPU**: Recommended (CUDA-capable with 8GB+ VRAM)
- **RAM**: 16GB+ for loading large datasets
- **Storage**: ~1GB for dataset + ~500MB for model checkpoints

## Results

After training, the model will output:
- Training and validation loss curves
- F1, MCC, and exact match scores
- Best model checkpoint based on validation F1
- Test set performance

Example output:
```
Epoch 25 Summary:
  Train Loss: 0.0523
  Val Loss:   0.0612
  Val F1:     0.8734
  Val MCC:    0.8456
  Val Exact:  0.6521
  New best val_f1: 0.8734
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `model_dimension`
- Use gradient checkpointing (requires code modification)

### Slow Training
- Increase `batch_size` (if memory allows)
- Set `num_workers > 0` in dataloader config (may not work well on Windows)
- Ensure data is on SSD, not HDD

### Poor Performance
- Check if sequences are properly encoded (A, C, G, U)
- Verify base pairs are in correct format
- Try increasing `model_dimension` or `num_s4`
- Adjust `pos_weight` for class balance

## Citation

If you use this code, please cite the original Lyra architecture and S4D papers.

## License

MIT License
