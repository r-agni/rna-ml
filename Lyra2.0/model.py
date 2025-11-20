import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    """Dropout for N-dimensional tensors with optional tied mask across spatial dims."""

    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
        return X


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class PGC(nn.Module):
    """Projected Gated Convolution for local pattern extraction and second-order interactions."""

    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        expanded_dim = int(d_model * expansion_factor)

        # Depthwise convolution for local patterns
        self.conv = nn.Conv1d(
            expanded_dim, expanded_dim,
            kernel_size=3, padding=1,
            groups=expanded_dim
        )

        # Input projection splits into two branches
        self.in_proj = nn.Linear(d_model, expanded_dim * 2)
        self.in_norm = nn.RMSNorm(expanded_dim * 2)

        # Output projection
        self.norm = nn.RMSNorm(expanded_dim)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # u: (B, L, d_model)

        # Project and normalize
        xv = self.in_norm(self.in_proj(u))
        x, v = xv.chunk(2, dim=-1)

        # Apply depthwise convolution to one branch
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

        # Gating: element-wise multiplication creates second-order interactions
        gate = v * x_conv

        # Project back to model dimension
        x = self.out_proj(self.norm(gate))
        x = self.dropout(x)

        return x


class S4DKernel(nn.Module):
    """S4D Kernel: Diagonal State Space Model kernel computation."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model

        # Initialize discretization timestep
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # Initialize C matrix (complex)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        # Register parameters with optional custom learning rate
        self.register("log_dt", log_dt, lr)

        # Initialize A matrix (diagonal, complex)
        # Real part controls decay, imaginary part controls oscillation frequency
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)

        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """Compute the S4D convolution kernel of length L."""
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        # Discretize
        dtA = A * dt.unsqueeze(-1)

        # Compute kernel
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor as parameter or buffer with optional custom LR."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    """S4D Layer: Diagonal State Space Model for sequence modeling."""

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        # Skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # S4D kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Output projection with GLU
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """
        Args:
            u: (B, d_model, L) if transposed else (B, L, d_model)
        Returns:
            y: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)

        L = u.size(-1)

        # Compute S4D kernel
        k = self.kernel(L=L)

        # FFT convolution
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]

        # Skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Activation and dropout
        y = self.dropout(self.activation(y))

        # Output projection
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)

        return y


class LyraContact(nn.Module):
    """
    Lyra model modified for RNA secondary structure contact prediction.

    Takes RNA sequences and predicts L×L contact matrices indicating
    which nucleotide positions pair with each other.

    Parameters:
        model_dimension: Internal dimension of the model
        pgc_configs: List of tuples (expansion_factor, num_layers) for PGC blocks
        num_s4: Number of S4D layers
        d_input: Input feature dimension (4 for RNA: A, U, G, C)
        d_state: State dimension for S4D
        dropout: Dropout rate
        prenorm: Whether to use pre-normalization
    """

    def __init__(
        self,
        model_dimension,
        pgc_configs,
        num_s4,
        d_input,
        d_state=64,
        dropout=0.2,
        prenorm=True,
    ):
        super().__init__()

        # Validate parameters
        if model_dimension <= 0:
            raise ValueError(f"model_dimension must be positive, got {model_dimension}")
        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if num_s4 < 0:
            raise ValueError(f"num_s4 must be non-negative, got {num_s4}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not isinstance(pgc_configs, list):
            raise ValueError(f"pgc_configs must be a list, got {type(pgc_configs)}")

        # Input encoder
        self.encoder = nn.Linear(d_input, model_dimension)

        # PGC layers for local pattern extraction
        self.pgc_layers = nn.ModuleList()
        for expansion_factor, num_layers in pgc_configs:
            for _ in range(num_layers):
                self.pgc_layers.append(
                    PGC(model_dimension, expansion_factor, dropout)
                )

        self.prenorm = prenorm

        # S4D layers for long-range dependencies
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(model_dimension, d_state=d_state, dropout=dropout, transposed=True)
            )
            self.norms.append(RMSNorm(model_dimension))
            self.dropouts.append(nn.Dropout(dropout))

        # Store model dimension for forward pass
        self.model_dimension = model_dimension

        # Multi-task prediction heads
        # Head 1: Dot-bracket prediction (primary, 3 classes: '.', '(', ')')
        self.dotbracket_head = nn.Sequential(
            nn.Linear(model_dimension, model_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension // 2, 3)
        )

        # Head 2: Binary pairing prediction (auxiliary, 2 classes: unpaired/paired)
        self.binary_head = nn.Sequential(
            nn.Linear(model_dimension, model_dimension // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension // 4, 2)
        )

        # Head 3: Pairing partner prediction (auxiliary, uses attention)
        # For each position, we'll compute attention scores over all positions
        # to predict which position it pairs with (or unpaired)
        self.query_proj = nn.Linear(model_dimension, model_dimension)
        self.key_proj = nn.Linear(model_dimension, model_dimension)

        # Bias for "unpaired" class in pairing head
        self.unpaired_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_embeddings=False):
        """
        Multi-task forward pass with three prediction heads.

        Args:
            x: (B, L, d_input) - Input sequences (one-hot encoded)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary containing:
                'dotbracket': (B, L, 3) - Dot-bracket classification logits
                'binary': (B, L, 2) - Binary paired/unpaired logits
                'pairing': (B, L, L+1) - Pairing partner logits
                'embeddings': (B, L, d_model) - Optional embeddings
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (B, L, d_input), got shape {x.shape}")

        B, L, d_in = x.shape

        if L == 0:
            raise ValueError("Sequence length L cannot be 0")

        if B == 0:
            raise ValueError("Batch size B cannot be 0")

        # Check for NaN/Inf in input
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input contains Inf values")

        # Encode input
        x = self.encoder(x)  # (B, L, d_model)

        # Check for NaN/Inf after encoding
        if torch.isnan(x).any():
            raise RuntimeError("NaN detected after encoder. Check model initialization.")
        if torch.isinf(x).any():
            raise RuntimeError("Inf detected after encoder. Check model initialization.")

        # Apply PGC layers
        for pgc_layer in self.pgc_layers:
            x = pgc_layer(x)

        # Transpose for S4D (expects channel-first)
        x = x.transpose(-1, -2)  # (B, d_model, L)

        # Apply S4D layers with residual connections
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        # Transpose back
        x = x.transpose(-1, -2)  # (B, L, d_model)

        embeddings = x

        # Multi-task prediction heads

        # Head 1: Dot-bracket prediction (PRIMARY - highest weight in loss)
        dotbracket_logits = self.dotbracket_head(x)  # (B, L, 3)

        # Head 2: Binary pairing prediction (AUXILIARY - helps learn paired regions)
        binary_logits = self.binary_head(x)  # (B, L, 2)

        # Head 3: Pairing partner prediction (AUXILIARY - for exact pairing recovery)
        # Compute per-position pairing scores using scaled dot-product attention
        queries = self.query_proj(x)  # (B, L, d_model)
        keys = self.key_proj(x)       # (B, L, d_model)

        # Compute attention scores: (B, L, L)
        scale = self.model_dimension ** 0.5
        pair_scores = torch.matmul(queries, keys.transpose(-1, -2)) / scale  # (B, L, L)

        # Add unpaired class: (B, L, L+1)
        unpaired_scores = self.unpaired_bias.expand(B, L, 1)  # (B, L, 1)
        pairing_logits = torch.cat([pair_scores, unpaired_scores], dim=-1)  # (B, L, L+1)

        # Validate output shapes
        if dotbracket_logits.shape != (B, L, 3):
            raise RuntimeError(f"Invalid dotbracket logits shape: {dotbracket_logits.shape}, expected ({B}, {L}, 3)")
        if binary_logits.shape != (B, L, 2):
            raise RuntimeError(f"Invalid binary logits shape: {binary_logits.shape}, expected ({B}, {L}, 2)")
        if pairing_logits.shape != (B, L, L+1):
            raise RuntimeError(f"Invalid pairing logits shape: {pairing_logits.shape}, expected ({B}, {L}, {L+1})")

        # Check for NaN/Inf in outputs
        if torch.isnan(dotbracket_logits).any():
            raise RuntimeError("NaN detected in dotbracket logits")
        if torch.isinf(dotbracket_logits).any():
            raise RuntimeError("Inf detected in dotbracket logits")
        if torch.isnan(binary_logits).any():
            raise RuntimeError("NaN detected in binary logits")
        if torch.isinf(binary_logits).any():
            raise RuntimeError("Inf detected in binary logits")
        if torch.isnan(pairing_logits).any():
            raise RuntimeError("NaN detected in pairing logits")
        if torch.isinf(pairing_logits).any():
            raise RuntimeError("Inf detected in pairing logits")

        outputs = {
            'dotbracket': dotbracket_logits,
            'binary': binary_logits,
            'pairing': pairing_logits,
        }

        if return_embeddings:
            outputs['embeddings'] = embeddings

        return outputs


def get_optimizer_groups(model, lr=1e-3, weight_decay=0.01):
    """
    Get parameter groups with special handling for S4D parameters.

    S4D kernel parameters (log_dt, log_A_real, A_imag) often need
    different learning rates and no weight decay.
    """
    # Separate S4D kernel parameters from others
    s4d_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter has custom optimization settings
        if hasattr(param, '_optim'):
            s4d_params.append((name, param))
        else:
            other_params.append((name, param))

    # Build parameter groups
    param_groups = []

    # Regular parameters
    if other_params:
        param_groups.append({
            'params': [p for _, p in other_params],
            'lr': lr,
            'weight_decay': weight_decay,
        })

    # S4D parameters (group by learning rate)
    s4d_by_lr = {}
    for name, param in s4d_params:
        optim = getattr(param, '_optim', {})
        param_lr = optim.get('lr', lr)
        param_wd = optim.get('weight_decay', 0.0)

        key = (param_lr, param_wd)
        if key not in s4d_by_lr:
            s4d_by_lr[key] = []
        s4d_by_lr[key].append(param)

    for (param_lr, param_wd), params in s4d_by_lr.items():
        param_groups.append({
            'params': params,
            'lr': param_lr,
            'weight_decay': param_wd,
        })

    return param_groups
