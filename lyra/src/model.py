import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ========== Utility Layers ==========

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), but got {}".format(p))
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
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


def dropout_fn(dropout):
    return nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()


# ========== PGC: Projected Gated Convolution ==========

class PGC(nn.Module):
    """Projected Gated Convolution layer"""

    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        d_inner = int(d_model * expansion_factor)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.norm = RMSNorm(d_inner)
        self.in_norm = RMSNorm(d_inner * 2)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, u):
        xv = self.in_norm(self.in_proj(u))
        x, v = xv.chunk(2, dim=-1)
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        gate = v * x_conv
        gate = self.norm(gate)
        x = self.out_proj(gate)
        return x


# ========== S4D: Structured State Space ==========

class S4DKernel(nn.Module):
    """S4D Kernel computation"""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn,hnl->hl', C, torch.exp(K)).real
        return K

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    """S4D Layer"""

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


# ========== Lyra Model for Contact Map Prediction ==========

class LyraRNAContactPredictor(nn.Module):
    """
    Lyra model adapted for RNA secondary structure prediction via contact map.

    Args:
        model_dimension: Internal dimension of S4D layers
        pgc_configs: List of tuples (expansion_factor, num_pgc_layers)
        num_s4: Number of S4D layers
        d_input: Input dimension (4 for one-hot RNA)
        dropout: Dropout rate
        prenorm: Whether to use pre-normalization
        final_dropout: Dropout before contact prediction head
    """

    def __init__(
        self,
        model_dimension,
        pgc_configs,
        num_s4,
        d_input=4,
        dropout=0.2,
        prenorm=True,
        final_dropout=0.2
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, model_dimension)
        self.pgc_layers = nn.ModuleList()

        for config in pgc_configs:
            expansion_factor, num_layers = config
            for _ in range(num_layers):
                self.pgc_layers.append(PGC(model_dimension, expansion_factor, dropout))

        self.prenorm = prenorm

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(model_dimension, dropout=dropout, transposed=True, lr=min(0.001, 0.002))
            )
            self.norms.append(RMSNorm(model_dimension))
            self.dropouts.append(dropout_fn(dropout))

        # Contact map prediction head
        self.contact_head = nn.Sequential(
            nn.Dropout(final_dropout),
            nn.Linear(model_dimension, model_dimension // 2),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(model_dimension // 2, model_dimension // 4),
            nn.ReLU()
        )

        # Final contact prediction: pairwise features followed by conv
        # Input will be 2 * (model_dimension // 4) = model_dimension // 2 channels
        in_channels = model_dimension // 2
        self.contact_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: Input sequences (B, L, d_input)
            seq_lens: Actual sequence lengths (B,) for masking

        Returns:
            contact_map: Predicted contact matrix (B, L, L)
        """
        batch_size, seq_len, _ = x.shape

        # Encode input
        x = self.encoder(x)  # (B, L, d_model)

        # Apply PGC layers
        for pgc_layer in self.pgc_layers:
            x = pgc_layer(x)

        # Transpose for S4D layers
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

        x = x.transpose(-1, -2)  # (B, L, d_model)

        # Contact prediction
        x = self.contact_head(x)  # (B, L, d_model//4)

        # Create contact map via pairwise feature combination
        # Reshape to (B, L, 1, d) and (B, 1, L, d) for broadcasting
        x1 = x.unsqueeze(2)  # (B, L, 1, d)
        x2 = x.unsqueeze(1)  # (B, 1, L, d)

        # Concatenate pairwise features instead of dot product
        # This allows the model to learn arbitrary pairwise interactions
        contact_features = torch.cat([
            x1.expand(-1, -1, seq_len, -1),  # (B, L, L, d)
            x2.expand(-1, seq_len, -1, -1)   # (B, L, L, d)
        ], dim=-1)  # (B, L, L, 2*d)

        # Permute to channel-first format for conv layers
        contact_features = contact_features.permute(0, 3, 1, 2)  # (B, 2*d, L, L)

        # Apply convolutional layers
        contact_map = self.contact_predictor(contact_features)  # (B, 1, L, L)
        contact_map = contact_map.squeeze(1)  # (B, L, L)

        # Make symmetric
        contact_map = (contact_map + contact_map.transpose(-1, -2)) / 2

        # Apply mask if sequence lengths provided
        if seq_lens is not None:
            mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            for i, length in enumerate(seq_lens):
                mask[i, :length, :length] = 1.0
            contact_map = contact_map * mask

        return contact_map


def create_lyra_model(config):
    """
    Create Lyra model from configuration dictionary.

    Args:
        config: Dictionary with model parameters

    Returns:
        LyraRNAContactPredictor model
    """
    model = LyraRNAContactPredictor(
        model_dimension=config.get('model_dimension', 128),
        pgc_configs=config.get('pgc_configs', [(2.0, 2)]),
        num_s4=config.get('num_s4', 4),
        d_input=config.get('d_input', 4),
        dropout=config.get('dropout', 0.2),
        prenorm=config.get('prenorm', True),
        final_dropout=config.get('final_dropout', 0.2)
    )
    return model
