"""Configuration for M6 Multi-Task Temporal Transformer (Proposed Champion).
Single source of truth for M6 Multi-Task architecture, loss weighting, and training.
"""
from dataclasses import dataclass, field


@dataclass
class M6Config:
    model_name: str = "M6_multitask"
    description: str = "Multi-Task Pre-LN Temporal Transformer (Proposed Champion Architecture)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    # Architecture hyperparameters
    input_dim: int = 44          # 44 behavioral features per monthly token
    seq_length: int = 12         # 12 monthly observation sequence
    d_model: int = 32            # Latent token embedding dimension (d_model)
    nhead: int = 2               # 2 Multi-head self-attention heads (head_dim = 16)
    num_layers: int = 3          # 3 Pre-LN Transformer encoder blocks
    dim_feedforward: int = 64    # FFN expansion dimension (2 * d_model)
    dropout: float = 0.20        # Attention & residual dropout (20%)
    activation: str = "gelu"     # Non-linear activation
    pooling: str = "cls"         # [CLS] classification token pooling
    horizon: int = 6             # Head B multi-period forward discrete hazard horizon (months)
    head_a_output_dim: int = 1   # Head A scalar binary distress logit
    # Multi-task loss formulation: L_total = L_distress + loss_lambda * L_hazard
    loss_lambda: float = 0.30    # Multi-task loss weighting lambda
    pos_weight: float = 8.62     # Positive class weight addressing low default prevalence (~4.2%)
    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    param_grid_lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 2e-4, 1e-4])
