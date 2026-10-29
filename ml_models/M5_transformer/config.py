"""Configuration for M5 Single-Task Temporal Transformer Benchmark.
Single source of truth for M5 Transformer architecture and training hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class M5TransformerConfig:
    model_name: str = "M5_transformer"
    description: str = "Single-Task Temporal Transformer Encoder (Head A Distress Only)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    # Architecture hyperparameters
    input_dim: int = 44          # 44 behavioral features per monthly token
    seq_length: int = 12         # 12 monthly observation sequence
    d_model: int = 64            # Latent token embedding dimension
    nhead: int = 4               # 4 Multi-head self-attention heads (d_k = 16)
    num_layers: int = 2          # 2 Pre-LN Transformer encoder layers
    dim_feedforward: int = 128   # FFN expansion dimension
    dropout: float = 0.1         # Attention & projection dropout
    activation: str = "gelu"     # Activation function
    pooling: str = "cls"         # [CLS] classification token pooling
    output_dim: int = 1          # Binary distress logit
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 15
    weight_decay: float = 1e-4
    param_grid_lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
