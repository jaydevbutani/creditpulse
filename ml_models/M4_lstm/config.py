"""Configuration for M4 LSTM Benchmark.
Single source of truth for M4 LSTM architecture and training hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class M4LSTMConfig:
    model_name: str = "M4_lstm"
    description: str = "2-Layer Long Short-Term Memory (LSTM) Sequential Network"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    # Architecture hyperparameters
    input_dim: int = 44       # 44 behavioral features per monthly token
    seq_length: int = 12      # 12 monthly observation sequence
    hidden_dim: int = 32      # Hidden units per recurrent layer
    num_layers: int = 2       # 2 stacked LSTM layers
    dropout: float = 0.1      # Inter-layer and head dropout
    output_dim: int = 1       # Scalar binary distress logit
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 15
    weight_decay: float = 1e-4
    param_grid_lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
