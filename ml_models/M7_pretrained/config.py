"""Configuration for M7 Pre-trained Sequence Benchmark.
Single source of truth for M7 architecture, SSL pre-training, and fine-tuning hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class M7PretrainedConfig:
    model_name: str = "M7_pretrained"
    description: str = "Self-Supervised Pre-trained Sequence Encoder Fine-Tuned for Distress"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    # Architecture hyperparameters
    input_dim: int = 44          # 44 behavioral features per monthly token
    seq_length: int = 12         # 12 monthly observation sequence
    hidden_dim: int = 48         # Hidden units in bidirectional GRU backbone (output = 96)
    dropout: float = 0.1         # Inter-layer dropout
    output_dim: int = 1          # Binary distress logit
    # Self-Supervised Learning (SSL) Pre-training
    pretrain_epochs: int = 10
    pretrain_lr: float = 1e-3
    mask_ratio: float = 0.15     # 15% random feature masking for denoising reconstruction
    # Downstream Supervised Fine-Tuning
    finetune_epochs: int = 15
    batch_size: int = 32
    weight_decay: float = 1e-4
    param_grid_lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
