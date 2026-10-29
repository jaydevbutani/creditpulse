"""Configuration for S2 Discrete-Time Hazard Survival Model.
Single source of truth for S2 Neural Hazard hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class S2HazardConfig:
    model_name: str = "S2_hazard"
    description: str = "Discrete-Time Neural Hazard Survival Model (Multi-Period Hazards)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    input_dim: int = 44
    hidden_dim: int = 32
    horizon: int = 6
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 15
    weight_decay: float = 1e-4
    param_grid_lr: list[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
