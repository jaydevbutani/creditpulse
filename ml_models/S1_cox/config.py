"""Configuration for S1 Cox Proportional Hazards Survival Model.
Single source of truth for S1 Cox parameters.
"""
from dataclasses import dataclass, field


@dataclass
class S1CoxConfig:
    model_name: str = "S1_cox"
    description: str = "Cox Proportional Hazards Survival Model (Continuous Time-to-Event)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    n_features_selected: int = 15
    time_horizon: float = 12.0
    l1_ratio: float = 0.0
    # Hyperparameter sweep for L2 penalizer in Cox regression to prevent collinear singularity
    penalizer_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 5.0])
