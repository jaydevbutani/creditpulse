"""Configuration for M2 LightGBM Benchmark.
Single source of truth for M2 LightGBM hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class M2LGBMConfig:
    model_name: str = "M2_lgbm"
    description: str = "LightGBM Gradient Boosted Decision Trees (Tabular Benchmark)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    class_weight: str = "balanced"
    # Hyperparameter grid for genuine validation sweep
    param_grid: list[dict] = field(default_factory=lambda: [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 5, "learning_rate": 0.10, "n_estimators": 100},
    ])
