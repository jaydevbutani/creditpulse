"""Configuration for M1 Logistic Regression Scorecard Benchmark.
Single source of truth for M1 hyperparameters.
"""
from dataclasses import dataclass, field


@dataclass
class M1Config:
    model_name: str = "M1_logreg"
    description: str = "L2-Regularized Logistic Regression (Basel IRB Regulatory Scorecard)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str = "balanced"
    # Hyperparameter search grid for C (inverse regularization strength)
    param_grid_C: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
