"""Configuration for M0 Empirical Prior Baseline.
Single source of truth for M0 parameters.
"""
from dataclasses import dataclass


@dataclass
class M0Config:
    model_name: str = "M0_prior"
    description: str = "Empirical Base Rate Prior (No Features Baseline)"
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
