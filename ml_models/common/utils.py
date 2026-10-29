"""Common utility functions for seed reproducibility, parameter counting, and metric serialization.
"""
from __future__ import annotations

import os
import random
import json
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Calculates total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_metrics(filepath: str | Path, metrics: dict) -> None:
    """Persists evaluation metrics dictionary to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_model_summary(model_name: str, metrics: dict, params: int | str = "N/A") -> None:
    """Prints a clean viva-ready summary of model evaluation."""
    print("=" * 60)
    print(f"  MODEL: {model_name}")
    print(f"  Trainable Parameters: {params}")
    print("-" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<20}: {v:.4f}")
        else:
            print(f"    {k:<20}: {v}")
    print("=" * 60)
