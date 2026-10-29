"""Data utilities for the benchmark Model Zoo.
Provides standard datasets in sequential [N, 12, 44] and tabular flattened formats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from model.dataset import load_dataset_from_csv
from model.preprocessing import FEATURE_REGISTRY

ROOT = Path(__file__).resolve().parent.parent.parent


class SequentialBenchmarkDataset(Dataset):
    """PyTorch Dataset for sequence benchmark models (GRU, LSTM, Transformer, M6)."""
    def __init__(self, features: np.ndarray, ages: np.ndarray, masks: np.ndarray,
                 labels_distress: np.ndarray, labels_hazard: np.ndarray | None = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.labels_distress = torch.tensor(labels_distress, dtype=torch.float32)
        if labels_hazard is not None:
            self.labels_hazard = torch.tensor(labels_hazard, dtype=torch.float32)
        else:
            self.labels_hazard = torch.zeros((len(features), 6), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "ages": self.ages[idx],
            "mask": self.masks[idx],
            "label_distress": self.labels_distress[idx],
            "label_hazard": self.labels_hazard[idx]
        }


def load_raw_dataset(csv_path: str | Path | None = None):
    """Loads raw [N, 12, 44] sequences and labels from the canonical dataset."""
    return load_dataset_from_csv(csv_path)


def get_tabular_dataset(csv_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Converts 12-month panel data [N, 12, 44] into standard aggregated tabular features for
    baseline models (M0, M1, M2-LGBM, M2-XGB, S1-Cox).

    Extracts:
      - mean across 12 months (44 features)
      - latest month T=12 (44 features)
      - delta: (month 12 - month 1) (44 features)
      - min across 12 months (44 features)
    Total = 176 interpretable cross-sectional banking indicators.

    Returns:
      (X_tab, y_distress, durations, events, feature_names)
    """
    features, ages, masks, distress, hazard = load_raw_dataset(csv_path)
    N, T, F = features.shape

    mean_feat = np.mean(features, axis=1)
    last_feat = features[:, -1, :]
    delta_feat = features[:, -1, :] - features[:, 0, :]
    min_feat = np.min(features, axis=1)

    X_tab = np.hstack([mean_feat, last_feat, delta_feat, min_feat]).astype(np.float32)

    # Clean any NaNs or infinities
    X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=1.0, neginf=-1.0)

    feature_names = []
    for prefix in ["mean_", "latest_", "delta_", "min_"]:
        for name in FEATURE_REGISTRY:
            feature_names.append(f"{prefix}{name}")

    # For survival models (duration until distress or observation window = 12)
    # If distressed, simulate time-to-event based on hazard or default month
    durations = np.zeros(N, dtype=np.float32)
    events = distress.astype(np.float32)
    for i in range(N):
        if events[i] == 1.0:
            haz = hazard[i]
            event_idx = np.where(haz > 0.5)[0]
            if len(event_idx) > 0:
                durations[i] = float(event_idx[0] + 1)
            else:
                durations[i] = float(np.random.RandomState(i).randint(3, 10))
        else:
            durations[i] = 12.0  # right-censored at 12 months

    return X_tab, distress, durations, events, feature_names
