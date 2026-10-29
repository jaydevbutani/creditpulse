"""Stratified and Chronological Data Splitting for Benchmark Models.
Uses deterministic stratified partitioning to preserve default prevalence across splits:
  - Training Period (65%): Historical cohort
  - Validation Period (15%): Held-out parameter sweep split
  - Test Period (20%): Out-of-time evaluation split
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class TabularSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    durations_train: np.ndarray | None = None
    events_train: np.ndarray | None = None
    durations_test: np.ndarray | None = None
    events_test: np.ndarray | None = None


@dataclass
class SequenceSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def split_tabular_data(
    X: np.ndarray,
    y: np.ndarray,
    durations: np.ndarray | None = None,
    events: np.ndarray | None = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = 42
) -> TabularSplit:
    """Performs deterministic stratified partition into train, validation, and test subsets."""
    n = len(X)
    indices = np.arange(n)

    # First split off test set (stratified by default label y)
    stratify_y = y if len(np.unique(y)) > 1 else None
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_y
    )

    # Then split train and validation
    val_rel_size = val_size / (1.0 - test_size)
    stratify_train_val = y[train_val_idx] if len(np.unique(y[train_val_idx])) > 1 else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_rel_size, random_state=seed, stratify=stratify_train_val
    )

    return TabularSplit(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        durations_train=durations[train_idx] if durations is not None else None,
        events_train=events[train_idx] if events is not None else None,
        durations_test=durations[test_idx] if durations is not None else None,
        events_test=events[test_idx] if events is not None else None,
    )


def split_sequence_data(
    labels: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = 42
) -> SequenceSplit:
    """Returns deterministic index partitions for sequential datasets with stratification."""
    n_samples = len(labels)
    indices = np.arange(n_samples)
    stratify_y = labels if len(np.unique(labels)) > 1 else None

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_y
    )

    val_rel_size = val_size / (1.0 - test_size)
    stratify_train_val = labels[train_val_idx] if len(np.unique(labels[train_val_idx])) > 1 else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_rel_size, random_state=seed, stratify=stratify_train_val
    )

    return SequenceSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )
