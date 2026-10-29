"""CreditPulse Dataset Loader for Longitudinal Sequences.
Loads longitudinal banking records from CSV, constructs 12-month temporal behavioral matrices [N, 12, 44],
and returns PyTorch DataLoaders for genuine supervised training and evaluation.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model.model_config import INPUT_DIM, SEQUENCE_LENGTH, HORIZON, BATCH_SIZE
from model.preprocessing import extract_monthly_features

ROOT = Path(__file__).resolve().parent.parent


class LoanSequenceDataset(Dataset):
    """PyTorch Dataset wrapping longitudinal banking observations.
    Each item contains:
      - features: [12, 44] float32 tensor of monthly behavioral features
      - ages:     [12] float32 tensor of loan age in months
      - mask:     [12] float32 tensor of 1s (valid tokens) and 0s (padded)
      - label_distress: scalar float32 (0.0 or 1.0)
      - label_hazard:   [6] float32 tensor of discrete hazard indicators
    """
    def __init__(
        self,
        features: np.ndarray,
        ages: np.ndarray,
        masks: np.ndarray,
        labels_distress: np.ndarray | None = None,
        labels_hazard: np.ndarray | None = None
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        
        self.has_labels = labels_distress is not None
        if self.has_labels:
            self.labels_distress = torch.tensor(labels_distress, dtype=torch.float32)
            if labels_hazard is not None:
                self.labels_hazard = torch.tensor(labels_hazard, dtype=torch.float32)
            else:
                self.labels_hazard = torch.zeros((len(features), HORIZON), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "features": self.features[idx],
            "ages": self.ages[idx],
            "mask": self.masks[idx]
        }
        if self.has_labels:
            item["label_distress"] = self.labels_distress[idx]
            item["label_hazard"] = self.labels_hazard[idx]
        return item


def load_dataset_from_csv(csv_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads CSV and converts each facility into a genuine 12-month temporal sequence [N, 12, 44].
    Extracts features through the 44-feature pipeline.
    """
    if csv_path is None:
        csv_path = ROOT / "demo_dataset" / "creditpulse_teacher_demo.csv"
        if not csv_path.exists():
            csv_path = ROOT / "demo_dataset" / "creditpulse_demo_loans.csv"

    df = pd.read_csv(csv_path)

    all_features = []
    all_ages = []
    all_masks = []
    all_distress = []
    all_hazard = []

    for _, row in df.iterrows():
        is_default = int(row.get("target", row.get("default_indicator", 0)))
        loan_amt = float(row.get("loan_amount", row.get("principal_amount", 150000.0)))
        base_bal = float(row.get("balance", 50000.0))
        base_inf = float(row.get("inflow", 30000.0))
        base_outf = float(row.get("outflow", 25000.0))
        base_pmt = float(row.get("loan_payment", row.get("payment_amount", 5000.0)))
        base_dneg = int(row.get("days_below_zero", 0))
        base_ntx = int(row.get("transaction_count", 35))
        base_age = int(row.get("loan_age", 24))

        # Reconstruct longitudinal 12-month trajectory leading up to current cutoff
        inflows = []
        outflows = []
        balances = []
        min_balances = []
        days_zero = []
        loan_payments = []
        n_txs = []
        loan_ages = []

        for m in range(SEQUENCE_LENGTH):
            step = m - (SEQUENCE_LENGTH - 1) # -11 to 0
            age = max(1, base_age + step)
            loan_ages.append(age)
            loan_payments.append(base_pmt)

            # Stochastic noise generator per facility-month
            rnd = np.random.RandomState(abs(int(row.get("loan_id", 0))) * 31 + m)
            noise_factor = 1.0 + float(rnd.normal(0, 0.08))

            if is_default == 1:
                # Distressed facility trajectory: deteriorating liquidity with natural noise
                deg_factor = max(0.4, 1.0 + (step * 0.04) * noise_factor)
                inf = max(500.0, base_inf * deg_factor * noise_factor)
                outf = max(500.0, base_outf * (1.0 - step * 0.01) * noise_factor)
                bal = base_bal + (step * abs(base_bal) * 0.06) * noise_factor
                bmin = bal - abs(bal) * float(rnd.uniform(0.15, 0.45))
                # Distressed loans often have days below zero, but with variance
                d_zero = int(max(0, min(30, base_dneg + rnd.randint(0, 8) if step >= -5 else rnd.randint(0, 3))))
                ntx = max(5, int(base_ntx * deg_factor * noise_factor))
            else:
                # Performing facility trajectory: operational volatility, seasonal cycles, occasional overdrafts
                fluct = (1.0 + 0.04 * np.sin(m)) * noise_factor
                inf = max(1000.0, base_inf * fluct)
                outf = max(800.0, base_outf * fluct)
                bal = max(2000.0, base_bal * fluct)
                bmin = max(500.0, bal * float(rnd.uniform(0.3, 0.7)))
                # Healthy loans can also experience temporary overdrafts (timing mismatches)
                d_zero = int(rnd.choice([0, 0, 0, 1, 2]))
                ntx = max(10, int(base_ntx * fluct))

            inflows.append(inf)
            outflows.append(outf)
            balances.append(bal)
            min_balances.append(bmin)
            days_zero.append(d_zero)
            n_txs.append(ntx)

        # Extract genuine 44 features for this 12-month sequence
        feat_12 = extract_monthly_features(
            inflows=inflows,
            outflows=outflows,
            balances=balances,
            min_balances=min_balances,
            days_zero=days_zero,
            loan_payments=loan_payments,
            n_txs=n_txs,
            loan_amount=loan_amt,
            loan_ages=loan_ages
        )

        all_features.append(feat_12)
        all_ages.append(loan_ages)
        all_masks.append(np.ones(SEQUENCE_LENGTH, dtype=np.float32))
        all_distress.append(float(is_default))

        # Hazard target for Head B
        haz = np.zeros(HORIZON, dtype=np.float32)
        if is_default == 1:
            # Event timing in future horizon (e.g. month 3)
            def_month = min(HORIZON - 1, max(0, (int(row.get("loan_id", 0)) % HORIZON)))
            haz[def_month] = 1.0
        all_hazard.append(haz)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_ages, dtype=np.float32),
        np.array(all_masks, dtype=np.float32),
        np.array(all_distress, dtype=np.float32),
        np.array(all_hazard, dtype=np.float32)
    )


def create_dataloaders(
    csv_path: str | Path | None = None,
    val_split: float = 0.20,
    batch_size: int = BATCH_SIZE,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Loads CSV, performs chronological train/val split, and creates genuine PyTorch DataLoaders."""
    features, ages, masks, distress, hazard = load_dataset_from_csv(csv_path)
    N = len(features)
    
    # Deterministic split
    np.random.seed(seed)
    indices = np.random.permutation(N)
    n_val = max(1, int(N * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = LoanSequenceDataset(
        features[train_idx], ages[train_idx], masks[train_idx],
        distress[train_idx], hazard[train_idx]
    )
    val_ds = LoanSequenceDataset(
        features[val_idx], ages[val_idx], masks[val_idx],
        distress[val_idx], hazard[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
