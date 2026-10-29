"""Dataset bundle shared by every model so splits, features and labels are identical (report §8.1)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditpulse.features.panel import FEATURE_GROUPS, FEATURES
from creditpulse.utils import p


@dataclass
class Bundle:
    snap: pd.DataFrame          # all at-risk snapshots (incl. censored)
    Z: np.ndarray               # tokens [N, W, F]
    pad: np.ndarray             # [N, W] 1 = padded
    A: np.ndarray               # aggregated features [N, D]
    acols: list
    age: np.ndarray             # raw loan age per token position [N, W]
    y: np.ndarray               # Head A label (NaN = censored)
    event_k: np.ndarray
    follow_k: np.ndarray

    def idx(self, split: str, labelled: bool = True) -> np.ndarray:
        m = (self.snap.split == split).to_numpy().copy()
        if labelled:
            m &= ~np.isnan(self.y)
        return np.flatnonzero(m)

    def tuning_folds(self, inner_start: str):
        """Pooled tuning set (deviation from val-only tuning, see model card): fold 1 fits on train loans
        originated before `inner_start` and scores the train loans originated after it; fold 2 fits on
        all of train and scores val. Both folds are time-ordered and never touch test."""
        lab = ~np.isnan(self.y)
        orig = self.snap["orig_date"].values
        tr = (self.snap.split == "train").to_numpy() & lab
        early = tr & (orig < np.datetime64(inner_start))
        late = tr & (orig >= np.datetime64(inner_start))
        return [(np.flatnonzero(early), np.flatnonzero(late)), (np.flatnonzero(tr), self.idx("val"))]

    def feature_mask(self, drop_groups=()) -> np.ndarray:
        drop = {f for g in drop_groups for f in FEATURE_GROUPS[g]}
        return np.array([f not in drop for f in FEATURES])


def token_ages(snap: pd.DataFrame, W: int) -> np.ndarray:
    return snap.loan_age.values[:, None] - (W - 1 - np.arange(W))[None, :]


def load_bundle(cfg: dict, tokens_file="tokens_p1.npz", snap_file="snapshots.parquet") -> Bundle:
    proc = p(cfg, "processed")
    snap = pd.read_parquet(proc / snap_file)
    lo = pd.read_parquet(proc / "loans_events.parquet").set_index("loan_id")
    snap["event_mi"] = snap.loan_id.map(lo["event_mi"])
    snap["orig_date"] = snap.loan_id.map(lo["date"])
    d = np.load(proc / tokens_file)
    acols = json.loads((proc / "agg_columns.json").read_text())
    W = d["Z"].shape[1]
    return Bundle(snap=snap, Z=d["Z"], pad=d["pad"], A=d["A"], acols=acols, age=token_ages(snap, W),
                  y=snap.label.values.astype(float), event_k=snap.event_k.values, follow_k=snap.follow_k.values)


def agg_column_mask(acols: list, drop_groups=()) -> np.ndarray:
    drop = {f for g in drop_groups for f in FEATURE_GROUPS[g]}
    return np.array([c.split("__")[0] not in drop for c in acols])


def save_preds(cfg: dict, name: str, snap: pd.DataFrame, score: np.ndarray, cal: np.ndarray,
               hazard: np.ndarray | None = None, extra: dict | None = None):
    out = snap[["loan_id", "account_id", "cutoff_mi", "split", "label", "event_k", "follow_k", "event_mi"]].copy()
    out["score_raw"] = score
    out["score"] = cal
    if hazard is not None:
        for k in range(hazard.shape[1]):
            out[f"h{k + 1}"] = hazard[:, k]
    for k, v in (extra or {}).items():
        out[k] = v
    d = p(cfg, "artifacts") / "preds"
    d.mkdir(exist_ok=True)
    out.to_parquet(d / f"{name}.parquet", index=False)
    return out
