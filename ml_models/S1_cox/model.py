"""S1 Cox Proportional Hazards Model Architecture.
h(t | x) = h_0(t) * exp(beta^T x)
Architecture and covariates are visibly driven by S1CoxConfig.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from ml_models.S1_cox.config import S1CoxConfig


class CoxSurvivalBaseline:
    """Wrapper around lifelines.CoxPHFitter strictly configured via S1CoxConfig."""
    def __init__(self, cfg: S1CoxConfig, penalizer: float | None = None):
        self.cfg = cfg
        self.penalizer = penalizer if penalizer is not None else cfg.penalizer_grid[0]
        self.cph = CoxPHFitter(penalizer=self.penalizer, l1_ratio=cfg.l1_ratio)
        self.selected_indices: np.ndarray = np.array([])
        self.feature_cols: list[str] = []

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray, feature_names: list[str] | None = None):
        # Select top non-zero variance features defined by cfg.n_features_selected
        variances = np.var(X, axis=0)
        k = min(self.cfg.n_features_selected, X.shape[1])
        top_idx = np.argsort(variances)[-k:]
        self.selected_indices = top_idx
        X_sub = X[:, top_idx]

        cols = [f"F_{i}" for i in range(X_sub.shape[1])]
        self.feature_cols = cols

        df = pd.DataFrame(X_sub, columns=cols)
        df["duration"] = np.maximum(durations, 0.1)
        df["event"] = events

        self.cph.fit(df, duration_col="duration", event_col="event")
        return self

    def predict_distress_probability(self, X: np.ndarray, time_horizon: float | None = None) -> np.ndarray:
        """Predicts cumulative failure probability 1 - S(t=horizon | x)."""
        h = time_horizon if time_horizon is not None else self.cfg.time_horizon
        X_sub = X[:, self.selected_indices]
        df = pd.DataFrame(X_sub, columns=self.feature_cols)
        surv = self.cph.predict_survival_function(df, times=[h])
        probs = (1.0 - surv.iloc[0].values).astype(np.float32)
        return np.clip(probs, 0.0, 1.0)
