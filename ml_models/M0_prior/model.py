"""M0 Prior Model: Constant base-rate predictor without feature conditioning.
P(Y = 1 | X) = P(Y = 1) = y_bar.
Architecture is visibly driven by M0Config.
"""
from __future__ import annotations
import numpy as np
from ml_models.M0_prior.config import M0Config


class EmpiricalPriorBaseline:
    """Predicts the constant empirical distress probability from training labels."""
    def __init__(self, cfg: M0Config | None = None):
        self.cfg = cfg or M0Config()
        self.prior_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.full(shape=(len(X),), fill_value=self.prior_, dtype=np.float32)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
