"""M1 Logistic Regression Model.
Standard banking scorecard model: logit(p) = w^T x + b.
Architecture is visibly driven by M1Config.
"""
from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ml_models.M1_logreg.config import M1Config


def build_logistic_regression(cfg: M1Config, C: float | None = None) -> Pipeline:
    """Builds a scikit-learn Pipeline with feature standardization and L2 logistic regression."""
    chosen_c = C if C is not None else cfg.param_grid_C[0]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=chosen_c,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            random_state=cfg.seed
        ))
    ])
