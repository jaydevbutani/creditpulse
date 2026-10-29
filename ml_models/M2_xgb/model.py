"""M2 XGBoost Model Architecture.
Gradient boosted decision tree ensemble with depth-wise tree growth.
Architecture is visibly driven by M2XGBConfig.
"""
from __future__ import annotations
from xgboost import XGBClassifier
from ml_models.M2_xgb.config import M2XGBConfig


def build_xgb_classifier(
    cfg: M2XGBConfig,
    params: dict | None = None
) -> XGBClassifier:
    """Builds an XGBoost Classifier strictly driven by M2XGBConfig."""
    p = params or cfg.param_grid[0]
    return XGBClassifier(
        max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.05),
        n_estimators=p.get("n_estimators", 50),
        random_state=cfg.seed,
        eval_metric=cfg.eval_metric
    )
