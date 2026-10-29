"""M2 LightGBM Model Architecture.
Gradient boosted decision tree ensemble with leaf-wise tree growth.
Architecture is visibly driven by M2LGBMConfig.
"""
from __future__ import annotations
import lightgbm as lgb
from ml_models.M2_lgbm.config import M2LGBMConfig


def build_lgbm_classifier(
    cfg: M2LGBMConfig,
    params: dict | None = None
) -> lgb.LGBMClassifier:
    """Builds a LightGBM Classifier strictly driven by M2LGBMConfig."""
    p = params or cfg.param_grid[0]
    return lgb.LGBMClassifier(
        max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.05),
        n_estimators=p.get("n_estimators", 50),
        random_state=cfg.seed,
        class_weight=cfg.class_weight,
        verbose=-1
    )
