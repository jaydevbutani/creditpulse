"""Training and evaluation pipeline for M0 Empirical Prior Baseline.
Trace:
  M0Config -> EmpiricalPriorBaseline(cfg) -> fit -> evaluate test -> save metrics

Run via:
    python -m ml_models.M0_prior.train
"""
from ml_models.common.utils import set_seed, print_model_summary, save_metrics
from ml_models.common.splits import split_tabular_data
from ml_models.common.data import get_tabular_dataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.M0_prior.config import M0Config
from ml_models.M0_prior.model import EmpiricalPriorBaseline


def run_m0():
    # 1. Config (Single source of truth)
    cfg = M0Config()
    set_seed(cfg.seed)

    # 2. Data loading & partition
    X, y, durations, events, feat_names = get_tabular_dataset()
    split = split_tabular_data(X, y, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)

    # 3. Model construction driven by config
    model = EmpiricalPriorBaseline(cfg=cfg)
    model.fit(split.X_train, split.y_train)

    # 4. Out-of-time evaluation
    y_pred_prob = model.predict_proba(split.X_test)[:, 1]
    metrics = calculate_classification_metrics(split.y_test, y_pred_prob)

    print_model_summary(cfg.model_name, metrics, params=0)
    save_metrics("ml_models/M0_prior/metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    run_m0()
