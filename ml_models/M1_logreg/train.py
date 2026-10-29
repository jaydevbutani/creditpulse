"""Training and validation sweep for M1 Logistic Regression.
Trace:
  M1Config -> validation sweep over cfg.param_grid_C -> best C -> final model -> test metrics

Run via:
    python -m ml_models.M1_logreg.train
"""
from ml_models.common.utils import set_seed, print_model_summary, save_metrics
from ml_models.common.splits import split_tabular_data
from ml_models.common.data import get_tabular_dataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.M1_logreg.config import M1Config
from ml_models.M1_logreg.model import build_logistic_regression


def run_m1():
    # 1. Config (Single source of truth)
    cfg = M1Config()
    set_seed(cfg.seed)

    # 2. Data loading & partition
    X, y, _, _, feat_names = get_tabular_dataset()
    split = split_tabular_data(X, y, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)

    # 3. Genuine Hyperparameter Tuning Sweep over C
    best_c = cfg.param_grid_C[0]
    best_val_score = -1.0
    tuning_log = []

    print("[M1] Sweeping inverse regularisation C...")
    for c_val in cfg.param_grid_C:
        candidate_model = build_logistic_regression(cfg=cfg, C=c_val)
        candidate_model.fit(split.X_train, split.y_train)
        val_probs = candidate_model.predict_proba(split.X_val)[:, 1]
        val_metrics = calculate_classification_metrics(split.y_val, val_probs)
        val_score = val_metrics["pr_auc"] if isinstance(val_metrics["pr_auc"], (int, float)) else 0.0
        tuning_log.append({"C": c_val, "val_pr_auc": val_score})
        print(f"  C={c_val:<6} | Val PR-AUC={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_c = c_val

    print(f"[M1] Best hyperparameter selected: C={best_c} (Val PR-AUC={best_val_score:.4f})")

    # 4. Final model fitted with selected C driven by cfg
    final_model = build_logistic_regression(cfg=cfg, C=best_c)
    final_model.fit(split.X_train, split.y_train)

    test_probs = final_model.predict_proba(split.X_test)[:, 1]
    test_metrics = calculate_classification_metrics(split.y_test, test_probs)
    test_metrics["best_C"] = best_c
    test_metrics["tuning_history"] = tuning_log

    num_params = split.X_train.shape[1] + 1
    print_model_summary(cfg.model_name, test_metrics, params=num_params)
    save_metrics("ml_models/M1_logreg/metrics.json", test_metrics)
    return test_metrics


if __name__ == "__main__":
    run_m1()
