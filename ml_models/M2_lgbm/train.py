"""Training and parameter tuning for M2 LightGBM.
Trace:
  M2LGBMConfig -> sweep over cfg.param_grid -> best params -> final model -> test metrics

Run via:
    python -m ml_models.M2_lgbm.train
"""
from ml_models.common.utils import set_seed, print_model_summary, save_metrics
from ml_models.common.splits import split_tabular_data
from ml_models.common.data import get_tabular_dataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.M2_lgbm.config import M2LGBMConfig
from ml_models.M2_lgbm.model import build_lgbm_classifier


def run_m2_lgbm():
    # 1. Config (Single source of truth)
    cfg = M2LGBMConfig()
    set_seed(cfg.seed)

    # 2. Data loading & partition
    X, y, _, _, feat_names = get_tabular_dataset()
    split = split_tabular_data(X, y, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)

    # 3. Genuine validation sweep
    best_params = cfg.param_grid[0]
    best_val_score = -1.0
    tuning_log = []

    print("[M2_lgbm] Sweeping hyperparameter grid...")
    for params in cfg.param_grid:
        clf = build_lgbm_classifier(cfg=cfg, params=params)
        clf.fit(split.X_train, split.y_train)
        val_probs = clf.predict_proba(split.X_val)[:, 1]
        val_metrics = calculate_classification_metrics(split.y_val, val_probs)
        val_score = val_metrics["pr_auc"] if isinstance(val_metrics["pr_auc"], (int, float)) else 0.0
        tuning_log.append({**params, "val_pr_auc": val_score})
        print(f"  depth={params['max_depth']} lr={params['learning_rate']} n_est={params['n_estimators']} | Val PR-AUC={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_params = params

    print(f"[M2_lgbm] Selected optimal params: {best_params} (Val PR-AUC={best_val_score:.4f})")

    # 4. Final model fitted with best params
    final_model = build_lgbm_classifier(cfg=cfg, params=best_params)
    final_model.fit(split.X_train, split.y_train)

    test_probs = final_model.predict_proba(split.X_test)[:, 1]
    test_metrics = calculate_classification_metrics(split.y_test, test_probs)
    test_metrics["best_params"] = best_params
    test_metrics["tuning_history"] = tuning_log

    print_model_summary(cfg.model_name, test_metrics, params=f"{best_params['n_estimators']} trees")
    save_metrics("ml_models/M2_lgbm/metrics.json", test_metrics)
    return test_metrics


if __name__ == "__main__":
    run_m2_lgbm()
