"""Training and regularisation parameter tuning for S1 Cox Proportional Hazards.
Trace:
  S1CoxConfig -> validation sweep over cfg.penalizer_grid -> best penalizer -> final model -> test metrics

Run via:
    python -m ml_models.S1_cox.train
"""
from ml_models.common.utils import set_seed, print_model_summary, save_metrics
from ml_models.common.splits import split_tabular_data
from ml_models.common.data import get_tabular_dataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.S1_cox.config import S1CoxConfig
from ml_models.S1_cox.model import CoxSurvivalBaseline


def run_s1_cox():
    # 1. Config (Single source of truth)
    cfg = S1CoxConfig()
    set_seed(cfg.seed)

    # 2. Data loading & partition
    X, y, durations, events, feat_names = get_tabular_dataset()
    split = split_tabular_data(
        X, y, durations=durations, events=events,
        test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed
    )

    # 3. Validation sweep over penalizer
    best_pen = cfg.penalizer_grid[0]
    best_val_score = -1.0
    tuning_log = []

    print("[S1_cox] Sweeping L2 penalizer...")
    for pen in cfg.penalizer_grid:
        try:
            m = CoxSurvivalBaseline(cfg=cfg, penalizer=pen)
            m.fit(split.X_train, split.durations_train, split.events_train)
            val_probs = m.predict_distress_probability(split.X_val, time_horizon=cfg.time_horizon)
            val_metrics = calculate_classification_metrics(split.y_val, val_probs)
            val_score = val_metrics["pr_auc"] if isinstance(val_metrics["pr_auc"], (int, float)) else 0.0
            tuning_log.append({"penalizer": pen, "val_pr_auc": val_score})
            print(f"  penalizer={pen:<5} | Val PR-AUC={val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_pen = pen
        except Exception as e:
            print(f"  penalizer={pen:<5} failed: {e}")

    print(f"[S1_cox] Selected optimal penalizer: {best_pen} (Val PR-AUC={best_val_score:.4f})")

    # 4. Final model with optimal penalizer
    final_model = CoxSurvivalBaseline(cfg=cfg, penalizer=best_pen)
    final_model.fit(split.X_train, split.durations_train, split.events_train)

    test_probs = final_model.predict_distress_probability(split.X_test, time_horizon=cfg.time_horizon)
    test_metrics = calculate_classification_metrics(split.y_test, test_probs)
    test_metrics["best_penalizer"] = best_pen
    test_metrics["concordance_index"] = round(float(final_model.cph.concordance_index_), 4)
    test_metrics["tuning_history"] = tuning_log

    num_params = len(final_model.selected_indices)
    print_model_summary(cfg.model_name, test_metrics, params=num_params)
    save_metrics("ml_models/S1_cox/metrics.json", test_metrics)
    return test_metrics


if __name__ == "__main__":
    run_s1_cox()
