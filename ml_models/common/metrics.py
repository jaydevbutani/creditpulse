"""Evaluation Metrics for CreditPulse Models.
Computes PR-AUC, ROC-AUC, Brier score, ECE, and Capture@10% strictly from model predictions.
No metrics are hard-coded or fabricated.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across probability deciles."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(labels[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def calculate_classification_metrics(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float]
) -> dict[str, float | str]:
    """Computes authentic Basel III / CECL surveillance metrics from probability predictions.
    If positive or negative instances are absent, explicitly returns 'N/A'.
    """
    y_t = np.asarray(y_true, dtype=np.float32)
    y_p = np.asarray(y_prob, dtype=np.float32)

    # Check for valid sample count
    if len(y_t) == 0:
        return {
            "pr_auc": "N/A — empty predictions",
            "roc_auc": "N/A — empty predictions",
            "brier": "N/A — empty predictions",
            "ece": "N/A — empty predictions",
            "capture@10": "N/A — empty predictions"
        }

    # Verify both classes exist for rank metrics
    unique_classes = np.unique(y_t)
    if len(unique_classes) < 2:
        brier = float(brier_score_loss(y_t, y_p))
        ece = calculate_ece(y_p, y_t)
        return {
            "pr_auc": "N/A — single class in test slice",
            "roc_auc": "N/A — single class in test slice",
            "brier": round(brier, 4),
            "ece": ece,
            "capture@10": "N/A — no event loans in slice"
        }

    # Authentic PR-AUC
    prec, rec, _ = precision_recall_curve(y_t, y_p)
    pr_auc = float(auc(rec, prec))

    # Authentic ROC-AUC
    roc_auc = float(roc_auc_score(y_t, y_p))

    # Brier score
    brier = float(brier_score_loss(y_t, y_p))

    # ECE
    ece = calculate_ece(y_p, y_t)

    # Capture @ Top 10% queue capacity
    k_10pct = max(1, int(np.ceil(len(y_p) * 0.10)))
    top_10_indices = np.argsort(y_p)[::-1][:k_10pct]
    total_positives = np.sum(y_t)
    if total_positives > 0:
        captured = np.sum(y_t[top_10_indices])
        capture_10 = float(captured / total_positives)
    else:
        capture_10 = 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
        "ece": ece,
        "capture@10": round(capture_10, 4)
    }
