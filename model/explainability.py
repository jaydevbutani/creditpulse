"""CreditPulse Model Explainability & Integrated Gradients Attribution.
Computes path-integrated gradients to identify risk-increasing drivers vs protective offsets.
"""
from __future__ import annotations

import numpy as np
import torch

from model.architecture import MultiTaskDistressTransformer
from model.preprocessing import FEATURE_NAMES, FEATURE_REGISTRY


def compute_integrated_gradients(
    model: MultiTaskDistressTransformer,
    features: np.ndarray,
    ages: np.ndarray,
    mask: np.ndarray,
    steps: int = 20
) -> np.ndarray:
    """Computes integrated gradients across linear interpolation path from zero baseline.
    Returns feature attributions of shape [W, INPUT_DIM].
    """
    model.eval()
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    age_tensor = torch.tensor(ages, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    baseline = torch.zeros_like(feat_tensor)
    alphas = torch.linspace(0, 1, steps)

    accumulated_grads = torch.zeros_like(feat_tensor)

    for alpha in alphas:
        interpolated = baseline + alpha * (feat_tensor - baseline)
        interpolated.requires_grad_(True)
        logit, _, _ = model(interpolated, age_tensor, mask_tensor)
        grad = torch.autograd.grad(logit, interpolated)[0]
        accumulated_grads += grad

    avg_grads = accumulated_grads / steps
    attributions = (feat_tensor - baseline) * avg_grads
    return attributions.squeeze(0).detach().numpy()


def extract_top_drivers(attributions: np.ndarray, top_k: int = 5) -> list[dict]:
    """Ranks feature blocks into human-understandable risk drivers."""
    total_attr = np.sum(attributions, axis=0) # sum across 12-month tokens

    drivers = []
    abs_sorted_indices = np.argsort(np.abs(total_attr))[::-1]

    for idx in abs_sorted_indices[:top_k]:
        score = float(total_attr[idx])
        if 0 <= idx < len(FEATURE_NAMES):
            name = FEATURE_NAMES[idx].replace("_", " ").title()
            desc = FEATURE_REGISTRY[idx]["viva_meaning"]
        else:
            name, desc = f"Feature #{idx}", "Longitudinal sequence feature"
        direction = "increasing" if score > 0 else "protective"
        drivers.append({
            "feature": name,
            "description": desc,
            "contribution": round(score, 4),
            "direction": direction,
            "is_risk_increasing": score > 0
        })

    return drivers


if __name__ == "__main__":
    print("=" * 70)
    print("CreditPulse Path-Integrated Gradients Feature Attribution")
    print("=" * 70)
    model = MultiTaskDistressTransformer()
    dummy_feat = np.random.randn(12, 44).astype(np.float32)
    dummy_age = np.linspace(12, 24, 12).astype(np.float32)
    dummy_mask = np.ones(12, dtype=np.float32)

    attrs = compute_integrated_gradients(model, dummy_feat, dummy_age, dummy_mask, steps=15)
    drivers = extract_top_drivers(attrs, top_k=5)
    
    print("Top Behavioral Risk Drivers (Path-Integrated Gradients):")
    for d in drivers:
        sign = "+" if d["is_risk_increasing"] else ""
        print(f"  • {d['feature']:30s}: {sign}{d['contribution']:+.4f} ({d['direction']})")
    print("=" * 70)
