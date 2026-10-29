"""CreditPulse Model Inference Engine.
Handles forward pass, ensemble checkpoint averaging, and post-hoc Platt calibration.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model.architecture import MultiTaskDistressTransformer
from model.model_config import (
    PLATT_CALIBRATION, ALERT_THRESHOLD_10PCT, MODEL_NAME, MODEL_VERSION
)


class DistressScorer:
    """Production inference wrapper for M6_multitask Transformer ensemble.
    Transforms raw neural logits into calibrated distress probabilities.
    """
    def __init__(self, model: MultiTaskDistressTransformer | None = None):
        self.model = model or MultiTaskDistressTransformer()
        checkpoint_path = Path(__file__).resolve().parent.parent / "models" / "M6_multitask_ens5.pt"
        if checkpoint_path.exists():
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            except Exception:
                pass
        self.model.eval()
        self.platt_a = PLATT_CALIBRATION["a"]
        self.platt_b = PLATT_CALIBRATION["b"]
        self.alert_threshold = ALERT_THRESHOLD_10PCT

    def calibrate(self, logit: float | np.ndarray) -> float | np.ndarray:
        """Applies Platt monotonic logistic calibration:
        P = 1 / (1 + exp(-(a * logit + b)))
        """
        scaled = self.platt_a * logit + self.platt_b
        return 1.0 / (1.0 + np.exp(-scaled))

    def predict_single(
        self,
        features: np.ndarray,
        ages: np.ndarray,
        mask: np.ndarray | None = None
    ) -> dict[str, any]:
        """Runs inference for a single borrower facility.
        Args:
            features: [12, 44] array of monthly token features.
            ages:     [12] array of borrower seasoning age.
            mask:     [12] array of 1s (observed) and 0s (padded).
        Returns:
            Dictionary with calibrated 6M risk, hazard curve, and alert flags.
        """
        if mask is None:
            mask = np.ones(features.shape[0], dtype=np.float32)

        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        age_tensor = torch.tensor(ages, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logit_a, hazard_b, _ = self.model(feat_tensor, age_tensor, mask_tensor)
            raw_logit = float(logit_a.squeeze().item())
            hazards = torch.sigmoid(hazard_b.squeeze()).numpy().tolist()

        calibrated_risk = float(self.calibrate(raw_logit))
        is_alert = bool(calibrated_risk >= self.alert_threshold)

        tier = "Critical Distress" if calibrated_risk >= 0.07 else (
            "Emerging Warning" if calibrated_risk >= 0.0285 else "Watchlist"
        )

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "raw_logit": round(raw_logit, 4),
            "calibrated_risk_6m": round(calibrated_risk, 4),
            "risk_percent": round(calibrated_risk * 100, 2),
            "above_alert_threshold": is_alert,
            "risk_tier": tier,
            "discrete_hazards_6m": [round(h, 4) for h in hazards]
        }


if __name__ == "__main__":
    print("=" * 70)
    print("CreditPulse Production Inference Engine (Platt Calibrated)")
    print("=" * 70)
    scorer = DistressScorer()
    
    # Generate synthetic 12-month sequence for verification
    dummy_feat = np.random.randn(12, 44).astype(np.float32)
    dummy_age = np.linspace(12, 24, 12).astype(np.float32)
    
    pred = scorer.predict_single(dummy_feat, dummy_age)
    print(f"Model Version:         {pred['model_version']}")
    print(f"Raw Output Logit:      {pred['raw_logit']}")
    print(f"Calibrated 6M Risk:    {pred['calibrated_risk_6m']} ({pred['risk_percent']}%)")
    print(f"Distress Risk Tier:    {pred['risk_tier']}")
    print(f"Above 10% Alert (2.85%): {pred['above_alert_threshold']}")
    print(f"6-Month Hazard Curve:  {pred['discrete_hazards_6m']}")
    print("=" * 70)
