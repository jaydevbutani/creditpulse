"""S2 Discrete-Time Hazard Survival Model Architecture.
Computes discrete monthly hazard probabilities h_t = P(T = t | T >= t, x) across horizon H=6.
Cumulative failure probability: F(H) = 1 - prod_{t=1}^H (1 - h_t).
Architecture is visibly driven by S2HazardConfig.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ml_models.S2_hazard.config import S2HazardConfig


class DiscreteTimeHazardNet(nn.Module):
    """Neural discrete-time hazard model for multi-period credit survival analysis."""
    def __init__(self, cfg: S2HazardConfig | None = None):
        super().__init__()
        self.cfg = cfg or S2HazardConfig()
        self.horizon = self.cfg.horizon
        
        # 1. Feature Projection driven by cfg.input_dim and cfg.hidden_dim
        self.feature_proj = nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.cfg.dropout)
        
        # 2. Multi-period discrete hazard output head
        self.hazard_head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.horizon)  # Logits for H monthly hazards
        )

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # x_seq: [B, 12, input_dim]
        h = self.relu(self.feature_proj(x_seq))  # [B, 12, hidden_dim]
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            pooled = (h * mask_exp).sum(dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1.0)
        else:
            pooled = h.mean(dim=1)  # [B, hidden_dim]

        hazard_logits = self.hazard_head(pooled)  # [B, horizon]
        hazard_probs = torch.sigmoid(hazard_logits)  # [B, horizon]

        # Cumulative distress probability over horizon:
        # P(Distress <= H) = 1 - prod_{t=1}^H (1 - h_t)
        survival_curve = torch.cumprod(1.0 - hazard_probs, dim=-1)
        cum_distress_prob = 1.0 - survival_curve[:, -1]

        return {
            "hazard_logits": hazard_logits,
            "hazard_probs": hazard_probs,
            "distress_prob": cum_distress_prob
        }
