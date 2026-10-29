"""=============================================================================
CREDITPULSE M6: MULTI-TASK PRE-LN TEMPORAL TRANSFORMER (PROPOSED CHAMPION)
=============================================================================
Layer Trace:
  Input [B, 12, 44]
        ↓
  1. Input Projection: Linear(input_dim=44, d_model=32)
        ↓
  2. Embeddings: Learnable [CLS] Token + Sinusoidal Positional Encoding
        ↓
  3. 3 × Pre-LN Transformer Blocks (nhead=2, d_model=32, dim_feedforward=64, dropout=0.20)
        ↓
  4. Sequence Pooling: [CLS] token representation at index 0 [B, 32]
        ↓
  5. Shared Representation
        ├── Head A (Distress Head): Linear(32, 32) → GELU → Dropout → Linear(32, 1) + Platt Scaling
        └── Head B (Hazard Head)  : Linear(32, 32) → GELU → Dropout → Linear(32, 6)

Multi-Task Loss Combination:
  L_total = L_distress + 0.30 × L_hazard
=============================================================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ml_models.M6_multitask.config import M6Config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal token positions."""
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiTaskDistressTransformer(nn.Module):
    """CreditPulse Champion Architecture: Multi-Task Pre-LN Temporal Transformer strictly driven by M6Config."""
    def __init__(self, cfg: M6Config | None = None):
        super().__init__()
        self.cfg = cfg or M6Config()

        # 1. Input Linear Projection
        self.input_proj = nn.Linear(self.cfg.input_dim, self.cfg.d_model)

        # 2. Embeddings (Learnable [CLS] Token + Positional Encoding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoder = PositionalEncoding(d_model=self.cfg.d_model, max_len=32)
        self.dropout = nn.Dropout(self.cfg.dropout)

        # 3. 3 × Pre-LN Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)
        self.norm = nn.LayerNorm(self.cfg.d_model)

        # 4. Head A: Distress Prediction Head (Primary 6M / 12M Distress Logit)
        self.head_a = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.cfg.head_a_output_dim)
        )
        # Platt Scaling Calibration parameters: P = sigmoid(a * logit + b)
        self.register_buffer("platt_a", torch.tensor(0.1456))
        self.register_buffer("platt_b", torch.tensor(-3.3719))

        # 5. Head B: Hazard Prediction Head (6-Month Forward Discrete Hazard Logits)
        self.head_b = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.cfg.horizon)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Input: [Batch, 12, input_dim=44]
        B, T, _ = x.shape

        # 1. Input Projection
        proj = self.input_proj(x)  # [B, T, d_model=32]

        # 2. Add [CLS] Token & Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, proj], dim=1)  # [B, 13, d_model]
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)

        # 3. 3 × Pre-LN Transformer Blocks
        enc_out = self.encoder(tokens)
        enc_out = self.norm(enc_out)

        # 4. Sequence Pooling: extract shared [CLS] representation at index 0
        shared_rep = enc_out[:, 0, :]  # [B, d_model=32]

        # 5. Dual Prediction Heads
        # Head A: Distress Prediction
        logits_a = self.head_a(shared_rep).squeeze(-1)  # [B]
        calibrated_prob = torch.sigmoid(self.platt_a * logits_a + self.platt_b)

        # Head B: Monthly Hazard Prediction
        logits_b = self.head_b(shared_rep)  # [B, horizon=6]
        hazard_probs = torch.sigmoid(logits_b)

        return {
            "distress_logits": logits_a,
            "distress_prob": calibrated_prob,
            "hazard_logits": logits_b,
            "hazard_probs": hazard_probs,
            "shared_rep": shared_rep
        }
