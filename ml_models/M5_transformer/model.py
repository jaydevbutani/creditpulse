"""=============================================================================
CREDITPULSE BENCHMARK M5: SINGLE-TASK TEMPORAL TRANSFORMER
=============================================================================
Layer Trace:
  Input: 12 monthly observations × 44 behavioral features [B, 12, 44]
  → 1. Input Projection: Linear(input_dim=44, d_model=64)
  → 2. Positional / Sequence Tokens: Learnable [CLS] token + Sinusoidal Positional Encoding
  → 3. Transformer Encoder: 2 Pre-LN blocks (nhead=4, d_model=64, dim_feedforward=128, dropout=0.1)
  → 4. Sequence Pooling: [CLS] token extraction at index 0 [B, 64]
  → 5. Prediction Head: Linear(64, 32) → GELU → Dropout → Linear(32, 1) producing distress logit
=============================================================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ml_models.M5_transformer.config import M5TransformerConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding adding temporal sequence order."""
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


class SingleTaskDistressTransformer(nn.Module):
    """Single-task temporal Transformer classifier strictly configured via M5TransformerConfig."""
    def __init__(self, cfg: M5TransformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or M5TransformerConfig()

        # 1. Input Linear Projection
        self.input_proj = nn.Linear(self.cfg.input_dim, self.cfg.d_model)

        # 2. Positional / Sequence Representation ([CLS] Token + Sinusoidal Positional Encoding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoder = PositionalEncoding(d_model=self.cfg.d_model, max_len=32)
        self.dropout = nn.Dropout(self.cfg.dropout)

        # 3. Pre-LN Transformer Encoder Blocks
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

        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg.d_model, 32),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(32, self.cfg.output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Input: [Batch, 12, 44]
        B, T, _ = x.shape

        # 1. Project input features to latent d_model dimension
        proj = self.input_proj(x)  # [B, T, d_model]

        # 2. Prepend [CLS] token and add positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, proj], dim=1)  # [B, 1 + T, d_model]
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)

        # 3. Multi-head self-attention encoder blocks
        enc_out = self.encoder(tokens)
        enc_out = self.norm(enc_out)

        # 4. Sequence pooling: extract [CLS] token representation at index 0
        cls_rep = enc_out[:, 0, :]  # [B, d_model]

        # 5. Prediction head: scalar distress logit
        logits = self.classifier(cls_rep).squeeze(-1)  # [B]
        return logits
