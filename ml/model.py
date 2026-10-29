"""CreditPulse Multi-Task Transformer Architecture.
Implements the M6_multitask neural network:
- Input projection & sequence embedding with temporal position & loan age encodings
- Pre-LayerNorm Multi-Head Self-Attention layers
- Head A: 6-month binary distress classification
- Head B: Discrete-time survival hazard estimation over horizons +1m through +6m
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import (
    INPUT_DIM, D_MODEL, N_LAYERS, N_HEADS, FEEDFORWARD_MULT,
    DROPOUT, POOLING, HORIZON, MAX_POS_WINDOW
)

AGE_MIN, AGE_MAX = -24, 72


def age_index(age: torch.Tensor) -> torch.Tensor:
    return (age.clamp(AGE_MIN, AGE_MAX) - AGE_MIN).long()


class PreLNTransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer Block.
    Normalizes inputs *before* multi-head self-attention and feedforward layers,
    improving gradient stability during deep sequence training.
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 ff_mult: int = FEEDFORWARD_MULT, dropout: float = DROPOUT):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        # 1. Multi-head self-attention with pre-norm
        h = self.ln1(x)
        attn_out, weights = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + self.drop(attn_out)

        # 2. Feedforward with pre-norm
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, weights


class MultiTaskDistressTransformer(nn.Module):
    """M6_multitask: Dual-Head Temporal Transformer for Loan Distress Surveillance.
    Shared representation:
      [B, W, input_dim] -> Token Embedding -> Positional/Age Embed -> PreLN Transformer -> [CLS] Vector
    Task Heads:
      - Head A: Scalar logit for binary 6-month default distress
      - Head B: Vector of 6 logits for discrete-time monthly hazard rates
    """
    def __init__(
        self,
        n_features: int = INPUT_DIM,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
        horizon: int = HORIZON,
        pooling: str = POOLING
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.pooling = pooling

        # Feature projection layer (+1 for mask indicator flag)
        self.inp = nn.Sequential(
            nn.Linear(n_features + 1, d_model),
            nn.Dropout(dropout)
        )
        self.mask_emb = nn.Parameter(torch.zeros(d_model))

        # Temporal position & loan age embeddings
        self.pos = nn.Embedding(MAX_POS_WINDOW, d_model)
        self.age = nn.Embedding(AGE_MAX - AGE_MIN + 1, d_model)
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)

        # Pre-LN Transformer stack
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, FEEDFORWARD_MULT, dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

        # Multi-task output heads
        self.head_a = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.head_hz = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

        self.last_attn = None

    def encode(self, x: torch.Tensor, pad: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """Encodes [B, W, features] into a unified sequence embedding [B, d_model]."""
        B, W, _ = x.shape
        flag = torch.zeros_like(pad)
        h = self.inp(torch.cat([x, flag.unsqueeze(-1)], -1))

        # Add positional distance from month-end cutoff and loan age
        back = torch.arange(W - 1, -1, -1, device=x.device)
        h = h + self.pos(back.clamp(max=MAX_POS_WINDOW - 1)).unsqueeze(0) + self.age(age_index(age))

        # Prepend [CLS] token
        key_pad = pad.bool()
        cls_token = self.cls.expand(B, 1, -1)
        h = torch.cat([cls_token, h], dim=1)
        kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), key_pad], dim=1)

        # Pass through Transformer layers
        attn_weights = []
        for layer in self.layers:
            h, w = layer(h, kp)
            attn_weights.append(w)
        self.last_attn = attn_weights

        h = self.ln(h)
        # Pool [CLS] token representation at position 0
        return h[:, 0]

    def forward(self, x: torch.Tensor, pad: torch.Tensor, age: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass generating predictions for both Head A and Head B."""
        z = self.encode(x, pad, age)
        return {
            "logit_a": self.head_a(z).squeeze(-1),     # Head A: 6-month distress logit
            "hz_logits": self.head_hz(z),             # Head B: 6-month discrete hazards
            "latent_vector": z
        }
